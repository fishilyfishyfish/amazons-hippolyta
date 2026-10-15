// UI thread: rendering and input only. All rules live in the engine worker.

const FILES = 'abcdefghij';
const sqName = (sq) => FILES[sq % 10] + (Math.floor(sq / 10) + 1);
// The engine opponent: the strongest setting (MCTS, 2 s per move, all
// workers). Friendly games always play against it.
const ENGINE = 'mcts:2000';
const VALUE_SCALE = 16;   // same squash as the engine: tanh(score / 16)

const els = {
  board: document.getElementById('board'),
  status: document.getElementById('status'),
  hint: document.getElementById('hint'),
  moves: document.getElementById('moves'),
  opponent: document.getElementById('opponent'),
  level: document.getElementById('level'),
  levelLabel: document.getElementById('levelLabel'),
  color: document.getElementById('color'),
  newGame: document.getElementById('newGame'),
  undo: document.getElementById('undo'),
  cancel: document.getElementById('cancel'),
  heat: document.getElementById('heat'),
  vfill: document.getElementById('vfill'),
  vlabel: document.getElementById('vlabel'),
  vbar: document.getElementById('vbar'),
  boardArea: document.getElementById('boardArea'),
  annot: document.getElementById('annot'),
  viewPrev: document.getElementById('viewPrev'),
  viewNext: document.getElementById('viewNext'),
  viewInfo: document.getElementById('viewInfo'),
  drawBtn: document.getElementById('drawBtn'),
  clearDraw: document.getElementById('clearDraw'),
  evPrev: document.getElementById('evPrev'),
  evNext: document.getElementById('evNext'),
  aiInfo: document.getElementById('aiInfo'),
  analysisPanel: document.getElementById('analysisPanel'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  showBest: document.getElementById('showBest'),
  posInfo: document.getElementById('posInfo'),
  graph: document.getElementById('graph'),
  explain: document.getElementById('explain'),
  analyzeAll: document.getElementById('analyzeAll'),
  exportKibo: document.getElementById('exportKibo'),
  importKibo: document.getElementById('importKibo'),
  kiboText: document.getElementById('kiboText'),
  kiboMsg: document.getElementById('kiboMsg'),
  modes: [...document.querySelectorAll('.modes [data-mode]')],
  opponentLabel: document.getElementById('opponentLabel'),
  colorLabel: document.getElementById('colorLabel'),
  coachPanel: document.getElementById('coachPanel'),
  coachBody: document.getElementById('coachBody'),
  hintBtn: document.getElementById('hintBtn'),
  coachLegend: document.getElementById('coachLegend'),
  analysisLegend: document.getElementById('analysisLegend'),
  bookPanel: document.getElementById('bookPanel'),
  bookBack: document.getElementById('bookBack'),
  bookStart: document.getElementById('bookStart'),
  bookInfo: document.getElementById('bookInfo'),
  bookList: document.getElementById('bookList'),
};

// ---- worker RPC -----------------------------------------------------------

// Each worker owns its own engine instance. `engine` holds the real game;
// helpers only search copies of the position (root parallelism).
function createWorker(onProgress) {
  const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  let nextId = 1;
  const pending = new Map();
  w.onmessage = (e) => {
    const { id, error, progress } = e.data;
    if (progress) {
      onProgress(progress);
      return;
    }
    const p = pending.get(id);
    if (!p) return;
    pending.delete(id);
    if (error) p.reject(new Error(error));
    else p.resolve(e.data);
  };
  w.onerror = (e) => showError(e.message || '워커 오류');
  return (type, args = {}) => {
    const id = nextId++;
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
      w.postMessage({ id, type, ...args });
    });
  };
}

// Searchers = engine worker + helpers, one per spare core (max 6).
const SEARCHERS = Math.max(1, Math.min(6, (navigator.hardwareConcurrency || 2) - 1));
const progress = new Array(SEARCHERS).fill(null);
let liveSearch = false; // show progress only for the AI's own search
const onProgress = (i) => (info) => {
  if (!liveSearch) return;
  progress[i] = info;
  showAiInfo(combineInfos(progress.filter(Boolean)), true);
};
const call = createWorker(onProgress(0));
const helpers = [];
function getHelpers(min = 0) {
  const want = Math.max(min, SEARCHERS - 1);
  while (helpers.length < want) helpers.push(createWorker(onProgress(helpers.length + 1)));
  return helpers.slice(0, want);
}

// Merge root statistics of independent searches: visits add up, values are
// visit-weighted. A proven result from any searcher wins outright.
function combineInfos(infos) {
  const byMove = new Map();
  let iterations = 0;
  let proven = 0;
  let provenMove = null;
  for (const info of infos) {
    iterations += info.iterations;
    if (info.proven && !proven) {
      proven = info.proven;
      provenMove = info.candidates[0];
    }
    for (const c of info.candidates) {
      const o = byMove.get(c.move);
      if (o) {
        if (c.visits) {
          o.qsum += c.q * c.visits;
          o.visits += c.visits;
        }
      } else {
        byMove.set(c.move, { ...c, qsum: c.visits ? c.q * c.visits : 0 });
      }
    }
  }
  const candidates = [...byMove.values()].sort((a, b) => b.visits - a.visits || b.prior - a.prior);
  for (const c of candidates) c.q = c.visits ? c.qsum / c.visits : null;
  const best = candidates[0];
  const value = proven || (best && best.q != null ? best.q : 0);
  // The engine worker's PV, re-rooted only if it starts with the chosen move.
  const pv = infos[0] && best && infos[0].pv[0] === best.move ? infos[0].pv : best ? [best.move] : [];
  return {
    iterations,
    searchers: infos.length,
    proven,
    value,
    pv,
    candidates,
    best: proven ? provenMove : best,
    ms: infos.every((i) => i.ms != null) ? Math.max(...infos.map((i) => i.ms)) : null,
  };
}

// ---- state ----------------------------------------------------------------

let state = null;        // last snapshot from the worker
let tree = new Map();    // from -> Map(to -> Set(arrow))
let sel = { from: -1, to: -1 };
let busy = false;        // waiting on the worker
let epoch = 0;           // bumps on new game / undo to drop stale AI turns
// mode: 'play' (engine or human opponent), 'coach' (friendly game against
// the engine), 'analysis' (both sides by hand), 'book' (opening book).
const settings = { mode: 'play', opponent: 'mcts:2000', human: 0 };
try {
  const m = localStorage.getItem('mode');
  if (['play', 'coach', 'analysis', 'book'].includes(m)) settings.mode = m;
} catch {}
// Difficulty 1–5 of the engine opponent (play and coach). 1–3 are
// human-like players, 4 a short search, 5 the full engine. The per-game
// seed gives each game of a human-like level its own style.
const LEVEL_NAMES = ['', '입문', '초급', '중급', '상급', '최강'];
let gameSeed = Math.floor(Math.random() * 2 ** 32);
try {
  const l = localStorage.getItem('level');
  if (l && els.level.querySelector(`option[value="${l}"]`)) els.level.value = l;
} catch {}
const level = () => Number(els.level.value) || 5;
let lastAiInfo = null;

// Every position the worker reported, by history key, so earlier positions
// can be shown again without asking the engine (← → in play/coach/book).
const snapshots = new Map();
// Ply being looked at (null = the live position). Viewing never changes
// the game; any new move returns to the live position.
let viewPly = null;

function setState(s) {
  const moved = !state || s.history.length !== state.history.length || !sameHistory(s.history, state.history);
  if (moved) {
    bookSel = 0;
    viewPly = null;
    annotations = [];
  }
  snapshots.set(hkey(s.history), s);
  state = s;
  tree = new Map();
  const L = s.legal;
  for (let i = 0; i < L.length; i += 3) {
    let tos = tree.get(L[i]);
    if (!tos) tree.set(L[i], (tos = new Map()));
    let arrows = tos.get(L[i + 1]);
    if (!arrows) tos.set(L[i + 1], (arrows = new Set()));
    arrows.add(L[i + 2]);
  }
  sel = { from: -1, to: -1 };
}

const isAnalysis = () => settings.mode === 'analysis';
const isBook = () => settings.mode === 'book';
const isHumanTurn = () =>
  state && !state.over && (settings.opponent === 'human' || isAnalysis() || isBook() || state.side === settings.human);

async function run(type, args) {
  busy = true;
  render();
  try {
    const r = await call(type, args);
    setState(r.state);
    return r.result;
  } catch (err) {
    showError(err.message);
  } finally {
    busy = false;
    render();
  }
}

async function maybeAiTurn() {
  if (!state || state.over || isHumanTurn()) return;
  if (level() < 5) return levelAiTurn();
  const my = epoch;
  const ms = settings.opponent.split(':')[1];
  busy = true;
  render();
  const before = Array.from(state.history);
  liveSearch = true;
  const info = await parallelSearch(Number(ms));
  liveSearch = false;
  if (my !== epoch) return;
  lastAiInfo = info;
  showAiInfo(info, false);
  if (!info || !info.best) {
    busy = false;
    render();
    return;
  }
  // The AI's search of this position is also the "after" analysis of the
  // human's last move.
  analysisCache.set(hkey(before), info);
  await run('play', { move: info.best.sq });
  if (my !== epoch) return;
  if (coachOn() && isHumanTurn()) coachPrepare();
  if (coachOn()) coachAfterAi(before, info, info.best.sq, my);
  maybeAiTurn();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// A weaker opponent (levels 1–4): the worker picks the move at once; a short
// pause keeps it from answering instantly. In a friendly game the position
// still needs a full-strength analysis (the "after" of the human's move and
// the "before" of the AI's), which the helper workers run meanwhile.
async function levelAiTurn() {
  const my = epoch;
  const lv = level();
  busy = true;
  render();
  const before = Array.from(state.history);
  if (coachOn()) analyzeInBackground(before);
  els.aiInfo.textContent = `상대: 난이도 ${lv} · ${LEVEL_NAMES[lv]}`;
  const think = 500 + Math.random() * (lv >= 4 ? 1200 : 900);
  let move;
  try {
    [move] = await Promise.all([
      call('levelMove', { level: lv, gameSeed, moveSeed: Math.floor(Math.random() * 2 ** 32) }).then((r) => r.result),
      sleep(think),
    ]);
  } catch (err) {
    showError(err.message);
  }
  if (my !== epoch) return;
  if (!move || move.length !== 3) {
    busy = false;
    render();
    return;
  }
  await run('play', { move });
  if (my !== epoch) return;
  if (coachOn() && isHumanTurn()) coachPrepare();
  if (coachOn()) coachAfterAi(before, null, move, my);
  maybeAiTurn();
}

// Searches `history` (default: the current position) with every searcher.
// The engine worker only joins when the position is its own current game,
// so analysing an earlier position never disturbs the game.
async function parallelSearch(ms, history = null, helpersOnly = false) {
  progress.fill(null);
  const current = !helpersOnly && (history === null || sameHistory(history, state.history));
  if (history === null) history = Array.from(state.history);
  const jobs = current ? [call('search', { ms, seed: 0, jitter: 0 })] : [];
  getHelpers(current ? 0 : 1).forEach((h, i) =>
    jobs.push(
      h('search', { ms, history, seed: 1 + i + Math.floor(Math.random() * 1e9), jitter: current || i > 0 ? 0.5 : 0 }),
    ),
  );
  try {
    const results = await Promise.all(jobs);
    return combineInfos(results.map((r) => r.result).filter(Boolean));
  } catch (err) {
    showError(err.message);
    return null;
  }
}

const pct = (v) => Math.round(50 * (v + 1)) + '%';
const sameHistory = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);

function showAiInfo(info, live) {
  if (!info) {
    els.aiInfo.textContent = '';
    return;
  }
  const head = live ? 'MCTS 탐색 중' : 'MCTS';
  const par = info.searchers > 1 ? ` ×${info.searchers}스레드` : '';
  const time = info.ms != null ? ` (${(info.ms / 1000).toFixed(1)}초)` : '';
  const verdict = info.proven
    ? ` · ${info.proven > 0 ? '백' : '흑'} 필승 확인`
    : ` · 백 승률 추정 ${pct(info.value)}`;
  const pv = info.pv.length ? ` · 예상 수순: ${info.pv.slice(0, 4).join(' ')}` : '';
  els.aiInfo.textContent = `${head} ${info.iterations.toLocaleString()}회${par}${time}${verdict}${pv}`;
}

// ---- board DOM ------------------------------------------------------------

const cellEls = [];
function buildBoard() {
  els.board.textContent = '';
  for (let i = 0; i < 100; i++) {
    const d = document.createElement('button');
    d.type = 'button';
    d.className = 'cell';
    d.addEventListener('click', () => onCell(Number(d.dataset.sq)));
    // Numbered badge for hinted moves: 1 from, 2 to, 3 arrow.
    const mark = document.createElement('span');
    mark.className = 'hm';
    mark.setAttribute('aria-hidden', 'true');
    d.appendChild(mark);
    cellEls.push(d);
    els.board.appendChild(d);
  }
}

// Visual position i -> square, with the human's side at the bottom.
function squareAt(i) {
  const vr = Math.floor(i / 10), vc = i % 10;
  const flip = isFlipped();
  const row = flip ? vr : 9 - vr;
  const col = flip ? 9 - vc : vc;
  return row * 10 + col;
}

// The position on screen: the live one, or an earlier one while viewing.
function shownState() {
  if (viewPly === null || !state) return state;
  return snapshots.get(hkey(state.history.slice(0, 3 * viewPly))) || state;
}
const viewing = () => viewPly !== null;

function render() {
  if (!state) return;
  const st = shownState();
  const cells = Array.from(st.cells);
  // Preview the amazon on its destination while choosing the arrow.
  if (sel.to >= 0) {
    cells[sel.to] = cells[sel.from];
    cells[sel.from] = 0;
  }
  const H = st.history;
  const last = H.length >= 3 ? H.slice(H.length - 3) : null;
  let targets = null, targetClass = '';
  if (sel.to >= 0) {
    targets = tree.get(sel.from).get(sel.to);
    targetClass = 't-arrow';
  } else if (sel.from >= 0) {
    targets = tree.get(sel.from);
    targetClass = 't-move';
  }
  const human = isHumanTurn() && !busy && !viewing();
  const flip = isFlipped();
  const ex = (isAnalysis() || viewing() || coachOn()) && sel.from < 0 ? shownExplanation() : null;
  const gained = new Set(ex ? ex.gained : []);
  const lost = new Set(ex ? ex.lost : []);
  const hintMove = sel.from < 0 && !viewing()
    ? isBook() ? bookHintMove() : (isAnalysis() ? showBest : coachHint) ? currentBestMove() : null
    : null;
  const heatOn = els.heat.checked;

  for (let i = 0; i < 100; i++) {
    const sq = squareAt(i);
    const el = cellEls[i];
    el.dataset.sq = sq;
    const r = Math.floor(sq / 10), c = sq % 10;
    const cls = ['cell', (r + c) % 2 === 0 ? 'dark' : 'light'];
    const v = cells[sq];
    if (heatOn && v === 0 && sq !== sel.from && sq !== sel.to) {
      const h = heatFor(sq);
      if (h) {
        cls.push('heat', h.cls);
        el.style.setProperty('--heat-a', h.amt);
      }
    }
    if (v === 1) cls.push('white');
    else if (v === 2) cls.push('black');
    else if (v === 3) cls.push('arrow');
    if (last && sel.from < 0) {
      if (sq === last[0]) cls.push('last-from');
      if (sq === last[1]) cls.push('last-to');
      if (sq === last[2]) cls.push('last-arrow');
    }
    if (sq === sel.from && sel.to < 0) cls.push('selected');
    if (sq === sel.to) cls.push('selected');
    if (sq === sel.from && sel.to >= 0) cls.push('vacated');
    if (targets && targets.has(sq)) cls.push('target', targetClass);
    if (v === 0 && gained.has(sq)) cls.push('chg-gain');
    if (v === 0 && lost.has(sq)) cls.push('chg-loss');
    let mark = '';
    if (hintMove) {
      // The arrow may land on the vacated start square: show both numbers.
      const n = [];
      if (sq === hintMove[0]) { cls.push('hint-from'); n.push('1'); }
      if (sq === hintMove[1]) { cls.push('hint-to', st.side === 0 ? 'ghost-w' : 'ghost-b'); n.push('2'); }
      if (sq === hintMove[2]) {
        // Arrow back onto the start square: keep the amazon drawn there.
        if (sq !== hintMove[0]) cls.push('hint-arrow');
        n.push('3');
      }
      mark = n.join('·');
    }
    if (el.firstChild.textContent !== mark) el.firstChild.textContent = mark;
    if (human && sel.from < 0 && tree.has(sq)) cls.push('movable');
    el.className = cls.join(' ');
    // Coordinates on the edge squares.
    const vr = Math.floor(i / 10), vc = i % 10;
    el.dataset.file = vr === 9 ? FILES[c] : '';
    el.dataset.rank = vc === 0 ? String(r + 1) : '';
    el.setAttribute('aria-label', sqName(sq) + labelFor(v));
  }
  els.board.classList.toggle('flipped', flip);
  els.boardArea.classList.toggle('viewing', viewing());
  renderStatus();
  renderEvalBar(st);
  renderMoves();
  renderAnalysis();
  renderCoach();
  renderBook();
  renderAnnotations();
  renderViewControls();
  els.undo.disabled = busy || undoCount() === 0;
  els.cancel.disabled = sel.from < 0;
}

// Territory by queen distance: who reaches the empty square first, and how
// clearly (distance gap).
function heatFor(sq) {
  const st = shownState();
  const dw = st.dist[sq], db = st.dist[100 + sq];
  if (dw === 255 && db === 255) return null;
  if (dw === db) return { cls: 'h-tie', amt: '30%' };
  const gap = Math.abs(dw - db);
  const amt = dw === 255 || db === 255 || gap >= 3 ? '85%' : gap === 2 ? '62%' : '42%';
  return { cls: dw < db ? 'h-white' : 'h-black', amt };
}

// White's winning chances for a position: the engine's search if there is
// one (or the search of the previous position when its best move was
// played), otherwise the static evaluation. Always from White's side, like
// a chess.com evaluation bar.
function whiteChance(st) {
  if (st.over) return st.winner === 0 ? 1 : 0;
  const info = analysisCache.get(hkey(st.history));
  if (info) return info.proven ? (info.proven > 0 ? 1 : 0) : (info.value + 1) / 2;
  const H = st.history;
  if (H.length >= 3) {
    const parent = analysisCache.get(hkey(H.slice(0, H.length - 3)));
    if (parent && parent.best && sameHistory(parent.best.sq, H.slice(H.length - 3))) {
      return parent.proven ? (parent.proven > 0 ? 1 : 0) : (parent.value + 1) / 2;
    }
  }
  return (Math.tanh(st.evaluation.score / VALUE_SCALE) + 1) / 2;
}

function renderEvalBar(st) {
  const p = whiteChance(st);
  const flip = isFlipped();
  // White fills from White's edge of the board (the bottom unless flipped).
  els.vfill.style.height = `${(100 * p).toFixed(1)}%`;
  els.vbar.classList.toggle('flip', flip);
  const lead = p >= 0.5 ? 0 : 1;
  const shownPct = Math.round(100 * (lead === 0 ? p : 1 - p));
  els.vlabel.textContent = `${shownPct}`;
  els.vlabel.className = `vlabel ${lead === 0 ? 'w' : 'b'}`;
  els.vbar.setAttribute('aria-label', `평가: 백 승률 ${Math.round(100 * p)}%`);
  els.vbar.title = `백 승률 ${Math.round(100 * p)}% · 흑 승률 ${Math.round(100 * (1 - p))}%`;
}

function labelFor(v) {
  return v === 1 ? ' 백 아마존' : v === 2 ? ' 흑 아마존' : v === 3 ? ' 화살' : '';
}

const colorName = (c) => (c === 0 ? '백' : '흑');

function renderStatus() {
  const s = state;
  let text, hint = '';
  if (viewing()) {
    els.status.textContent = `${viewPly}수째 국면을 보는 중`;
    els.hint.textContent = '→ 키나 ▶로 넘기고, 마지막 국면에서 다시 둘 수 있어요.';
    return;
  }
  if (s.over) {
    text = `게임 종료 — ${colorName(s.winner)} 승리 (${colorName(s.side)}이 둘 곳이 없음)`;
  } else if (isHumanTurn()) {
    const who = settings.opponent === 'human' || isAnalysis() ? '' : ' (당신)';
    text = `${colorName(s.side)} 차례${who} · 합법수 ${s.legal.length / 3}개`;
    hint = sel.to >= 0 ? '화살을 쏠 칸을 고르세요.'
      : sel.from >= 0 ? '이동할 칸을 고르세요.'
      : '움직일 아마존을 고르세요.';
  } else {
    text = `${colorName(s.side)} (엔진) 생각 중…`;
  }
  if (isAnalysis() && analyzingNow) hint = '분석 중…';
  els.status.textContent = text;
  els.hint.textContent = hint;
}

function renderMoves() {
  if (isAnalysis()) return renderLine();
  const n = state.notation;
  const rows = [];
  const shown = shownPly();
  const cell = (j) => {
    if (j >= n.length) return '<td></td>';
    const ex = explainCache.get(hkey(state.history.slice(0, 3 * (j + 1))));
    const mark = ex ? GRADE_MARK[ex.grade] : '';
    const cls = ['mv', j === n.length - 1 ? 'cur' : '', j === shown - 1 ? 'shown' : '', j >= shown ? 'ahead' : '', ex ? `g-${ex.grade}` : ''].join(' ');
    return `<td class="${cls}" data-ply="${j + 1}">${n[j]}${mark ? `<span class="mark">${mark}</span>` : ''}</td>`;
  };
  for (let i = 0; i < n.length; i += 2) {
    rows.push(`<tr><td>${i / 2 + 1}</td>${cell(i)}${cell(i + 1)}</tr>`);
  }
  const html = rows.join('');
  if (els.moves.innerHTML !== html) {
    els.moves.innerHTML = html;
    const wrap = els.moves.closest('.moves-wrap');
    wrap.scrollTop = wrap.scrollHeight;
  }
}

function showError(msg) {
  els.status.textContent = '오류: ' + msg;
  console.error(msg);
}

// ---- input ----------------------------------------------------------------

function onCell(sq) {
  if (drawMode || suppressClick) return;
  if (viewing()) {
    // Tapping the board while looking back returns to the live position.
    viewPly = null;
    render();
    return;
  }
  if (!isHumanTurn() || busy) return;
  if (sel.to >= 0) {
    // Arrow stage.
    if (tree.get(sel.from).get(sel.to).has(sq)) {
      const move = [sel.from, sel.to, sq];
      if (isAnalysis()) playInAnalysis(move);
      else {
        coachHint = false;
        const before = Array.from(state.history);
        run('play', { move }).then(() => {
          if (state.over && coachOn()) coachFinal(before.concat(move));
          maybeAiTurn();
        });
      }
    } else if (sq === sel.to) {
      sel.to = -1; // tap the moved amazon again to take the step back
      render();
    }
    return;
  }
  if (sel.from >= 0 && tree.get(sel.from).has(sq)) {
    sel.to = sq;
  } else if (tree.has(sq) && sq !== sel.from) {
    sel.from = sq;
  } else {
    sel.from = -1;
  }
  render();
}

// Plies to take back so that it is the human's turn again, having undone at
// least one human move. In two-player mode, one ply.
function undoCount() {
  if (!state) return 0;
  const plies = state.history.length / 3;
  if (settings.opponent === 'human' || isAnalysis() || isBook()) return plies > 0 ? 1 : 0;
  // Ply i (0-based) was played by colour i % 2.
  for (let i = plies - 1; i >= 0; i--) {
    if (i % 2 === settings.human) return plies - i;
  }
  return 0;
}

els.undo.addEventListener('click', async () => {
  if (isAnalysis()) return navigate(state.history.length / 3 - 1);
  const count = undoCount();
  if (!count) return;
  epoch++;
  analysisToken++;
  await run('undo', { count });
  if (coachOn() && isHumanTurn()) coachPrepare();
});

els.heat.addEventListener('change', () => {
  try { localStorage.setItem('heat', els.heat.checked ? '1' : ''); } catch {}
  render();
});
try { els.heat.checked = localStorage.getItem('heat') === '1'; } catch {}

els.cancel.addEventListener('click', () => {
  sel = { from: -1, to: -1 };
  render();
});

async function newGame() {
  settings.opponent = isAnalysis() ? 'analysis' : isBook() ? 'book' : settings.mode === 'coach' ? ENGINE : els.opponent.value;
  settings.human = Number(els.color.value);
  gameSeed = Math.floor(Math.random() * 2 ** 32);
  for (const b of els.modes) {
    const on = b.dataset.mode === settings.mode;
    b.classList.toggle('on', on);
    b.setAttribute('aria-selected', String(on));
  }
  els.opponentLabel.hidden = settings.mode !== 'play';
  els.levelLabel.hidden = !(settings.mode === 'coach' || (settings.mode === 'play' && settings.opponent !== 'human'));
  els.colorLabel.hidden = isAnalysis() || isBook() || settings.opponent === 'human';
  els.bookPanel.hidden = !isBook();
  lastAiInfo = null;
  showAiInfo(null);
  els.analysisPanel.hidden = !isAnalysis();
  els.undo.textContent = isAnalysis() ? '한 수 뒤로' : '무르기';
  line = [];
  analysisCache.clear();
  explainCache.clear();
  analysisToken++;
  epoch++;
  await run('newGame');
  coachHint = false;
  bookSel = 0;
  if (isAnalysis()) ensureAnalyzed();
  else if (isBook()) loadBook();
  else {
    if (coachOn() && isHumanTurn()) coachPrepare();
    maybeAiTurn();
  }
}

els.newGame.addEventListener('click', newGame);
els.opponent.addEventListener('change', newGame);
// A new level applies from the opponent's next move; the game goes on.
els.level.addEventListener('change', () => {
  try { localStorage.setItem('level', els.level.value); } catch {}
  els.level.blur();
});
els.color.addEventListener('change', newGame);

document.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return;
  if (e.key === 'ArrowLeft') { e.preventDefault(); stepView(-1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepView(1); }
  else if (e.key === 'Home') { e.preventDefault(); goToPly(0); }
  else if (e.key === 'End') { e.preventDefault(); goToPly(Infinity); }
  else if (e.key === 'Escape') { if (drawMode) setDrawMode(false); else els.cancel.click(); }
  else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    els.undo.click();
  }
});


// ---- analysis mode --------------------------------------------------------
//
// The user plays both sides. Every position reached is searched once
// (ANALYSIS_MS, all workers); a move is then explained from the search of
// the position before it (best move, win chance with best play) and the one
// after it (win chance after the move, the opponent's best reply).

const ANALYSIS_MS = 1000;
let line = [];                    // full move line [[f,t,a], ...] (may run past the current ply)
const analysisCache = new Map();  // history key -> combined search info
const explainCache = new Map();   // history key (through the move) -> explanation
let analysisToken = 0;
let analyzingNow = false;
let showBest = false;
const GRADE_MARK = { best: '!', good: '', inaccuracy: '?!', mistake: '?', blunder: '??' };
const GRADE_LABEL = { best: '최선의 수', good: '좋은 수', inaccuracy: '아쉬운 수', mistake: '실수', blunder: '큰 실수' };
const GRADE_HEAD = {
  best: '엔진이 고른 수와 같은 값어치예요.',
  good: '엔진 최선과 거의 차이 없어요.',
  inaccuracy: '나쁘지 않지만 더 나은 수가 있었어요.',
  mistake: '형세가 상대 쪽으로 기울었어요.',
  blunder: '형세가 크게 기울었어요.',
};
const explaining = new Set(); // history keys whose move is being explained

const flat = (moves) => moves.flat();
const hkey = (hist) => Array.from(hist).join(',');
const moveName = (m) => `${sqName(m[0])}-${sqName(m[1])}(${sqName(m[2])})`;
const isFlipped = () => !isAnalysis() && !isBook() && settings.opponent !== 'human' && settings.human === 1;

function plyOf() {
  return state.history.length / 3;
}

// Mover's win probability from a combined search info (White's view).
function moverP(info, mover) {
  const v = info.proven ? info.proven : info.value;
  const pw = (v + 1) / 2;
  return mover === 0 ? pw : 1 - pw;
}

function currentExplanation() {
  if (!state || plyOf() === 0) return null;
  return explainCache.get(hkey(state.history)) || null;
}

// Explanation of the move that led to the position on screen.
function shownExplanation() {
  const st = shownState();
  if (!st || st.history.length === 0) return null;
  return explainCache.get(hkey(st.history)) || null;
}

function currentBestMove() {
  const info = analysisCache.get(hkey(state.history));
  return info && info.best ? info.best.sq : null;
}

async function playInAnalysis(move) {
  const ply = plyOf();
  const same = line[ply] && sameHistory(line[ply], move);
  if (!same) line = line.slice(0, ply).concat([move]);
  await run('play', { move });
  ensureAnalyzed();
}

async function navigate(n) {
  if (!isAnalysis() || busy) return;
  n = Math.max(0, Math.min(line.length, n));
  if (n === plyOf()) return;
  analysisToken++;
  analyzingNow = true;
  await run('load', { history: flat(line.slice(0, n)) });
  ensureAnalyzed();
}

// Analyses the current position and the one before it (if needed), then
// explains the last move.
async function ensureAnalyzed() {
  const token = ++analysisToken;
  const hist = Array.from(state.history);
  const n = hist.length / 3;
  const targets = n > 0 ? [n - 1, n] : [0];
  analyzingNow = true;
  render();
  try {
    for (const t of targets) {
      if (!(await analyzePrefix(hist, t, token))) return;
    }
    if (n > 0) await explainPrefix(hist, n, token);
  } finally {
    if (token === analysisToken) {
      analyzingNow = false;
      render();
    }
  }
}

// Searches the position after the first `t` moves of `hist` (cached).
// Returns false if superseded.
async function analyzePrefix(hist, t, token) {
  const h = hist.slice(0, 3 * t);
  const key = hkey(h);
  if (analysisCache.has(key)) return true;
  let info;
  if (t === hist.length / 3 && state.over) {
    info = { value: state.winner === 0 ? 1 : -1, proven: state.winner === 0 ? 1 : -1, best: null, pv: [], iterations: 0 };
  } else {
    info = await parallelSearch(ANALYSIS_MS, h);
    busy = false;
  }
  if (token !== analysisToken || !info) return false;
  analysisCache.set(key, info);
  render();
  return true;
}

async function explainPrefix(hist, n, token) {
  const key = hkey(hist.slice(0, 3 * n));
  if (explainCache.has(key) || explaining.has(key)) return;
  explaining.add(key);
  try {
    await explainPrefixInner(hist, n, token, key);
  } finally {
    explaining.delete(key);
    render();
  }
}

async function explainPrefixInner(hist, n, token, key) {
  const beforeKey = hkey(hist.slice(0, 3 * (n - 1)));
  const before = analysisCache.get(beforeKey) || (await pendingAnalysis.get(beforeKey));
  const after = analysisCache.get(key) || (await pendingAnalysis.get(key));
  if (!before || !after || token !== analysisToken) return;
  const mover = (n - 1) % 2;
  // Compare within one search tree when the played move was explored
  // enough there: separate searches are each optimistic for their side to
  // move, which would exaggerate every loss.
  const played = hist.slice(3 * (n - 1), 3 * n);
  const total = before.candidates ? before.candidates.reduce((a, c) => a + c.visits, 0) : 0;
  const cand = before.candidates && before.candidates.find((c) => sameHistory(c.sq, played));
  const best = before.candidates && before.candidates[0];
  let pBefore, pAfter;
  if (!before.proven && !after.proven && cand && best && cand.q != null && cand.visits >= Math.max(30, 0.03 * total)) {
    const toMover = (q) => (mover === 0 ? (q + 1) / 2 : (1 - q) / 2);
    pAfter = toMover(cand.q);
    pBefore = Math.max(toMover(best.q), pAfter);
  } else if (!before.proven && !after.proven && before.best && !sameHistory(before.best.sq, played)) {
    // Measure the recommended move the same way as the played one: search
    // the position after it, so both values carry the same bias.
    const hb = hist.slice(0, 3 * (n - 1)).concat(before.best.sq);
    let afterBest = analysisCache.get(hkey(hb));
    if (!afterBest) {
      afterBest = await parallelSearch(ANALYSIS_MS, hb);
      busy = false;
      if (token !== analysisToken || !afterBest) return;
      analysisCache.set(hkey(hb), afterBest);
    }
    pAfter = moverP(after, mover);
    pBefore = Math.max(moverP(afterBest, mover), pAfter);
  } else {
    pAfter = moverP(after, mover);
    pBefore = Math.max(moverP(before, mover), pAfter);
  }
  const r = await call('explain', {
    history: hist.slice(0, 3 * n),
    ply: n - 1,
    best: before.best ? before.best.sq : [],
    reply: after.best ? after.best.sq : [],
    pBefore,
    pAfter,
  });
  if (token !== analysisToken || !r.result) return;
  explainCache.set(key, r.result);
  render();
}

function esc(t) {
  return t.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
}

// Engine texts say "내/상대" from the mover's side; name the colours instead
// so the card reads the same whoever is looking (and whoever moved).
function inColors(text, mover) {
  const me = colorName(mover), opp = colorName(1 - mover);
  // 백/흑 both end in a consonant: 가→이, 는→은, 를→을.
  return text.replace(/내가 /g, '\u0001이 ').replace(/내 /g, '\u0001 ')
    .replace(/상대가 /g, `${opp}이 `).replace(/상대는 /g, `${opp}은 `).replace(/상대를 /g, `${opp}을 `)
    .replace(/상대의 /g, `${opp}의 `).replace(/상대 /g, `${opp} `).replace(/\u0001/g, me);
}

// Move-evaluation card (layout follows the Lines of Action site): grade badge,
// move, verdict, White's winning chances with the best move as a tick, then
// reasons and what the engine would have played / the opponent's answer.
// `n` is the 1-based ply of the move `mv`.
function evalCard(ex, n, mv, who = '') {
  const mover = (n - 1) % 2;
  const me = colorName(mover), opp = colorName(1 - mover);
  const top = `<span class="ev-move"><span class="chip ${mover === 0 ? 'w' : 'b'}"></span>${Math.floor((n - 1) / 2) + 1}. ${me} ${moveName(mv)}${who ? `<span class="ev-who">${who}</span>` : ''}</span>`;
  if (!ex) {
    return `<div class="ev-card"><div class="ev-top">${top}</div><div class="ev-wait"><span class="pulse"></span>이 수를 따져 보는 중</div></div>`;
  }
  const win = /이기는 수/.test(ex.summary);
  const label = win ? '이기는 수' : GRADE_LABEL[ex.grade] || ex.label;
  const head = win ? ex.summary
    : who === '상대 수' && ex.grade === 'best' ? '이 국면의 최선의 수예요.'
    : GRADE_HEAD[ex.grade] || ex.summary;
  const toWhite = (p) => (mover === 0 ? p : 1 - p);
  const a = Math.round(100 * toWhite(ex.pAfter));
  const b = Math.round(100 * toWhite(ex.pBefore));
  const loss = Math.round(100 * (ex.pBefore - ex.pAfter));
  let h = `<div class="ev-card"><div class="ev-top"><span class="ev-badge" data-k="${win ? 'best' : ex.grade}">${label}</span>${top}</div>`;
  h += `<p class="ev-head">${esc(head)}</p>`;
  h += `<div class="ev-odds"><div class="ev-odds-row"><span>둔 뒤 백 승률</span><b>${a}%</b>` +
    (loss > 0
      ? `<span class="ev-delta" data-k="${ex.grade}">${me}이 최선을 뒀다면 백 ${b}% · ${me} −${loss}%p</span>`
      : `<span class="ev-delta">엔진 최선과 같아요</span>`) +
    `</div><div class="ev-track" aria-hidden="true"><div class="ev-now" style="width:${a}%"></div>${loss > 0 ? `<div class="ev-tick" style="left:${b}%"></div>` : ''}</div></div>`;
  const li = (t) => (p) => `<li data-t="${t === null ? (p.positive ? 'plus' : 'minus') : t}">${esc(inColors(p.text, mover))}</li>`;
  if (ex.points.length) h += `<ul class="ev-why">${ex.points.map(li(null)).join('')}</ul>`;
  if (ex.best) {
    h += `<div class="ev-sub"><div class="label">${ex.grade === 'good' ? '비슷하게 좋은 대안' : '엔진이라면'} · <strong>${me} ${ex.best.move}</strong></div>`;
    if (ex.bestPoints.length) h += `<ul class="ev-why">${ex.bestPoints.map(li('plus')).join('')}</ul>`;
    h += '</div>';
  }
  if (ex.replyPoints.length) {
    h += `<div class="ev-sub"><div class="label">${opp}의 좋은 응수${ex.reply ? ` · <strong>${opp} ${ex.reply}</strong>` : ''}</div>`;
    h += `<ul class="ev-why">${ex.replyPoints.map(li('minus')).join('')}</ul></div>`;
  }
  h += `<p class="note">${esc(ex.balance)}</p></div>`;
  return h;
}

function renderAnalysis() {
  if (!isAnalysis() || !state) return;
  const n = plyOf();
  els.prev.disabled = busy || n === 0;
  els.next.disabled = busy || n >= line.length;
  els.analyzeAll.disabled = busy || line.length === 0;
  els.showBest.setAttribute('aria-pressed', String(showBest));
  els.showBest.classList.toggle('on', showBest);
  els.analysisLegend.hidden = !showBest;

  const info = analysisCache.get(hkey(state.history));
  if (state.over) {
    els.posInfo.textContent = `게임 종료 — ${colorName(state.winner)} 승리`;
  } else if (info) {
    const side = colorName(state.side);
    const pw = Math.round(100 * moverP(info, 0));
    const verdict = info.proven ? `${info.proven > 0 ? '백' : '흑'} 필승 확인` : `백 승률 ${pw}%`;
    const best = info.best ? ` · ${side} 추천 ${info.best.move}` : '';
    els.posInfo.textContent = `${side} 차례 · ${verdict}${best}`;
  } else {
    els.posInfo.textContent = analyzingNow ? '이 국면을 분석하는 중…' : '';
  }

  const ex = currentExplanation();
  if (!ex) {
    els.explain.innerHTML = n === 0
      ? '<p class="muted">수를 두면 그 수가 왜 좋은지, 왜 아쉬운지 설명해 드려요.</p>'
      : analyzingNow ? evalCard(null, n, state.history.slice(3 * (n - 1), 3 * n)) : '';
  } else {
    els.explain.innerHTML = evalCard(ex, n, state.history.slice(3 * (n - 1), 3 * n));
  }
  renderGraph();
}

// White's win chance per ply along the line (analysed positions only).
function renderGraph() {
  const N = line.length;
  const W = 300, H = 70;
  const x = (i) => (N === 0 ? W / 2 : (i / N) * W);
  const y = (p) => H - p * H;
  const pts = [];
  const dots = [];
  for (let i = 0; i <= N; i++) {
    const info = analysisCache.get(hkey(flat(line.slice(0, i))));
    if (!info) continue;
    const p = moverP(info, 0);
    pts.push(`${x(i).toFixed(1)},${y(p).toFixed(1)}`);
    const ex = i > 0 ? explainCache.get(hkey(flat(line.slice(0, i)))) : null;
    if (ex && GRADE_MARK[ex.grade]) dots.push(`<circle class="gd g-${ex.grade}" cx="${x(i).toFixed(1)}" cy="${y(p).toFixed(1)}" r="3.5"/>`);
  }
  const cur = x(plyOf());
  const hits = [];
  for (let i = 0; i <= N; i++) {
    const w = N === 0 ? W : W / N;
    hits.push(`<rect class="hit" data-ply="${i}" x="${(x(i) - w / 2).toFixed(1)}" y="0" width="${w.toFixed(1)}" height="${H}"/>`);
  }
  els.graph.innerHTML =
    `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img">` +
    `<rect class="bg-black" x="0" y="0" width="${W}" height="${H}"/>` +
    `<line class="mid" x1="0" y1="${H / 2}" x2="${W}" y2="${H / 2}"/>` +
    (pts.length ? `<polygon class="area" points="0,${H} ${pts.join(' ')} ${pts[pts.length - 1].split(',')[0]},${H}"/>` : '') +
    (pts.length ? `<polyline class="line" points="${pts.join(' ')}"/>` : '') +
    `<line class="cur" x1="${cur}" y1="0" x2="${cur}" y2="${H}"/>` +
    dots.join('') +
    hits.join('') +
    '</svg>';
}

els.graph.addEventListener('click', (e) => {
  const r = e.target.closest('rect.hit');
  if (r) navigate(Number(r.dataset.ply));
});

function renderLine() {
  const n = plyOf();
  const cells = line.map((m, i) => {
    const ex = explainCache.get(hkey(flat(line.slice(0, i + 1))));
    const mark = ex ? GRADE_MARK[ex.grade] : '';
    const cls = ['mv', i === n - 1 ? 'cur shown' : '', i >= n ? 'ahead' : '', ex ? `g-${ex.grade}` : ''].join(' ');
    return `<td class="${cls}" data-ply="${i + 1}">${moveName(m)}${mark ? `<span class="mark">${mark}</span>` : ''}</td>`;
  });
  const rows = [];
  for (let i = 0; i < cells.length; i += 2) {
    rows.push(`<tr><td>${i / 2 + 1}</td>${cells[i]}${cells[i + 1] || '<td></td>'}</tr>`);
  }
  const html = rows.join('');
  if (els.moves.innerHTML !== html) {
    els.moves.innerHTML = html;
    const curEl = els.moves.querySelector('td.cur');
    if (curEl) curEl.scrollIntoView({ block: 'nearest' });
  }
}

els.moves.addEventListener('click', (e) => {
  const td = e.target.closest('td.mv');
  if (td) goToPly(Number(td.dataset.ply));
});

els.prev.addEventListener('click', () => navigate(plyOf() - 1));
els.next.addEventListener('click', () => navigate(plyOf() + 1));
els.showBest.addEventListener('click', () => {
  showBest = !showBest;
  render();
});

// Analyse every position of the line, then explain every move.
els.analyzeAll.addEventListener('click', async () => {
  const token = ++analysisToken;
  const all = flat(line);
  analyzingNow = true;
  render();
  try {
    for (let t = 0; t <= line.length; t++) {
      if (!(await analyzePrefix(all, t, token))) return;
    }
    for (let t = 1; t <= line.length; t++) {
      await explainPrefix(all, t, token);
      if (token !== analysisToken) return;
    }
  } finally {
    if (token === analysisToken) {
      analyzingNow = false;
      render();
    }
  }
});

let importing = false;

els.exportKibo.addEventListener('click', async () => {
  importing = false;
  els.importKibo.textContent = '기보 불러오기';
  const text = line.map(moveName).join(' ');
  els.kiboText.hidden = false;
  els.kiboText.value = text;
  els.kiboText.select();
  try {
    await navigator.clipboard.writeText(text);
    els.kiboMsg.textContent = '기보를 클립보드에 복사했어요.';
  } catch {
    els.kiboMsg.textContent = '아래 칸의 기보를 복사하세요.';
  }
});

// First press opens an empty box, second press loads what was pasted.
els.importKibo.addEventListener('click', async () => {
  if (!importing) {
    importing = true;
    els.kiboText.hidden = false;
    els.kiboText.value = '';
    els.kiboText.focus();
    els.importKibo.textContent = '이 기보로 불러오기';
    return;
  }
  importing = false;
  const text = els.kiboText.value.trim();
  els.importKibo.textContent = '기보 불러오기';
  els.kiboText.hidden = true;
  if (!text) return;
  analysisToken++;
  const res = await run('loadNotation', { text });
  const h = Array.from(state.history);
  line = [];
  for (let i = 0; i < h.length; i += 3) line.push(h.slice(i, i + 3));
  analysisCache.clear();
  explainCache.clear();
  ensureAnalyzed();
  els.kiboMsg.textContent = res && res.failed
    ? `${res.played}수까지 불러왔어요. 읽을 수 없는 수: ${res.failed}`
    : `${res ? res.played : 0}수를 불러왔어요.`;
});

// ---- friendly game (coach) -------------------------------------------------
//
// Playing the AI with explanations of the human's moves. While the human
// thinks, helper workers analyse the position ("before"). The AI's own
// search of the position after the human's move is the "after" analysis and
// yields the AI's reply, so coaching costs no extra search except when the
// played move must be compared against the recommended one.

const pendingAnalysis = new Map(); // history key -> Promise<info> (helpers only)
let coachHint = false;

const coachOn = () => settings.mode === 'coach';

// Starts the background analysis of the current position (human to move).
function coachPrepare() {
  analyzeInBackground(Array.from(state.history));
}

// Helper-worker analysis of `hist`, registered in `pendingAnalysis` so
// explanations can wait for it.
function analyzeInBackground(hist) {
  const key = hkey(hist);
  if (analysisCache.has(key) || pendingAnalysis.has(key)) return;
  const p = parallelSearch(ANALYSIS_MS, hist, true).then((info) => {
    pendingAnalysis.delete(key);
    if (info) analysisCache.set(key, info);
    render();
    return info;
  });
  pendingAnalysis.set(key, p);
}

// After the AI replied: explain the human's move, then the AI's move. The
// AI's own search is the "before" of its move; the "after" is the analysis
// of the human's new position, which coachPrepare() has already started.
async function coachAfterAi(beforeAi, aiInfo, aiMove, my) {
  const token = analysisToken;
  const n = beforeAi.length / 3; // plies up to and including the human move
  const withAi = beforeAi.concat(aiMove);
  if (state.over && sameHistory(state.history, withAi)) {
    // The AI's move ended the game: the "after" value is the result.
    const v = state.winner === 0 ? 1 : -1;
    analysisCache.set(hkey(withAi), { value: v, proven: v, best: null, pv: [], candidates: [], iterations: 0 });
  }
  if (n > 0 && (n - 1) % 2 === settings.human) {
    coachBusy = true;
    render();
    await explainPrefix(beforeAi, n, token);
    coachBusy = false;
  }
  if (my !== epoch || token !== analysisToken) return;
  await explainPrefix(withAi, n + 1, token);
}
let coachBusy = false;

// The human's move ended the game (the AI cannot move): explain it with a
// terminal "after" value.
async function coachFinal(hist) {
  const key = hkey(hist);
  analysisCache.set(key, { value: state.winner === 0 ? 1 : -1, proven: state.winner === 0 ? 1 : -1, best: null, pv: [], candidates: [], iterations: 0 });
  coachBusy = true;
  render();
  await explainPrefix(hist, hist.length / 3, analysisToken);
  coachBusy = false;
  render();
}

function renderCoach() {
  const on = coachOn() && state;
  els.coachPanel.hidden = !on;
  if (!on) return;
  const H = state.history;
  const plies = H.length / 3;
  const card = (i) => {
    const key = hkey(H.slice(0, 3 * (i + 1)));
    const ex = explainCache.get(key) || null;
    const who = i % 2 === settings.human ? '내 수' : '상대 수';
    if (!ex && !explaining.has(key) && !pendingAnalysis.size && !coachBusy && !busy) return '';
    return evalCard(ex, i + 1, H.slice(3 * i, 3 * i + 3), who);
  };
  let html = '';
  if (viewing()) {
    // Looking back: the move that led to the position on screen.
    html = viewPly > 0 ? card(viewPly - 1) || '<p class="muted">이 수는 평가하지 않았어요.</p>' : '<p class="muted">시작 국면이에요.</p>';
  } else {
    // Live: my latest move and the engine's answer to it, both evaluated.
    let mine = -1;
    for (let i = plies - 1; i >= 0; i--) {
      if (i % 2 === settings.human) {
        mine = i;
        break;
      }
    }
    if (mine < 0) {
      html = plies > 0 ? card(plies - 1) : '';
      html += '<p class="muted">수를 두면 내 수와 엔진의 응수가 왜 좋은지, 왜 아쉬운지 바로 알려 드려요.</p>';
    } else {
      html = card(mine);
      if (mine + 1 < plies) html += card(mine + 1);
    }
    if (state.over) html += coachSummary();
  }
  els.coachBody.innerHTML = html;
  els.evPrev.disabled = shownPly() <= 0;
  els.evNext.disabled = shownPly() >= livePlies();

  const info = analysisCache.get(hkey(H));
  const ready = isHumanTurn() && info && info.best;
  els.hintBtn.disabled = viewing() || !isHumanTurn() || (!ready && !pendingAnalysis.size);
  els.hintBtn.textContent = !isHumanTurn()
    ? '힌트'
    : ready ? (coachHint ? '힌트 숨기기' : '힌트 (추천 수 보기)') : '힌트 준비 중…';
  els.hintBtn.classList.toggle('on', coachHint && ready);
  els.coachLegend.hidden = !(coachHint && ready) || viewing();
}

// Grade counts of the human's moves, shown at the end of the game.
function coachSummary() {
  const H = state.history;
  const counts = { best: 0, good: 0, inaccuracy: 0, mistake: 0, blunder: 0 };
  let total = 0;
  for (let i = settings.human; i < H.length / 3; i += 2) {
    const ex = explainCache.get(hkey(H.slice(0, 3 * (i + 1))));
    if (!ex) continue;
    counts[ex.grade]++;
    total++;
  }
  if (!total) return '';
  const parts = Object.keys(counts).map(
    (k) => `<span class="ev-badge" data-k="${k}">${GRADE_LABEL[k]} ${counts[k]}</span>`,
  );
  const won = state.winner === settings.human;
  return `<div class="coach-sum"><p class="ex-head">${won ? '승리했어요!' : '대국 종료'} · 내 수 ${total}개 평가</p><p class="sum-row">${parts.join(' ')}</p></div>`;
}

els.evPrev.addEventListener('click', () => stepView(-1));
els.evNext.addEventListener('click', () => stepView(1));
els.hintBtn.addEventListener('click', () => {
  coachHint = !coachHint;
  render();
});
for (const b of els.modes) {
  b.addEventListener('click', () => {
    if (b.dataset.mode === settings.mode) return;
    settings.mode = b.dataset.mode;
    try { localStorage.setItem('mode', settings.mode); } catch {}
    newGame();
  });
}

// ---- opening book -----------------------------------------------------------
//
// web/book.json is generated offline (engine/examples/book.rs): for every
// position of the first plies, the engine's top moves with win rates, the
// expected continuation and pros/cons. Keys are the move notation joined by
// spaces ("" = start position).

let book = null;
let bookSel = 0;

async function loadBook() {
  if (book) return render();
  els.bookInfo.textContent = '오프닝 북을 불러오는 중…';
  try {
    const r = await fetch(new URL('./book.json', import.meta.url));
    book = await r.json();
  } catch (err) {
    els.bookInfo.textContent = '오프닝 북을 불러오지 못했어요.';
    return;
  }
  render();
}

const bookEntry = () => (book && state ? book.nodes[state.notation.join(' ')] : null);

function bookHintMove() {
  const e = bookEntry();
  return e && e.moves[bookSel] ? e.moves[bookSel].sq : null;
}

function renderBook() {
  if (!isBook() || !state) return;
  const n = state.history.length / 3;
  els.bookBack.disabled = busy || n === 0;
  els.bookStart.disabled = busy || n === 0;
  if (!book) return;
  const e = bookEntry();
  if (!e) {
    els.bookInfo.textContent = `${n}수째 · 북에 없는 국면이에요.`;
    els.bookList.innerHTML =
      `<p class="muted">오프닝 북은 처음 ${book.depth}수까지, 국면마다 엔진이 고른 상위 ${book.width}개 수만 담고 있어요.</p>` +
      '<button id="toAnalysis" type="button">분석 모드에서 이어서 보기</button>';
    return;
  }
  const side = e.side;
  els.bookInfo.textContent = `${n}수째 · ${side} 차례 · 최선을 두면 ${side} 승률 ${Math.round(e.win)}%`;
  const li = (cls) => (t) => `<li class="${cls}">${esc(t)}</li>`;
  els.bookList.innerHTML = e.moves.map((m, i) => {
    const tag = i === 0 ? 'best' : m.summary.includes('거의') ? 'good' : m.summary.includes('무난') ? 'inaccuracy' : 'mistake';
    const mirror = m.mirror ? `<span class="mirror">좌우 대칭 ${m.mirror}도 같아요</span>` : '';
    return `<div class="bookmove${i === bookSel ? ' sel' : ''}" data-i="${i}">
      <p class="ex-head"><span class="rank">${i + 1}</span> <b>${m.move}</b> <span class="grade g-${tag}">${esc(m.summary)}</span> <span class="win">${side} ${Math.round(m.win)}%</span></p>
      ${mirror}
      <ul class="ex-pts">${m.pros.map(li('pos')).join('')}${m.cons.map(li('neg')).join('')}</ul>
      <p class="line">대표 수순: ${m.line.map(esc).join(' ')}</p>
      <button type="button" class="play" data-i="${i}">이 수 두기</button>
    </div>`;
  }).join('');
}

els.bookList.addEventListener('click', async (ev) => {
  if (ev.target.id === 'toAnalysis') return openInAnalysis();
  const card = ev.target.closest('.bookmove');
  if (!card || busy) return;
  const i = Number(card.dataset.i);
  if (ev.target.classList.contains('play')) {
    const m = bookEntry().moves[i];
    bookSel = 0;
    await run('play', { move: m.sq });
  } else {
    bookSel = i;
    render();
  }
});
els.bookBack.addEventListener('click', async () => {
  bookSel = 0;
  await run('undo', { count: 1 });
});
els.bookStart.addEventListener('click', async () => {
  bookSel = 0;
  await run('newGame');
});

// Continue the current line in analysis mode.
async function openInAnalysis() {
  const h = Array.from(state.history);
  settings.mode = 'analysis';
  try { localStorage.setItem('mode', settings.mode); } catch {}
  await newGame();
  analysisToken++;
  await run('load', { history: h });
  line = [];
  for (let i = 0; i < h.length; i += 3) line.push(h.slice(i, i + 3));
  ensureAnalyzed();
}

// ---- looking back through the game (← →) -----------------------------------
//
// Analysis mode moves the real position (navigate); the other modes only
// show stored snapshots of earlier positions.

function livePlies() {
  return isAnalysis() ? line.length : state.history.length / 3;
}

function shownPly() {
  if (isAnalysis()) return plyOf();
  return viewPly === null ? state.history.length / 3 : viewPly;
}

function goToPly(n) {
  if (!state) return;
  n = Math.max(0, Math.min(livePlies(), n));
  if (isAnalysis()) return navigate(n);
  if (sel.from >= 0) sel = { from: -1, to: -1 };
  viewPly = n >= state.history.length / 3 ? null : n;
  annotations = [];
  render();
}

function stepView(d) {
  goToPly(shownPly() + d);
}

function renderViewControls() {
  const n = shownPly();
  els.viewPrev.disabled = n <= 0 || (isAnalysis() && busy);
  els.viewNext.disabled = n >= livePlies() || (isAnalysis() && busy);
  els.drawBtn.setAttribute('aria-pressed', String(drawMode));
  els.drawBtn.classList.toggle('on', drawMode);
  els.clearDraw.disabled = annotations.length === 0;
  els.viewInfo.textContent = drawMode
    ? '그리기: 칸을 누르면 동그라미, 칸에서 칸으로 끌면 화살표. 같은 걸 다시 그리면 지워져요.'
    : viewing() ? `${n}수째 국면 · 마지막 국면은 → 또는 판을 누르세요` : '';
}

els.viewPrev.addEventListener('click', () => stepView(-1));
els.viewNext.addEventListener('click', () => stepView(1));

// ---- drawing on the board ---------------------------------------------------
//
// Circles and arrows for explaining a position. Right-click / right-drag on
// a computer; on touch screens switch on "그리기" and tap / drag. Drawings
// belong to the position on screen and disappear when it changes.

let annotations = []; // { kind: 'circle', sq } | { kind: 'arrow', from, to }
let drawMode = false;
let drawStart = null; // { sq, pointerId }
let suppressClick = false;

function setDrawMode(on) {
  drawMode = on;
  els.boardArea.classList.toggle('drawing', on);
  render();
}

// Board square under a screen point, or -1.
function squareFromPoint(x, y) {
  const r = els.board.getBoundingClientRect();
  if (x < r.left || y < r.top || x >= r.right || y >= r.bottom) return -1;
  const vc = Math.floor(((x - r.left) / r.width) * 10);
  const vr = Math.floor(((y - r.top) / r.height) * 10);
  return squareAt(vr * 10 + vc);
}

// Centre of a square in the 0..100 overlay coordinates.
function squareCenter(sq) {
  const flip = isFlipped();
  const row = Math.floor(sq / 10), col = sq % 10;
  const vr = flip ? row : 9 - row;
  const vc = flip ? 9 - col : col;
  return [vc * 10 + 5, vr * 10 + 5];
}

function toggleAnnotation(a) {
  const same = (b) => b.kind === a.kind && (a.kind === 'circle' ? b.sq === a.sq : b.from === a.from && b.to === a.to);
  const i = annotations.findIndex(same);
  if (i >= 0) annotations.splice(i, 1);
  else annotations.push(a);
  render();
}

function renderAnnotations() {
  const parts = ['<defs><marker id="ah" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="3.2" markerHeight="3.2" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="ahead"/></marker></defs>'];
  for (const a of annotations) {
    if (a.kind === 'circle') {
      const [x, y] = squareCenter(a.sq);
      parts.push(`<circle cx="${x}" cy="${y}" r="4.1" class="acircle"/>`);
    } else {
      const [x1, y1] = squareCenter(a.from);
      let [x2, y2] = squareCenter(a.to);
      // Stop short of the centre so the head sits inside the target square.
      const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
      x2 -= (dx / len) * 2.2;
      y2 -= (dy / len) * 2.2;
      parts.push(`<line x1="${x1}" y1="${y1}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}" class="aline" marker-end="url(#ah)"/>`);
    }
  }
  els.annot.innerHTML = parts.join('');
}

els.boardArea.addEventListener('contextmenu', (e) => e.preventDefault());
els.boardArea.addEventListener('pointerdown', (e) => {
  if (!(e.button === 2 || (drawMode && e.button === 0))) return;
  const sq = squareFromPoint(e.clientX, e.clientY);
  if (sq < 0) return;
  e.preventDefault();
  drawStart = { sq, pointerId: e.pointerId };
});
window.addEventListener('pointerup', (e) => {
  if (!drawStart || e.pointerId !== drawStart.pointerId) return;
  const from = drawStart.sq;
  drawStart = null;
  const to = squareFromPoint(e.clientX, e.clientY);
  if (to < 0) return;
  toggleAnnotation(to === from ? { kind: 'circle', sq: to } : { kind: 'arrow', from, to });
  // The click that follows a left-button drawing must not play a move.
  suppressClick = true;
  setTimeout(() => (suppressClick = false), 0);
});
els.drawBtn.addEventListener('click', () => setDrawMode(!drawMode));
els.clearDraw.addEventListener('click', () => {
  annotations = [];
  render();
});

// Expose for debugging and automated checks.
window.amazons = {
  call,
  combineInfos,
  get state() { return state; },
  get lastAiInfo() { return lastAiInfo; },
  get line() { return line; },
  get book() { return book; },
  analysisCache,
  explainCache,
  get analyzing() { return analyzingNow; },
  get busy() { return busy; },
  get viewPly() { return viewPly; },
  get annotations() { return annotations; },
  get coachBusy() { return coachBusy || pendingAnalysis.size > 0; },
};

buildBoard();
newGame();
