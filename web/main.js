// UI thread: rendering and input only. All rules live in the engine worker.

const FILES = 'abcdefghij';
const sqName = (sq) => FILES[sq % 10] + (Math.floor(sq / 10) + 1);
const AI_DELAY_MS = 350; // pause before instant (random/greedy) replies
const VALUE_SCALE = 16;   // same squash as the engine: tanh(score / 16)

const els = {
  board: document.getElementById('board'),
  status: document.getElementById('status'),
  hint: document.getElementById('hint'),
  moves: document.getElementById('moves'),
  opponent: document.getElementById('opponent'),
  color: document.getElementById('color'),
  newGame: document.getElementById('newGame'),
  undo: document.getElementById('undo'),
  cancel: document.getElementById('cancel'),
  heat: document.getElementById('heat'),
  evalFill: document.getElementById('evalFill'),
  evalText: document.getElementById('evalText'),
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
  coach: document.getElementById('coach'),
  coachLabel: document.getElementById('coachLabel'),
  coachPanel: document.getElementById('coachPanel'),
  coachBody: document.getElementById('coachBody'),
  hintBtn: document.getElementById('hintBtn'),
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
const settings = { opponent: 'mcts:2000', human: 0 };
let lastAiInfo = null;

function setState(s) {
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

const isAnalysis = () => settings.opponent === 'analysis';
const isHumanTurn = () =>
  state && !state.over && (settings.opponent === 'human' || isAnalysis() || state.side === settings.human);

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
  const my = epoch;
  const [kind, ms] = settings.opponent.split(':');
  busy = true;
  render();
  if (kind !== 'mcts') {
    await new Promise((r) => setTimeout(r, AI_DELAY_MS));
    if (my !== epoch) return;
    const before = Array.from(state.history);
    await run('aiMove', { kind });
    if (my !== epoch) return;
    if (coachOn() && state.history.length > before.length) {
      // Random/greedy AIs do not search: analyse the position after the
      // human's move separately (helpers only, in the background).
      const aiMove = Array.from(state.history.slice(before.length, before.length + 3));
      parallelSearch(ANALYSIS_MS, before, true).then((info) => {
        if (!info || my !== epoch) return;
        analysisCache.set(hkey(before), info);
        coachAfterAi(before, info, aiMove, my);
      });
    }
  } else {
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
    if (coachOn()) coachAfterAi(before, info, info.best.sq, my);
  }
  if (my !== epoch) return;
  if (coachOn() && isHumanTurn()) coachPrepare();
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

function render() {
  if (!state) return;
  const cells = Array.from(state.cells);
  // Preview the amazon on its destination while choosing the arrow.
  if (sel.to >= 0) {
    cells[sel.to] = cells[sel.from];
    cells[sel.from] = 0;
  }
  const H = state.history;
  const last = H.length >= 3 ? H.slice(H.length - 3) : null;
  let targets = null, targetClass = '';
  if (sel.to >= 0) {
    targets = tree.get(sel.from).get(sel.to);
    targetClass = 't-arrow';
  } else if (sel.from >= 0) {
    targets = tree.get(sel.from);
    targetClass = 't-move';
  }
  const human = isHumanTurn() && !busy;
  const flip = isFlipped();
  const ex = isAnalysis() && sel.from < 0 ? currentExplanation() : null;
  const gained = new Set(ex ? ex.gained : []);
  const lost = new Set(ex ? ex.lost : []);
  const hintMove = (isAnalysis() ? showBest : coachHint) && sel.from < 0 ? currentBestMove() : null;
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
    if (hintMove) {
      if (sq === hintMove[0]) cls.push('hint-from');
      if (sq === hintMove[1]) cls.push('hint-to');
      if (sq === hintMove[2]) cls.push('hint-arrow');
    }
    if (human && sel.from < 0 && tree.has(sq)) cls.push('movable');
    el.className = cls.join(' ');
    // Coordinates on the edge squares.
    const vr = Math.floor(i / 10), vc = i % 10;
    el.dataset.file = vr === 9 ? FILES[c] : '';
    el.dataset.rank = vc === 0 ? String(r + 1) : '';
    el.setAttribute('aria-label', sqName(sq) + labelFor(v));
  }
  els.board.classList.toggle('flipped', flip);
  renderStatus();
  renderEval();
  renderMoves();
  renderAnalysis();
  renderCoach();
  els.undo.disabled = busy || undoCount() === 0;
  els.cancel.disabled = sel.from < 0;
}

// Territory by queen distance: who reaches the empty square first, and how
// clearly (distance gap).
function heatFor(sq) {
  const dw = state.dist[sq], db = state.dist[100 + sq];
  if (dw === 255 && db === 255) return null;
  if (dw === db) return { cls: 'h-tie', amt: '30%' };
  const gap = Math.abs(dw - db);
  const amt = dw === 255 || db === 255 || gap >= 3 ? '85%' : gap === 2 ? '62%' : '42%';
  return { cls: dw < db ? 'h-white' : 'h-black', amt };
}

function renderEval() {
  const e = state.evaluation;
  let v;
  if (state.over) v = state.winner === 0 ? 1 : -1;
  else v = Math.tanh(e.score / VALUE_SCALE);
  els.evalFill.style.width = pct(v);
  const sign = (x) => (x >= 0 ? '+' : '') + x.toFixed(1);
  els.evalText.textContent = state.over
    ? ''
    : `정적 평가 (백 기준) ${sign(e.score)} · 영역 ${sign(e.t1)} · 국면 ${e.phase >= 0.99 ? '초반' : e.phase > 0.3 ? '중반' : '종반'}`;
}

function labelFor(v) {
  return v === 1 ? ' 백 아마존' : v === 2 ? ' 흑 아마존' : v === 3 ? ' 화살' : '';
}

const colorName = (c) => (c === 0 ? '백' : '흑');

function renderStatus() {
  const s = state;
  let text, hint = '';
  if (s.over) {
    text = `게임 종료 — ${colorName(s.winner)} 승리 (${colorName(s.side)}이 둘 곳이 없음)`;
  } else if (isHumanTurn()) {
    const who = settings.opponent === 'human' || isAnalysis() ? '' : ' (당신)';
    text = `${colorName(s.side)} 차례${who} · 합법수 ${s.legal.length / 3}개`;
    hint = sel.to >= 0 ? '화살을 쏠 칸을 고르세요.'
      : sel.from >= 0 ? '이동할 칸을 고르세요.'
      : '움직일 아마존을 고르세요.';
  } else {
    const name = els.opponent.selectedOptions[0].textContent;
    text = `${colorName(s.side)} (${name}) 생각 중…`;
  }
  if (isAnalysis() && analyzingNow) hint = '분석 중…';
  els.status.textContent = text;
  els.hint.textContent = hint;
}

function renderMoves() {
  if (isAnalysis()) return renderLine();
  const n = state.notation;
  const rows = [];
  const cell = (j) => {
    if (j >= n.length) return '<td></td>';
    const ex = coachOn() && j % 2 === settings.human
      ? explainCache.get(hkey(state.history.slice(0, 3 * (j + 1))))
      : null;
    const mark = ex ? GRADE_MARK[ex.grade] : '';
    const cls = [j === n.length - 1 ? 'cur' : '', ex ? `g-${ex.grade}` : ''].join(' ').trim();
    return `<td${cls ? ` class="${cls}"` : ''}>${n[j]}${mark ? `<span class="mark">${mark}</span>` : ''}</td>`;
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
  if (settings.opponent === 'human' || isAnalysis()) return plies > 0 ? 1 : 0;
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
  settings.opponent = els.opponent.value;
  settings.human = Number(els.color.value);
  lastAiInfo = null;
  showAiInfo(null);
  els.color.disabled = settings.opponent === 'human' || isAnalysis();
  els.analysisPanel.hidden = !isAnalysis();
  els.undo.textContent = isAnalysis() ? '한 수 뒤로' : '무르기';
  line = [];
  analysisCache.clear();
  explainCache.clear();
  analysisToken++;
  epoch++;
  await run('newGame');
  coachHint = false;
  if (isAnalysis()) ensureAnalyzed();
  else {
    if (coachOn() && isHumanTurn()) coachPrepare();
    maybeAiTurn();
  }
}

els.newGame.addEventListener('click', newGame);
els.opponent.addEventListener('change', newGame);
els.color.addEventListener('change', newGame);

document.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLTextAreaElement) return;
  if (isAnalysis() && e.key === 'ArrowLeft') els.prev.click();
  else if (isAnalysis() && e.key === 'ArrowRight') els.next.click();
  else if (e.key === 'Escape') els.cancel.click();
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
const GRADE_MARK = { best: '', good: '', inaccuracy: '?!', mistake: '?', blunder: '??' };

const flat = (moves) => moves.flat();
const hkey = (hist) => Array.from(hist).join(',');
const moveName = (m) => `${sqName(m[0])}-${sqName(m[1])}(${sqName(m[2])})`;
const isFlipped = () => !isAnalysis() && settings.opponent !== 'human' && settings.human === 1;

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
  if (explainCache.has(key)) return;
  const beforeKey = hkey(hist.slice(0, 3 * (n - 1)));
  const before = analysisCache.get(beforeKey) || (await pendingAnalysis.get(beforeKey));
  const after = analysisCache.get(key);
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

// Explanation card for move number `n` (1-based ply) `mv`.
function explanationHtml(ex, n, mv, who = '') {
  const num = `${Math.floor((n - 1) / 2) + 1}. ${colorName((n - 1) % 2)}${who}`;
  const li = (p) => `<li class="${p.positive ? 'pos' : 'neg'}">${esc(p.text)}</li>`;
  let html = `<p class="ex-head"><span class="grade g-${ex.grade}">${ex.label}</span> ${num} ${moveName(mv)}</p>`;
  html += `<p class="ex-sum">${esc(ex.summary)}</p>`;
  if (ex.points.length) html += `<ul class="ex-pts">${ex.points.map(li).join('')}</ul>`;
  if (ex.replyPoints.length) html += `<ul class="ex-pts threat">${ex.replyPoints.map((p) => `<li class="neg">${esc(p.text)}</li>`).join('')}</ul>`;
  if (ex.best) {
    html += `<p class="ex-best">${ex.grade === 'good' ? '비슷하게 좋은 대안' : '추천 수'}: <b>${ex.best.move}</b></p>`;
    if (ex.bestPoints.length) html += `<ul class="ex-pts">${ex.bestPoints.map(li).join('')}</ul>`;
  }
  html += `<p class="ex-bal">${esc(ex.balance)}</p>`;
  return html;
}

function renderAnalysis() {
  if (!isAnalysis() || !state) return;
  const n = plyOf();
  els.prev.disabled = busy || n === 0;
  els.next.disabled = busy || n >= line.length;
  els.analyzeAll.disabled = busy || line.length === 0;
  els.showBest.setAttribute('aria-pressed', String(showBest));
  els.showBest.classList.toggle('on', showBest);

  const info = analysisCache.get(hkey(state.history));
  if (state.over) {
    els.posInfo.textContent = `게임 종료 — ${colorName(state.winner)} 승리`;
  } else if (info) {
    const side = colorName(state.side);
    const p = Math.round(100 * moverP(info, state.side));
    const verdict = info.proven ? (moverP(info, state.side) > 0.5 ? '필승 확인' : '필패 확인') : `승률 ${p}%`;
    const best = info.best ? ` · 추천 ${info.best.move}` : '';
    els.posInfo.textContent = `${side} 차례 · ${side} ${verdict}${best}`;
  } else {
    els.posInfo.textContent = analyzingNow ? '이 국면을 분석하는 중…' : '';
  }

  const ex = currentExplanation();
  if (!ex) {
    els.explain.innerHTML = n === 0
      ? '<p class="muted">수를 두면 그 수가 왜 좋은지, 왜 아쉬운지 설명해 드려요.</p>'
      : analyzingNow ? '<p class="muted">직전 수를 분석하는 중…</p>' : '';
  } else {
    els.explain.innerHTML = explanationHtml(ex, n, state.history.slice(3 * (n - 1), 3 * n));
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
    const cls = ['mv', i === n - 1 ? 'cur' : '', i >= n ? 'ahead' : '', ex ? `g-${ex.grade}` : ''].join(' ');
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
  if (td && isAnalysis()) navigate(Number(td.dataset.ply));
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

const coachOn = () => !isAnalysis() && settings.opponent !== 'human' && els.coach.checked;

// Starts the background analysis of the current position (human to move).
function coachPrepare() {
  const hist = Array.from(state.history);
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

// After the AI replied: explain the human's move and the AI's reply.
async function coachAfterAi(beforeAi, aiInfo, aiMove, my) {
  const token = analysisToken;
  const n = beforeAi.length / 3; // plies up to and including the human move
  if (n > 0 && (n - 1) % 2 === settings.human) {
    coachBusy = true;
    render();
    await explainPrefix(beforeAi, n, token);
    coachBusy = false;
  }
  if (my !== epoch || token !== analysisToken) return;
  // The AI's move, explained as if best: this lists what it aims at.
  const withAi = beforeAi.concat(aiMove);
  const p = moverP(aiInfo, n % 2);
  const r = await call('explain', { history: withAi, ply: n, best: aiMove, reply: [], pBefore: p, pAfter: p });
  if (my !== epoch || token !== analysisToken || !r.result) return;
  explainCache.set(hkey(withAi), r.result);
  render();
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

// Engine-move texts are written from the mover's (engine's) point of view;
// turn "내/상대" around for the human reading them.
function fromHumanView(text) {
  return text.replace(/내 /g, '\u0001').replace(/상대 /g, '내 ').replace(/\u0001/g, '엔진의 ');
}

function renderCoach() {
  const on = coachOn() && state;
  els.coachPanel.hidden = !on;
  els.coachLabel.hidden = isAnalysis() || settings.opponent === 'human';
  if (!on) return;
  const H = state.history;
  const plies = H.length / 3;
  // Latest human move and the AI move after it.
  let mine = -1;
  for (let i = plies - 1; i >= 0; i--) {
    if (i % 2 === settings.human) {
      mine = i;
      break;
    }
  }
  const exMine = mine >= 0 ? explainCache.get(hkey(H.slice(0, 3 * (mine + 1)))) : null;
  const exAi = mine >= 0 && mine + 1 < plies ? explainCache.get(hkey(H.slice(0, 3 * (mine + 2)))) : null;
  let html = '';
  if (exMine) {
    html += explanationHtml(exMine, mine + 1, H.slice(3 * mine, 3 * mine + 3), ' (나)');
  } else if (mine >= 0 && coachBusy) {
    html += '<p class="muted">내 수를 분석하는 중…</p>';
  } else if (mine < 0) {
    html += '<p class="muted">수를 두면 엔진이 응수한 뒤 그 수가 왜 좋은지, 왜 아쉬운지 알려 드려요.</p>';
  }
  if (exAi) {
    const pts = exAi.points.filter((p) => p.positive).slice(0, 2);
    const mv = H.slice(3 * (mine + 1), 3 * (mine + 2));
    html += `<div class="ai-intent"><p class="ex-head">엔진의 수 ${moveName(mv)}</p>`;
    html += pts.length
      ? `<ul class="ex-pts">${pts.map((p) => `<li class="pos">${esc(fromHumanView(p.text))}</li>`).join('')}</ul>`
      : '<p class="muted">큰 변화 없이 균형을 유지하는 수예요.</p>';
    html += '</div>';
  }
  if (state.over) html += coachSummary();
  els.coachBody.innerHTML = html;

  const info = analysisCache.get(hkey(H));
  const ready = isHumanTurn() && info && info.best;
  els.hintBtn.disabled = !isHumanTurn() || (!ready && !pendingAnalysis.size);
  els.hintBtn.textContent = !isHumanTurn()
    ? '힌트'
    : ready ? (coachHint ? '힌트 숨기기' : '힌트 (추천 수 보기)') : '힌트 준비 중…';
  els.hintBtn.classList.toggle('on', coachHint && ready);
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
  const labels = { best: '최선', good: '좋음', inaccuracy: '부정확', mistake: '실수', blunder: '대실수' };
  const parts = Object.keys(counts).map(
    (k) => `<span class="grade g-${k}">${labels[k]} ${counts[k]}</span>`,
  );
  const won = state.winner === settings.human;
  return `<div class="coach-sum"><p class="ex-head">${won ? '승리했어요!' : '대국 종료'} · 내 수 ${total}개 평가</p><p class="sum-row">${parts.join(' ')}</p></div>`;
}

els.hintBtn.addEventListener('click', () => {
  coachHint = !coachHint;
  render();
});
els.coach.addEventListener('change', () => {
  try { localStorage.setItem('coach', els.coach.checked ? '1' : '0'); } catch {}
  if (coachOn() && isHumanTurn()) coachPrepare();
  render();
});
try { if (localStorage.getItem('coach') === '0') els.coach.checked = false; } catch {}

// Expose for debugging and automated checks.
window.amazons = {
  call,
  combineInfos,
  get state() { return state; },
  get lastAiInfo() { return lastAiInfo; },
  get line() { return line; },
  analysisCache,
  explainCache,
  get analyzing() { return analyzingNow; },
  get busy() { return busy; },
  get coachBusy() { return coachBusy || pendingAnalysis.size > 0; },
};

buildBoard();
newGame();
