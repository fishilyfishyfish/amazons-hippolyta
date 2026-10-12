// UI thread: rendering and input only. All rules live in the engine worker.

const FILES = 'abcdefghij';
const sqName = (sq) => FILES[sq % 10] + (Math.floor(sq / 10) + 1);
const AI_DELAY_MS = 350;

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
};

// ---- worker RPC -----------------------------------------------------------

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
let nextId = 1;
const pending = new Map();
worker.onmessage = (e) => {
  const { id, error } = e.data;
  const p = pending.get(id);
  if (!p) return;
  pending.delete(id);
  if (error) p.reject(new Error(error));
  else p.resolve(e.data);
};
worker.onerror = (e) => showError(e.message || '워커 오류');

function call(type, args = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...args });
  });
}

// ---- state ----------------------------------------------------------------

let state = null;        // last snapshot from the worker
let tree = new Map();    // from -> Map(to -> Set(arrow))
let sel = { from: -1, to: -1 };
let busy = false;        // waiting on the worker
let epoch = 0;           // bumps on new game / undo to drop stale AI turns
const settings = { opponent: 'random', human: 0 };

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

const isHumanTurn = () =>
  state && !state.over && (settings.opponent === 'human' || state.side === settings.human);

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
  busy = true;
  render();
  await new Promise((r) => setTimeout(r, AI_DELAY_MS));
  if (my !== epoch) return;
  await run('randomMove');
  if (my === epoch) maybeAiTurn();
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
  const flip = settings.opponent !== 'human' && settings.human === 1;
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
  const flip = settings.opponent !== 'human' && settings.human === 1;

  for (let i = 0; i < 100; i++) {
    const sq = squareAt(i);
    const el = cellEls[i];
    el.dataset.sq = sq;
    const r = Math.floor(sq / 10), c = sq % 10;
    const cls = ['cell', (r + c) % 2 === 0 ? 'dark' : 'light'];
    const v = cells[sq];
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
  renderMoves();
  els.undo.disabled = busy || undoCount() === 0;
  els.cancel.disabled = sel.from < 0;
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
    const who = settings.opponent === 'human' ? '' : ' (당신)';
    text = `${colorName(s.side)} 차례${who} · 합법수 ${s.legal.length / 3}개`;
    hint = sel.to >= 0 ? '화살을 쏠 칸을 고르세요.'
      : sel.from >= 0 ? '이동할 칸을 고르세요.'
      : '움직일 아마존을 고르세요.';
  } else {
    text = `${colorName(s.side)} (랜덤 AI) 두는 중…`;
  }
  els.status.textContent = text;
  els.hint.textContent = hint;
}

function renderMoves() {
  const n = state.notation;
  const rows = [];
  for (let i = 0; i < n.length; i += 2) {
    const cur = (j) => (j === n.length - 1 ? ' class="cur"' : '');
    rows.push(
      `<tr><td>${i / 2 + 1}</td><td${cur(i)}>${n[i]}</td><td${cur(i + 1)}>${n[i + 1] || ''}</td></tr>`,
    );
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
      run('play', { move }).then(maybeAiTurn);
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
  if (settings.opponent === 'human') return plies > 0 ? 1 : 0;
  // Ply i (0-based) was played by colour i % 2.
  for (let i = plies - 1; i >= 0; i--) {
    if (i % 2 === settings.human) return plies - i;
  }
  return 0;
}

els.undo.addEventListener('click', async () => {
  const count = undoCount();
  if (!count) return;
  epoch++;
  await run('undo', { count });
});

els.cancel.addEventListener('click', () => {
  sel = { from: -1, to: -1 };
  render();
});

async function newGame() {
  settings.opponent = els.opponent.value;
  settings.human = Number(els.color.value);
  els.color.disabled = settings.opponent === 'human';
  epoch++;
  await run('newGame');
  maybeAiTurn();
}

els.newGame.addEventListener('click', newGame);
els.opponent.addEventListener('change', newGame);
els.color.addEventListener('change', newGame);

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') els.cancel.click();
  else if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
    e.preventDefault();
    els.undo.click();
  }
});

// Expose for debugging and automated checks.
window.amazons = { call, get state() { return state; } };

buildBoard();
newGame();
