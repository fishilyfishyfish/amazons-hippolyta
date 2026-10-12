// Owns the single engine instance. The UI thread never calls WASM directly,
// so engine work (search) can never freeze the page.
import init, { Game } from './pkg/amazons_engine.js';

let game = null;
const ready = init().then(() => {
  game = new Game(Math.floor(Math.random() * 2 ** 32));
});

function snapshot() {
  const notation = game.history_notation();
  return {
    cells: game.cells(),
    side: game.side_to_move(),
    legal: game.legal_moves(),
    history: game.history(),
    notation: notation ? notation.split(' ') : [],
    over: game.is_over(),
    winner: game.winner(),
    evaluation: JSON.parse(game.evaluation()),
    dist: game.queen_distances(),
  };
}

const play = (m) => m.length === 3 && game.play(m[0], m[1], m[2]);

// Runs MCTS for `ms` milliseconds, posting progress about every 250 ms.
function search(ms, id) {
  game.search_start();
  const start = performance.now();
  let lastReport = start;
  for (;;) {
    game.search_run(32);
    const now = performance.now();
    if (now - start >= ms) break;
    if (now - lastReport >= 250) {
      lastReport = now;
      self.postMessage({ id, progress: JSON.parse(game.search_info(5)) });
    }
  }
  const info = JSON.parse(game.search_info(5));
  info.ms = Math.round(performance.now() - start);
  return info;
}

const handlers = {
  state: () => true,
  newGame() {
    game.reset();
    return true;
  },
  play: ({ move }) => game.play(move[0], move[1], move[2]),
  undo({ count }) {
    let n = 0;
    while (n < count && game.undo()) n++;
    return n;
  },
  aiMove({ kind, ms }, id) {
    if (kind === 'random') return { ok: play(game.random_move()) };
    if (kind === 'greedy') return { ok: play(game.greedy_move()) };
    const info = search(ms, id);
    return { ok: play(game.search_best()), info };
  },
};

self.onmessage = async (e) => {
  const { id, type, ...args } = e.data;
  try {
    await ready;
    const result = handlers[type](args, id);
    const state = snapshot();
    self.postMessage({ id, result, state }, [state.legal.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String((err && err.stack) || err) });
  }
};
