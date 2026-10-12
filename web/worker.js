// Owns the single engine instance. The UI thread never calls WASM directly,
// so engine work (later: search) can never freeze the page.
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
  };
}

const handlers = {
  state() {
    return true;
  },
  newGame() {
    game.reset();
    return true;
  },
  play({ move }) {
    return game.play(move[0], move[1], move[2]);
  },
  undo({ count }) {
    let n = 0;
    while (n < count && game.undo()) n++;
    return n;
  },
  randomMove() {
    const m = game.random_move();
    if (m.length !== 3) return false;
    return game.play(m[0], m[1], m[2]);
  },
};

self.onmessage = async (e) => {
  const { id, type, ...args } = e.data;
  try {
    await ready;
    const result = handlers[type](args);
    const state = snapshot();
    self.postMessage({ id, result, state }, [state.legal.buffer]);
  } catch (err) {
    self.postMessage({ id, error: String(err && err.stack || err) });
  }
};
