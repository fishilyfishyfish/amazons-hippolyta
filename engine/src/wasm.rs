//! JavaScript bindings. The web UI talks to a single `Game` living inside a
//! Web Worker.

use wasm_bindgen::prelude::*;

use crate::board::{Bits, Board, Color, Move};
use crate::eval::{evaluate_terms, queen_distance_map, EvalWeights};
use crate::rng::Rng;
use crate::search::{greedy_move, Mcts, MctsConfig};

#[wasm_bindgen]
pub struct Game {
    board: Board,
    history: Vec<Move>,
    rng: Rng,
    search: Option<Mcts>,
}

fn bits_to_vec(b: u128) -> Vec<u8> {
    Bits(b).collect()
}

#[wasm_bindgen]
impl Game {
    #[wasm_bindgen(constructor)]
    pub fn new(seed: f64) -> Game {
        Game {
            board: Board::new(),
            history: Vec::new(),
            rng: Rng::new(seed as u64),
            search: None,
        }
    }

    pub fn reset(&mut self) {
        self.board = Board::new();
        self.history.clear();
        self.search = None;
    }

    /// 100 cells, index = row*10+col: 0 empty, 1 white, 2 black, 3 arrow.
    pub fn cells(&self) -> Vec<u8> {
        (0..100).map(|sq| self.board.cell(sq)).collect()
    }

    /// 0 = white, 1 = black.
    pub fn side_to_move(&self) -> u8 {
        self.board.side as u8
    }

    pub fn ply(&self) -> u32 {
        self.history.len() as u32
    }

    /// Flat `[from, to, arrow, from, to, arrow, ...]`.
    pub fn legal_moves(&self) -> Vec<u8> {
        self.board
            .legal_moves()
            .iter()
            .flat_map(|m| [m.from, m.to, m.arrow])
            .collect()
    }

    pub fn move_count(&self) -> u32 {
        self.board.count_moves() as u32
    }

    pub fn destinations(&self, from: u8) -> Vec<u8> {
        if from >= 100 {
            return Vec::new();
        }
        bits_to_vec(self.board.destinations(from))
    }

    pub fn arrow_targets(&self, from: u8, to: u8) -> Vec<u8> {
        if from >= 100 || to >= 100 || self.board.destinations(from) & (1u128 << to) == 0 {
            return Vec::new();
        }
        bits_to_vec(self.board.arrow_targets(from, to))
    }

    /// Plays the move if legal; returns whether it was played.
    pub fn play(&mut self, from: u8, to: u8, arrow: u8) -> bool {
        let m = Move::new(from, to, arrow);
        if !self.board.is_legal(m) {
            return false;
        }
        self.board.make_move(m);
        self.history.push(m);
        true
    }

    /// Plays a move given in notation, e.g. `d1-d7(g7)`.
    pub fn play_notation(&mut self, text: &str) -> bool {
        match text.parse::<Move>() {
            Ok(m) => self.play(m.from, m.to, m.arrow),
            Err(_) => false,
        }
    }

    pub fn undo(&mut self) -> bool {
        match self.history.pop() {
            Some(m) => {
                self.board.unmake_move(m);
                true
            }
            None => false,
        }
    }

    pub fn is_over(&self) -> bool {
        self.board.is_game_over()
    }

    /// -1 while the game is running, otherwise 0 = white won, 1 = black won.
    pub fn winner(&self) -> i32 {
        match self.board.winner() {
            None => -1,
            Some(Color::White) => 0,
            Some(Color::Black) => 1,
        }
    }

    /// `[from, to, arrow]` of a uniformly random legal move (not played), or
    /// an empty array if there is none.
    pub fn random_move(&mut self) -> Vec<u8> {
        match crate::random_move(&self.board, &mut self.rng) {
            Some(m) => vec![m.from, m.to, m.arrow],
            None => Vec::new(),
        }
    }

    /// Flat `[from, to, arrow, ...]` of the moves played so far.
    pub fn history(&self) -> Vec<u8> {
        self.history
            .iter()
            .flat_map(|m| [m.from, m.to, m.arrow])
            .collect()
    }

    /// Moves played so far in notation, separated by single spaces.
    pub fn history_notation(&self) -> String {
        self.history
            .iter()
            .map(|m| m.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Static evaluation from White's point of view, for display. The
    /// side-to-move tie bonus is left out so the number does not swing by
    /// the tempo every ply. JSON:
    /// `{"score":..,"t1":..,"t2":..,"c1":..,"c2":..,"w":..,"mobility":..,"phase":..}`.
    pub fn evaluation(&self) -> String {
        let weights = EvalWeights {
            tie: 0.0,
            ..EvalWeights::default()
        };
        let t = evaluate_terms(&self.board, &weights);
        let s = if self.board.side == Color::White {
            1.0
        } else {
            -1.0
        };
        format!(
            "{{\"score\":{},\"t1\":{},\"t2\":{},\"c1\":{},\"c2\":{},\"w\":{},\"mobility\":{},\"phase\":{}}}",
            s * t.score,
            s * t.t1,
            s * t.t2,
            s * t.c1,
            s * t.c2,
            t.w,
            s * t.mobility,
            t.phase
        )
    }

    /// Queen distances: 100 values for White then 100 for Black (255 =
    /// unreachable or occupied).
    pub fn queen_distances(&self) -> Vec<u8> {
        let [w, b] = queen_distance_map(&self.board);
        w.iter().chain(b.iter()).copied().collect()
    }

    /// `[from, to, arrow]` of the best one-ply move by static evaluation.
    pub fn greedy_move(&self) -> Vec<u8> {
        match greedy_move(&self.board, &EvalWeights::default()) {
            Some(m) => vec![m.from, m.to, m.arrow],
            None => Vec::new(),
        }
    }

    /// `[from, to, arrow]` of the move a difficulty-level opponent plays
    /// (levels 1–4; level 5 is the full parallel search driven from JS).
    /// Levels 1–3 are human-like (`human.rs`) with a per-game style derived
    /// from `game_seed`; level 4 is a short fixed-iteration MCTS. Empty if
    /// there is no legal move.
    pub fn level_move(&self, level: u8, game_seed: f64, move_seed: f64) -> Vec<u8> {
        if !self.board.has_moves() {
            return Vec::new();
        }
        let mut rng = Rng::new(move_seed as u64);
        let m = match crate::players::level_spec(level) {
            crate::players::PlayerSpec::Human(l) => {
                let style = crate::human::game_style(&l, &mut Rng::new(game_seed as u64));
                crate::human::human_move(&self.board, &self.history, &l, &style, &mut rng)
            }
            _ => {
                let cfg = MctsConfig {
                    seed: move_seed as u64,
                    ..MctsConfig::default()
                };
                let mut t = Mcts::new(self.board, cfg);
                t.run(crate::players::LEVEL4_ITERATIONS);
                t.best_move()
            }
        };
        match m {
            Some(m) => vec![m.from, m.to, m.arrow],
            None => Vec::new(),
        }
    }

    /// Starts a fresh MCTS on the current position (default config).
    pub fn search_start(&mut self) {
        self.search = Some(Mcts::new(self.board, MctsConfig::default()));
    }

    /// Starts a search whose move ordering is perturbed by `jitter`, so that
    /// several workers searching the same position explore differently.
    pub fn search_start_seeded(&mut self, seed: f64, jitter: f32) {
        let cfg = MctsConfig {
            seed: seed as u64,
            jitter,
            ..MctsConfig::default()
        };
        self.search = Some(Mcts::new(self.board, cfg));
    }

    /// Replaces the game with the given flat `[from, to, arrow, ...]` move
    /// list (used by helper workers). Returns false if a move is illegal.
    pub fn load_history(&mut self, moves: &[u8]) -> bool {
        self.reset();
        for m in moves.chunks_exact(3) {
            if !self.play(m[0], m[1], m[2]) {
                return false;
            }
        }
        true
    }

    /// Runs `n` more iterations; returns the total so far.
    pub fn search_run(&mut self, n: u32) -> u32 {
        match &mut self.search {
            Some(t) if *t.root() == self.board => {
                t.run(n);
                t.iterations()
            }
            _ => 0,
        }
    }

    /// `[from, to, arrow]` of the most visited root move, or empty.
    pub fn search_best(&self) -> Vec<u8> {
        match self.search.as_ref().and_then(|t| t.best_move()) {
            Some(m) => vec![m.from, m.to, m.arrow],
            None => Vec::new(),
        }
    }

    /// Search summary as JSON. `value` and each candidate's `q` are in
    /// [-1, 1] from White's point of view.
    pub fn search_info(&self, max_candidates: usize) -> String {
        let Some(t) = &self.search else {
            return "null".into();
        };
        let s = if t.root().side == Color::White {
            1.0
        } else {
            -1.0
        };
        let pv: Vec<String> = t.pv(8).iter().map(|m| format!("\"{m}\"")).collect();
        let cands: Vec<String> = t
            .candidates()
            .iter()
            .take(max_candidates)
            .map(|c| {
                let q = if c.q.is_nan() {
                    "null".to_string()
                } else {
                    (s * c.q).to_string()
                };
                format!(
                    "{{\"move\":\"{}\",\"sq\":[{},{},{}],\"visits\":{},\"q\":{},\"prior\":{}}}",
                    c.mv, c.mv.from, c.mv.to, c.mv.arrow, c.visits, q, c.prior
                )
            })
            .collect();
        format!(
            "{{\"iterations\":{},\"nodes\":{},\"proven\":{},\"value\":{},\"pv\":[{}],\"candidates\":[{}]}}",
            t.iterations(),
            t.node_count(),
            t.root_proven() as f32 * s,
            s * t.root_value(),
            pv.join(","),
            cands.join(",")
        )
    }
}

/// Explains move number `ply` (0-based) of the flat `history`
/// (`[from, to, arrow, ...]`). `best` is the engine's recommended move in the
/// position before it and `reply` its best answer in the position after it
/// (`[from, to, arrow]` or empty); `p_before` / `p_after`
/// are the mover's win probabilities before (with best play) and after the
/// move. Returns JSON (see `Explanation::to_json`), or `null` on bad input.
#[wasm_bindgen]
pub fn explain_move(
    history: &[u8],
    ply: u32,
    best: &[u8],
    reply: &[u8],
    p_before: f32,
    p_after: f32,
) -> String {
    let ply = ply as usize;
    if history.len() < 3 * (ply + 1) {
        return "null".into();
    }
    let mut board = Board::new();
    for m in history[..3 * ply].chunks_exact(3) {
        let m = Move::new(m[0], m[1], m[2]);
        if !board.is_legal(m) {
            return "null".into();
        }
        board.make_move(m);
    }
    let h = &history[3 * ply..3 * ply + 3];
    let played = Move::new(h[0], h[1], h[2]);
    if !board.is_legal(played) {
        return "null".into();
    }
    let best = (best.len() == 3)
        .then(|| Move::new(best[0], best[1], best[2]))
        .filter(|m| board.is_legal(*m));
    let reply = (reply.len() == 3).then(|| Move::new(reply[0], reply[1], reply[2]));
    crate::explain::explain(&board, played, best, reply, p_before, p_after).to_json()
}
