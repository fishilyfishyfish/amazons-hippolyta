//! JavaScript bindings. The web UI talks to a single `Game` living inside a
//! Web Worker.

use wasm_bindgen::prelude::*;

use crate::board::{Bits, Board, Color, Move};
use crate::rng::Rng;

#[wasm_bindgen]
pub struct Game {
    board: Board,
    history: Vec<Move>,
    rng: Rng,
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
        }
    }

    pub fn reset(&mut self) {
        self.board = Board::new();
        self.history.clear();
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
}
