//! Game of the Amazons engine.

pub mod board;
pub mod naive;
pub mod rng;
pub mod wasm;

pub use board::{Board, Color, Move};

/// Picks a uniformly random legal move, or `None` if the side to move has lost.
pub fn random_move(board: &Board, rng: &mut rng::Rng) -> Option<Move> {
    let moves = board.legal_moves();
    if moves.is_empty() {
        None
    } else {
        Some(moves[rng.below(moves.len())])
    }
}
