//! Deliberately simple array-based reference implementation, used only to
//! cross-check the bitboard move generator. No shared code with `board.rs`
//! beyond the `Move` type.

use crate::board::Move;

pub const EMPTY: u8 = 0;
pub const WHITE: u8 = 1;
pub const BLACK: u8 = 2;
pub const ARROW: u8 = 3;

const DIRS: [(i32, i32); 8] = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NaiveBoard {
    /// `cells[row][col]`, row 0 = rank 1.
    pub cells: [[u8; 10]; 10],
    /// WHITE or BLACK.
    pub side: u8,
}

impl Default for NaiveBoard {
    fn default() -> Self {
        NaiveBoard::new()
    }
}

impl NaiveBoard {
    pub fn new() -> NaiveBoard {
        let mut cells = [[EMPTY; 10]; 10];
        // White: a4 d1 g1 j4, Black: a7 d10 g10 j7.
        for (r, c) in [(3, 0), (0, 3), (0, 6), (3, 9)] {
            cells[r][c] = WHITE;
        }
        for (r, c) in [(6, 0), (9, 3), (9, 6), (6, 9)] {
            cells[r][c] = BLACK;
        }
        NaiveBoard { cells, side: WHITE }
    }

    fn slides(&self, r: i32, c: i32) -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        for (dr, dc) in DIRS {
            let (mut nr, mut nc) = (r + dr, c + dc);
            while (0..10).contains(&nr)
                && (0..10).contains(&nc)
                && self.cells[nr as usize][nc as usize] == EMPTY
            {
                out.push((nr, nc));
                nr += dr;
                nc += dc;
            }
        }
        out
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        let mut b = self.clone();
        for r in 0..10 {
            for c in 0..10 {
                if self.cells[r][c] != self.side {
                    continue;
                }
                for (tr, tc) in self.slides(r as i32, c as i32) {
                    // Physically move the amazon, then look for arrow squares.
                    b.cells[r][c] = EMPTY;
                    b.cells[tr as usize][tc as usize] = self.side;
                    for (ar, ac) in b.slides(tr, tc) {
                        moves.push(Move::new(
                            (r * 10 + c) as u8,
                            (tr * 10 + tc) as u8,
                            (ar * 10 + ac) as u8,
                        ));
                    }
                    b.cells[tr as usize][tc as usize] = EMPTY;
                    b.cells[r][c] = self.side;
                }
            }
        }
        moves
    }

    pub fn make_move(&mut self, m: Move) {
        let (fr, fc) = (m.from as usize / 10, m.from as usize % 10);
        let (tr, tc) = (m.to as usize / 10, m.to as usize % 10);
        let (ar, ac) = (m.arrow as usize / 10, m.arrow as usize % 10);
        self.cells[fr][fc] = EMPTY;
        self.cells[tr][tc] = self.side;
        self.cells[ar][ac] = ARROW;
        self.side = if self.side == WHITE { BLACK } else { WHITE };
    }

    pub fn unmake_move(&mut self, m: Move) {
        self.side = if self.side == WHITE { BLACK } else { WHITE };
        let (fr, fc) = (m.from as usize / 10, m.from as usize % 10);
        let (tr, tc) = (m.to as usize / 10, m.to as usize % 10);
        let (ar, ac) = (m.arrow as usize / 10, m.arrow as usize % 10);
        self.cells[ar][ac] = EMPTY;
        self.cells[tr][tc] = EMPTY;
        self.cells[fr][fc] = self.side;
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        if depth == 0 {
            return 1;
        }
        let moves = self.legal_moves();
        if depth == 1 {
            return moves.len() as u64;
        }
        let mut n = 0;
        for m in moves {
            self.make_move(m);
            n += self.perft(depth - 1);
            self.unmake_move(m);
        }
        n
    }

    /// Same cell encoding as `Board::cell`.
    pub fn cell(&self, sq: u8) -> u8 {
        self.cells[sq as usize / 10][sq as usize % 10]
    }
}
