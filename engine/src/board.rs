//! Bitboard board representation and move generation.
//!
//! Squares are numbered `row * 10 + col`, where row 0 is rank 1 and col 0 is
//! file `a`. Square `n` is bit `n` of a `u128`; bits 100..128 are always zero.

use std::fmt;

pub const SIZE: usize = 10;
pub const NUM_SQUARES: usize = SIZE * SIZE;
pub const BOARD_MASK: u128 = (1u128 << NUM_SQUARES) - 1;

pub type Bitboard = u128;

#[inline(always)]
pub const fn bit(sq: u8) -> Bitboard {
    1u128 << sq
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[inline(always)]
    pub fn opposite(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    #[inline(always)]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// One full move: amazon `from` -> `to`, then an arrow shot from `to` to `arrow`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Move {
    pub from: u8,
    pub to: u8,
    pub arrow: u8,
}

impl Move {
    pub const fn new(from: u8, to: u8, arrow: u8) -> Move {
        Move { from, to, arrow }
    }
}

pub fn square_name(sq: u8) -> String {
    let file = (b'a' + sq % 10) as char;
    format!("{}{}", file, sq / 10 + 1)
}

/// Parses `a1`..`j10` into a square index.
pub fn parse_square(s: &str) -> Option<u8> {
    let bytes = s.as_bytes();
    if bytes.len() < 2 || bytes.len() > 3 {
        return None;
    }
    let file = bytes[0].to_ascii_lowercase();
    if !(b'a'..=b'j').contains(&file) {
        return None;
    }
    let rank: u8 = s[1..].parse().ok()?;
    if !(1..=10).contains(&rank) || s[1..].starts_with('0') {
        return None;
    }
    Some((rank - 1) * 10 + (file - b'a'))
}

impl fmt::Display for Move {
    /// Notation `from-to(arrow)`, e.g. `d1-d7(g7)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{}({})",
            square_name(self.from),
            square_name(self.to),
            square_name(self.arrow)
        )
    }
}

impl std::str::FromStr for Move {
    type Err = String;

    fn from_str(s: &str) -> Result<Move, String> {
        let s = s.trim();
        let err = || format!("invalid move notation: {s:?}");
        let (from, rest) = s.split_once('-').ok_or_else(err)?;
        let (to, rest) = rest.split_once('(').ok_or_else(err)?;
        let arrow = rest.strip_suffix(')').ok_or_else(err)?;
        Ok(Move::new(
            parse_square(from.trim()).ok_or_else(err)?,
            parse_square(to.trim()).ok_or_else(err)?,
            parse_square(arrow.trim()).ok_or_else(err)?,
        ))
    }
}

// Ray directions. The first four increase the square index (scan with
// trailing_zeros for the nearest blocker), the last four decrease it
// (scan with leading_zeros).
const DIRS: [(i32, i32); 8] = [
    (0, 1),   // +1  east
    (1, 0),   // +10 north
    (1, 1),   // +11 north-east
    (1, -1),  // +9  north-west
    (0, -1),  // -1  west
    (-1, 0),  // -10 south
    (-1, -1), // -11 south-west
    (-1, 1),  // -9  south-east
];

const fn build_rays() -> [[Bitboard; 8]; NUM_SQUARES] {
    let mut rays = [[0u128; 8]; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        let mut d = 0;
        while d < 8 {
            let (dr, dc) = DIRS[d];
            let mut r = (sq / SIZE) as i32 + dr;
            let mut c = (sq % SIZE) as i32 + dc;
            let mut ray = 0u128;
            while r >= 0 && r < SIZE as i32 && c >= 0 && c < SIZE as i32 {
                ray |= 1u128 << (r as usize * SIZE + c as usize);
                r += dr;
                c += dc;
            }
            rays[sq][d] = ray;
            d += 1;
        }
        sq += 1;
    }
    rays
}

const fn build_neighbors() -> [Bitboard; NUM_SQUARES] {
    let mut out = [0u128; NUM_SQUARES];
    let mut sq = 0;
    while sq < NUM_SQUARES {
        let mut d = 0;
        while d < 8 {
            let (dr, dc) = DIRS[d];
            let r = (sq / SIZE) as i32 + dr;
            let c = (sq % SIZE) as i32 + dc;
            if r >= 0 && r < SIZE as i32 && c >= 0 && c < SIZE as i32 {
                out[sq] |= 1u128 << (r as usize * SIZE + c as usize);
            }
            d += 1;
        }
        sq += 1;
    }
    out
}

static RAYS: [[Bitboard; 8]; NUM_SQUARES] = build_rays();
static NEIGHBORS: [Bitboard; NUM_SQUARES] = build_neighbors();

/// Squares reachable from `sq` by a queen move, stopping before any occupied
/// square. `sq` itself must not matter for the result (it never lies on its
/// own rays).
#[inline]
pub fn queen_reach(sq: u8, occ: Bitboard) -> Bitboard {
    let rays = &RAYS[sq as usize];
    let mut reach = 0u128;
    for d in 0..4 {
        let ray = rays[d];
        let blockers = ray & occ;
        reach |= if blockers == 0 {
            ray
        } else {
            let b = blockers.trailing_zeros() as usize;
            ray ^ RAYS[b][d] ^ (1u128 << b)
        };
    }
    for d in 4..8 {
        let ray = rays[d];
        let blockers = ray & occ;
        reach |= if blockers == 0 {
            ray
        } else {
            let b = 127 - blockers.leading_zeros() as usize;
            ray ^ RAYS[b][d] ^ (1u128 << b)
        };
    }
    reach
}

#[inline(always)]
pub fn neighbors(sq: u8) -> Bitboard {
    NEIGHBORS[sq as usize]
}

/// Iterates over the set bits of a bitboard, lowest first.
pub struct Bits(pub Bitboard);

impl Iterator for Bits {
    type Item = u8;
    #[inline(always)]
    fn next(&mut self) -> Option<u8> {
        if self.0 == 0 {
            None
        } else {
            let sq = self.0.trailing_zeros() as u8;
            self.0 &= self.0 - 1;
            Some(sq)
        }
    }
}

pub const WHITE_START: [&str; 4] = ["a4", "d1", "g1", "j4"];
pub const BLACK_START: [&str; 4] = ["a7", "d10", "g10", "j7"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Board {
    /// Amazons per color, indexed by `Color::index()`.
    pub amazons: [Bitboard; 2],
    pub arrows: Bitboard,
    pub side: Color,
}

impl Default for Board {
    fn default() -> Self {
        Board::new()
    }
}

impl Board {
    pub fn new() -> Board {
        let mut amazons = [0u128; 2];
        for s in WHITE_START {
            amazons[0] |= bit(parse_square(s).unwrap());
        }
        for s in BLACK_START {
            amazons[1] |= bit(parse_square(s).unwrap());
        }
        Board {
            amazons,
            arrows: 0,
            side: Color::White,
        }
    }

    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.amazons[0] | self.amazons[1] | self.arrows
    }

    #[inline(always)]
    pub fn empty(&self) -> Bitboard {
        !self.occupied() & BOARD_MASK
    }

    /// 0 = empty, 1 = white amazon, 2 = black amazon, 3 = arrow.
    pub fn cell(&self, sq: u8) -> u8 {
        let b = bit(sq);
        if self.amazons[0] & b != 0 {
            1
        } else if self.amazons[1] & b != 0 {
            2
        } else if self.arrows & b != 0 {
            3
        } else {
            0
        }
    }

    /// Appends all legal moves for the side to move.
    pub fn generate_moves(&self, out: &mut Vec<Move>) {
        let occ = self.occupied();
        for from in Bits(self.amazons[self.side.index()]) {
            let occ_wo = occ ^ bit(from);
            for to in Bits(queen_reach(from, occ_wo)) {
                for arrow in Bits(queen_reach(to, occ_wo)) {
                    out.push(Move::new(from, to, arrow));
                }
            }
        }
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        let mut v = Vec::with_capacity(2200);
        self.generate_moves(&mut v);
        v
    }

    /// Number of legal moves, without materialising them.
    #[inline]
    pub fn count_moves(&self) -> u64 {
        let occ = self.occupied();
        let mut n = 0u64;
        for from in Bits(self.amazons[self.side.index()]) {
            let occ_wo = occ ^ bit(from);
            for to in Bits(queen_reach(from, occ_wo)) {
                n += queen_reach(to, occ_wo).count_ones() as u64;
            }
        }
        n
    }

    /// Destinations the amazon on `from` can move to (empty if `from` is not
    /// an amazon of the side to move).
    pub fn destinations(&self, from: u8) -> Bitboard {
        if self.amazons[self.side.index()] & bit(from) == 0 {
            return 0;
        }
        queen_reach(from, self.occupied())
    }

    /// Arrow targets after moving `from` -> `to` (assumes that step is legal).
    pub fn arrow_targets(&self, from: u8, to: u8) -> Bitboard {
        queen_reach(to, self.occupied() ^ bit(from))
    }

    /// The side to move has at least one legal move. An amazon that can step
    /// to an adjacent empty square can always shoot back at the vacated
    /// square, and any queen move passes over an adjacent empty square first,
    /// so this reduces to a neighbor check.
    #[inline]
    pub fn has_moves(&self) -> bool {
        let empty = self.empty();
        Bits(self.amazons[self.side.index()]).any(|sq| neighbors(sq) & empty != 0)
    }

    pub fn is_game_over(&self) -> bool {
        !self.has_moves()
    }

    /// Winner if the game is over: the player who cannot move loses.
    pub fn winner(&self) -> Option<Color> {
        if self.has_moves() {
            None
        } else {
            Some(self.side.opposite())
        }
    }

    pub fn is_legal(&self, m: Move) -> bool {
        if m.from as usize >= NUM_SQUARES
            || m.to as usize >= NUM_SQUARES
            || m.arrow as usize >= NUM_SQUARES
        {
            return false;
        }
        if self.amazons[self.side.index()] & bit(m.from) == 0 {
            return false;
        }
        let occ_wo = self.occupied() ^ bit(m.from);
        queen_reach(m.from, occ_wo) & bit(m.to) != 0
            && queen_reach(m.to, occ_wo) & bit(m.arrow) != 0
    }

    /// Applies a move assumed to be legal.
    #[inline]
    pub fn make_move(&mut self, m: Move) {
        let s = self.side.index();
        self.amazons[s] ^= bit(m.from) | bit(m.to);
        self.arrows |= bit(m.arrow);
        self.side = self.side.opposite();
    }

    /// Reverts `make_move(m)`.
    #[inline]
    pub fn unmake_move(&mut self, m: Move) {
        self.side = self.side.opposite();
        let s = self.side.index();
        self.arrows ^= bit(m.arrow);
        self.amazons[s] ^= bit(m.from) | bit(m.to);
    }

    /// Leaf count at `depth` plies (bulk-counted at the last ply).
    pub fn perft(&mut self, depth: u32) -> u64 {
        match depth {
            0 => 1,
            1 => self.count_moves(),
            _ => {
                let moves = self.legal_moves();
                let mut n = 0;
                for m in moves {
                    self.make_move(m);
                    n += self.perft(depth - 1);
                    self.unmake_move(m);
                }
                n
            }
        }
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in (0..SIZE as u8).rev() {
            write!(f, "{:>2} ", row + 1)?;
            for col in 0..SIZE as u8 {
                let c = match self.cell(row * 10 + col) {
                    1 => 'W',
                    2 => 'B',
                    3 => 'x',
                    _ => '.',
                };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        writeln!(f, "   a b c d e f g h i j")?;
        write!(f, "{:?} to move", self.side)
    }
}
