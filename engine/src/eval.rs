//! Static evaluation: territory by queen/king distance (Lieberum-style).
//!
//! For each color we compute, for every empty square, the minimum number of
//! queen moves (and king moves) any of its amazons needs to reach it, moving
//! only through empty squares. Distances are kept as one bitboard per level,
//! so the comparison terms reduce to ANDs and popcounts.

use crate::board::{queen_reach, Bitboard, Bits, Board, BOARD_MASK};

/// Distance levels tracked; anything farther counts as unreachable.
pub const MAX_DIST: usize = 16;
/// Marker for "unreachable" in per-square distance arrays.
pub const UNREACHABLE: u8 = 255;

/// Squares at each distance: `levels[d]` holds squares at distance exactly
/// `d` (index 0 unused). `depth` is the largest non-empty level.
#[derive(Clone, Copy, Debug)]
pub struct Levels {
    pub levels: [Bitboard; MAX_DIST + 1],
    pub reached: Bitboard,
    pub depth: usize,
}

impl Levels {
    pub fn to_array(&self) -> [u8; 100] {
        let mut out = [UNREACHABLE; 100];
        for d in 1..=self.depth {
            for sq in Bits(self.levels[d]) {
                out[sq as usize] = d as u8;
            }
        }
        out
    }
}

/// Queen-move distances from `sources` through empty squares (`occ` blocks).
pub fn queen_levels(sources: Bitboard, occ: Bitboard) -> Levels {
    let empty = !occ & BOARD_MASK;
    let mut lv = Levels {
        levels: [0; MAX_DIST + 1],
        reached: 0,
        depth: 0,
    };
    let mut frontier = sources;
    for d in 1..=MAX_DIST {
        let next = queen_attacks_set(frontier, empty) & !lv.reached;
        if next == 0 {
            break;
        }
        lv.levels[d] = next;
        lv.reached |= next;
        lv.depth = d;
        frontier = next;
    }
    lv
}

#[inline(always)]
fn sh(x: Bitboard, s: i32) -> Bitboard {
    if s > 0 {
        x << s
    } else {
        x >> -s
    }
}

/// Occluded fill (Kogge-Stone) of `gen` in direction `dir` through `prop`,
/// returning the squares attacked one step beyond the fill. `mask` removes
/// squares that a shift by `dir` would wrap onto (file a or j).
#[inline(always)]
fn ks_attacks(mut gen: Bitboard, prop: Bitboard, dir: i32, mask: Bitboard) -> Bitboard {
    let mut p = prop & mask;
    gen |= p & sh(gen, dir);
    p &= sh(p, dir);
    gen |= p & sh(gen, 2 * dir);
    p &= sh(p, 2 * dir);
    gen |= p & sh(gen, 4 * dir);
    p &= sh(p, 4 * dir);
    gen |= p & sh(gen, 8 * dir);
    sh(gen, dir) & mask & BOARD_MASK
}

/// Union of queen moves from every square in `set`, through `empty`.
#[inline]
pub fn queen_attacks_set(set: Bitboard, empty: Bitboard) -> Bitboard {
    (ks_attacks(set, empty, 1, NOT_FILE_A)
        | ks_attacks(set, empty, -1, NOT_FILE_J)
        | ks_attacks(set, empty, 10, BOARD_MASK)
        | ks_attacks(set, empty, -10, BOARD_MASK)
        | ks_attacks(set, empty, 11, NOT_FILE_A)
        | ks_attacks(set, empty, -11, NOT_FILE_J)
        | ks_attacks(set, empty, 9, NOT_FILE_J)
        | ks_attacks(set, empty, -9, NOT_FILE_A))
        & empty
}

const NOT_FILE_A: Bitboard = {
    let mut m = BOARD_MASK;
    let mut r = 0;
    while r < 10 {
        m &= !(1u128 << (r * 10));
        r += 1;
    }
    m
};
const NOT_FILE_J: Bitboard = {
    let mut m = BOARD_MASK;
    let mut r = 0;
    while r < 10 {
        m &= !(1u128 << (r * 10 + 9));
        r += 1;
    }
    m
};

/// All squares within one king step of `b` (including `b`).
#[inline(always)]
fn king_spread(b: Bitboard) -> Bitboard {
    let h = b | ((b << 1) & NOT_FILE_A) | ((b >> 1) & NOT_FILE_J);
    (h | (h << 10) | (h >> 10)) & BOARD_MASK
}

/// King-move distances from `sources` through empty squares.
pub fn king_levels(sources: Bitboard, occ: Bitboard) -> Levels {
    let empty = !occ & BOARD_MASK;
    let mut lv = Levels {
        levels: [0; MAX_DIST + 1],
        reached: 0,
        depth: 0,
    };
    let mut frontier = sources;
    for d in 1..=MAX_DIST {
        let next = king_spread(frontier) & empty & !lv.reached;
        if next == 0 {
            break;
        }
        lv.levels[d] = next;
        lv.reached |= next;
        lv.depth = d;
        frontier = next;
    }
    lv
}

/// Weights of the evaluation terms. All terms are from the side to move's
/// point of view and measured roughly in "squares".
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvalWeights {
    /// Value of a square both sides reach equally fast, for the side to move.
    pub tie: f32,
    /// `w` (sum of 2^-|dq_p - dq_o|) at which the position counts as a full
    /// opening; `phase = clamp(w / phase_w, 0, 1)`.
    pub phase_w: f32,
    /// t1 (queen territory) weight in the endgame / opening.
    pub t1_end: f32,
    pub t1_open: f32,
    /// Opening-phase weights of the remaining terms (scaled by `phase`).
    pub t2: f32,
    pub c1: f32,
    pub c2: f32,
    pub mobility: f32,
}

impl Default for EvalWeights {
    fn default() -> Self {
        EvalWeights {
            tie: 0.2,
            phase_w: 40.0,
            t1_end: 1.0,
            t1_open: 0.5,
            t2: 0.6,
            c1: 0.0,
            c2: 0.6,
            mobility: 1.0,
        }
    }
}

/// Raw terms, side-to-move perspective. Exposed for UI/debugging.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvalTerms {
    pub t1: f32,
    pub t2: f32,
    pub c1: f32,
    pub c2: f32,
    pub w: f32,
    pub mobility: f32,
    pub phase: f32,
    pub score: f32,
}

const POW2_NEG: [f32; MAX_DIST + 1] = {
    let mut t = [0f32; MAX_DIST + 1];
    let mut i = 0;
    let mut v = 1.0f32;
    while i <= MAX_DIST {
        t[i] = v;
        v *= 0.5;
        i += 1;
    }
    t
};

/// t (territory with tie bonus), plus the per-level-pair sums needed for
/// c1/c2/w. Returns (closer_p - closer_o + tie * ties).
fn territory(p: &Levels, o: &Levels, tie: f32) -> f32 {
    let mut cum_p = 0;
    let mut cum_o = 0;
    let mut p_closer = 0u32;
    let mut o_closer = 0u32;
    let mut ties = 0u32;
    let depth = p.depth.max(o.depth);
    for d in 1..=depth {
        let lp = p.levels[d];
        let lo = o.levels[d];
        // Reached at d by p and not reached by o at distance <= d.
        cum_o |= lo;
        cum_p |= lp;
        p_closer += (lp & !cum_o).count_ones();
        o_closer += (lo & !cum_p).count_ones();
        ties += (lp & lo).count_ones();
    }
    p_closer as f32 - o_closer as f32 + tie * ties as f32
}

/// Mobility penalty for one amazon with `m` reachable squares: trapped
/// amazons are heavily penalised, beyond ~10 squares it barely matters.
#[inline]
fn mobility_value(m: u32) -> f32 {
    -8.0 / (1.0 + m as f32)
}

pub fn evaluate_terms(board: &Board, wts: &EvalWeights) -> EvalTerms {
    let occ = board.occupied();
    let p_am = board.amazons[board.side.index()];
    let o_am = board.amazons[board.side.opposite().index()];

    let qp = queen_levels(p_am, occ);
    let qo = queen_levels(o_am, occ);
    let kp = king_levels(p_am, occ);
    let ko = king_levels(o_am, occ);

    let t1 = territory(&qp, &qo, wts.tie);
    let t2 = territory(&kp, &ko, wts.tie);

    let mut c1 = 0.0;
    for d in 1..=qp.depth {
        c1 += qp.levels[d].count_ones() as f32 * POW2_NEG[d];
    }
    for d in 1..=qo.depth {
        c1 -= qo.levels[d].count_ones() as f32 * POW2_NEG[d];
    }
    c1 *= 2.0;

    let mut w = 0.0;
    for dp in 1..=qp.depth {
        for dq in 1..=qo.depth {
            let n = (qp.levels[dp] & qo.levels[dq]).count_ones();
            if n != 0 {
                w += n as f32 * POW2_NEG[dp.abs_diff(dq)];
            }
        }
    }

    // c2: per square clamp((D_o - D_p) / 6, -1, 1) with king distances.
    let mut c2 = 0.0;
    for dp in 1..=kp.depth {
        for dq in 1..=ko.depth {
            let n = (kp.levels[dp] & ko.levels[dq]).count_ones();
            if n != 0 {
                c2 += n as f32 * ((dq as f32 - dp as f32) / 6.0).clamp(-1.0, 1.0);
            }
        }
    }
    c2 += (kp.reached & !ko.reached).count_ones() as f32;
    c2 -= (ko.reached & !kp.reached).count_ones() as f32;

    let mut mobility = 0.0;
    for sq in Bits(p_am) {
        mobility += mobility_value(queen_reach(sq, occ).count_ones());
    }
    for sq in Bits(o_am) {
        mobility -= mobility_value(queen_reach(sq, occ).count_ones());
    }

    let phase = (w / wts.phase_w).clamp(0.0, 1.0);
    let t1_w = wts.t1_end + (wts.t1_open - wts.t1_end) * phase;
    let score =
        t1_w * t1 + phase * (wts.t2 * t2 + wts.c1 * c1 + wts.c2 * c2 + wts.mobility * mobility);
    EvalTerms {
        t1,
        t2,
        c1,
        c2,
        w,
        mobility,
        phase,
        score,
    }
}

/// Score for the side to move (positive = good for the side to move).
/// Terminal positions are not special-cased here; callers check `has_moves`.
#[inline]
pub fn evaluate(board: &Board, wts: &EvalWeights) -> f32 {
    evaluate_terms(board, wts).score
}

/// Per-square queen distances for both colors (`[white, black]`, 255 =
/// unreachable, occupied squares are 255).
pub fn queen_distance_map(board: &Board) -> [[u8; 100]; 2] {
    let occ = board.occupied();
    [
        queen_levels(board.amazons[0], occ).to_array(),
        queen_levels(board.amazons[1], occ).to_array(),
    ]
}

/// Simple reference BFS (array stepping, no bitboard tricks) used by tests to
/// check `queen_levels` / `king_levels`.
pub fn reference_distances(board: &Board, sources: Bitboard, king: bool) -> [u8; 100] {
    let blocked = |r: i32, c: i32| board.cell((r * 10 + c) as u8) != 0;
    let mut dist = [UNREACHABLE; 100];
    let mut queue: Vec<(i32, i32)> = Bits(sources)
        .map(|s| (s as i32 / 10, s as i32 % 10))
        .collect();
    let mut cur = 0u8;
    while !queue.is_empty() && (cur as usize) < MAX_DIST {
        cur += 1;
        let mut next = Vec::new();
        for &(r, c) in &queue {
            for (dr, dc) in [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ] {
                let (mut nr, mut nc) = (r + dr, c + dc);
                while (0..10).contains(&nr) && (0..10).contains(&nc) && !blocked(nr, nc) {
                    let i = (nr * 10 + nc) as usize;
                    if dist[i] == UNREACHABLE {
                        dist[i] = cur;
                        next.push((nr, nc));
                    }
                    if king {
                        break;
                    }
                    nr += dr;
                    nc += dc;
                }
            }
        }
        queue = next;
    }
    dist
}

/// The two sides can no longer reach any common square: the game is decided
/// by counting (filling) each side's own region.
pub fn regions_separated(board: &Board) -> bool {
    let occ = board.occupied();
    let a = queen_levels(board.amazons[0], occ).reached;
    let b = queen_levels(board.amazons[1], occ).reached;
    a & b == 0
}
