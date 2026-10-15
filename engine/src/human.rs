//! Human-like opponents for the difficulty levels.
//!
//! A weaker engine that merely plays a random inferior move now and then does
//! not feel like a person: it plays perfectly and then drops a move nobody
//! would play. People lose games differently, so the weak levels model how
//! people choose a move:
//!
//! - **Attention**: only a handful of amazon steps get looked at. What catches
//!   the eye is what just happened (squares near the opponent's last move),
//!   an amazon of mine that is running out of room, contact with an opponent
//!   amazon, the amazon I moved last time, and moves that simply look good.
//!   Quiet moves far from the action are easily missed.
//! - **Arrows**: for each step only some arrows are tried, preferring arrows
//!   next to an opponent amazon or near the moved amazon.
//! - **Shallow reading**: the best-looking moves are checked against the
//!   opponent's replies that come to mind (those aimed at my amazons), so
//!   obvious traps are seen but quiet threats are not. The lowest level only
//!   checks its top two ideas against three or so replies.
//! - **Intuitive judgement**: the evaluation leans on what is visible (queen
//!   reach, freedom of the amazons) more than the engine's king-distance
//!   terms, with a per-game style (some games more aggressive, some more
//!   territorial) instead of per-move noise, so play stays consistent within a
//!   game.
//! - **Obvious moves**: an immediate win is usually noticed (always from
//!   level 2), and an amazon in danger draws attention.

use crate::board::{bit, neighbors, queen_reach, Bits, Board, Move, BOARD_MASK};
use crate::eval::{evaluate, regions_separated, EvalWeights};
use crate::rng::Rng;

/// Parameters of one human-like level.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HumanLevel {
    /// Amazon steps (from, to) looked at.
    pub vision: usize,
    /// How strongly a step's apparent quality draws attention (per square of
    /// evaluation). 0 = attention is driven by salience alone.
    pub sharpness: f32,
    /// Arrows tried per step that was looked at.
    pub arrows: usize,
    /// Best-looking moves that are checked against replies (0 = no reading).
    pub check: usize,
    /// Opponent steps / arrows per step considered when checking a move.
    pub reply_vision: usize,
    pub reply_arrows: usize,
    /// Softmax temperature of the final choice, in evaluation units.
    pub choice_temp: f32,
    /// The same once the regions are separated (counting squares, where
    /// weaker players waste a square or two).
    pub endgame_temp: f32,
    /// Probability of noticing a move that wins on the spot.
    pub spot_win: f32,
    /// Spread of the per-game style (log-scale multiplier on weights).
    pub style: f32,
    /// How intuitive the judgement is: 1 = `intuitive_weights`, 0 = the
    /// engine's weights.
    pub intuition: f32,
}

/// Levels 1–3 (1 = beginner). Levels 4 and 5 are plain MCTS.
pub fn human_level(level: u8) -> HumanLevel {
    match level {
        0 | 1 => HumanLevel {
            vision: 10,
            sharpness: 0.15,
            arrows: 6,
            check: 2,
            reply_vision: 3,
            reply_arrows: 3,
            choice_temp: 1.0,
            endgame_temp: 2.5,
            spot_win: 0.5,
            style: 0.4,
            intuition: 1.0,
        },
        2 => HumanLevel {
            vision: 12,
            sharpness: 0.3,
            arrows: 8,
            check: 4,
            reply_vision: 6,
            reply_arrows: 4,
            choice_temp: 1.0,
            endgame_temp: 1.5,
            spot_win: 0.9,
            style: 0.3,
            intuition: 0.7,
        },
        _ => HumanLevel {
            vision: 20,
            sharpness: 0.5,
            arrows: 14,
            check: 8,
            reply_vision: 14,
            reply_arrows: 8,
            choice_temp: 0.6,
            endgame_temp: 0.8,
            spot_win: 1.0,
            style: 0.2,
            intuition: 0.4,
        },
    }
}

/// Intuitive evaluation weights: queen reach and amazon freedom count more,
/// king-distance territory (what the engine relies on early) less.
pub fn intuitive_weights() -> EvalWeights {
    EvalWeights {
        tie: 0.2,
        phase_w: 40.0,
        t1_end: 1.0,
        t1_open: 0.8,
        t2: 0.3,
        c1: 0.0,
        c2: 0.3,
        mobility: 1.5,
    }
}

fn unit(rng: &mut Rng) -> f32 {
    ((rng.next_u64() >> 40) as f32 + 0.5) / (1u64 << 24) as f32
}

/// A per-game style: intuitive weights with mobility (aggression: blocking
/// and freedom), opening queen territory and king territory scaled by random
/// factors in `exp(±spread)`.
pub fn game_style(lvl: &HumanLevel, rng: &mut Rng) -> EvalWeights {
    let (h, e) = (intuitive_weights(), EvalWeights::default());
    let t = lvl.intuition;
    let mix = |a: f32, b: f32| t * a + (1.0 - t) * b;
    let mut w = EvalWeights {
        tie: mix(h.tie, e.tie),
        phase_w: mix(h.phase_w, e.phase_w),
        t1_end: mix(h.t1_end, e.t1_end),
        t1_open: mix(h.t1_open, e.t1_open),
        t2: mix(h.t2, e.t2),
        c1: mix(h.c1, e.c1),
        c2: mix(h.c2, e.c2),
        mobility: mix(h.mobility, e.mobility),
    };
    let f = |rng: &mut Rng| (lvl.style * (2.0 * unit(rng) - 1.0)).exp();
    w.mobility *= f(rng);
    w.t1_open *= f(rng);
    w.t2 *= f(rng);
    w.c2 *= f(rng);
    w
}

#[inline]
fn cheb(a: u8, b: u8) -> u8 {
    let (ar, ac) = (a / 10, a % 10);
    let (br, bc) = (b / 10, b % 10);
    ar.abs_diff(br).max(ac.abs_diff(bc))
}

/// Picks `k` items without replacement with probability proportional to
/// weight (Efraimidis–Spirakis keys). Returns indices.
fn sample(weights: &[f32], k: usize, rng: &mut Rng) -> Vec<usize> {
    let mut keys: Vec<(f32, usize)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (unit(rng).ln() / w.max(1e-9), i))
        .collect();
    keys.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    keys.truncate(k);
    keys.into_iter().map(|(_, i)| i).collect()
}

/// Score of the position after the step `from → to` (arrow not yet shot),
/// for the mover.
fn step_score(board: &Board, from: u8, to: u8, w: &EvalWeights) -> f32 {
    let mut b = *board;
    let me = b.side.index();
    b.amazons[me] ^= bit(from) | bit(to);
    b.side = b.side.opposite();
    -evaluate(&b, w)
}

const WIN: f32 = 1000.0;

/// Moves "seen" by a player at `board` with attention `focus` (squares that
/// draw the eye, e.g. the last move, or the amazons a reply would attack).
/// Returns `(score for the mover, move)`; a move that leaves the opponent
/// without moves scores `WIN`.
#[allow(clippy::too_many_arguments)]
fn noticed_moves(
    board: &Board,
    focus: &[u8],
    same_amazon: Option<u8>,
    vision: usize,
    arrows: usize,
    sharpness: f32,
    spot_win: f32,
    w: &EvalWeights,
    rng: &mut Rng,
) -> Vec<(f32, Move)> {
    let me = board.side.index();
    let occ = board.occupied();
    let opp = board.amazons[1 - me];
    let opp_nbrs = Bits(opp).fold(0, |acc, s| acc | neighbors(s)) & BOARD_MASK;

    // Amazon steps with salience.
    // (from, to, score, wins, salience)
    let mut steps: Vec<(u8, u8, f32, bool, f32)> = Vec::new();
    for from in Bits(board.amazons[me]) {
        let occ_wo = occ ^ bit(from);
        let room = queen_reach(from, occ).count_ones();
        for to in Bits(queen_reach(from, occ_wo)) {
            let libs = opp_nbrs & !(occ_wo | bit(to));
            let wins = libs == 0 || (libs.count_ones() == 1 && queen_reach(to, occ_wo) & libs != 0);
            let q = if wins {
                WIN
            } else {
                step_score(board, from, to, w)
            };
            let mut s = 1.0f32;
            if focus.iter().any(|&f| cheb(to, f) <= 2) {
                s *= 2.5;
            }
            if focus.iter().any(|&f| cheb(from, f) <= 2) {
                s *= 1.5;
            }
            if room <= 4 {
                s *= 3.0;
            }
            if opp_nbrs & bit(to) != 0 {
                s *= 1.8;
            }
            if same_amazon == Some(from) {
                s *= 1.4;
            }
            steps.push((from, to, q, wins, s));
        }
    }
    if steps.is_empty() {
        return Vec::new();
    }
    let qmax = steps
        .iter()
        .filter(|s| !s.3)
        .map(|s| s.2)
        .fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = steps
        .iter()
        .map(|s| {
            let qf = if s.3 || !qmax.is_finite() {
                1.0
            } else {
                (sharpness * (s.2 - qmax)).max(-30.0).exp()
            };
            s.4 * qf
        })
        .collect();
    let mut picked = sample(&weights, vision.min(steps.len()), rng);
    // An immediate win is noticed with probability `spot_win`.
    for (i, s) in steps.iter().enumerate() {
        if s.3 && !picked.contains(&i) && unit(rng) < spot_win {
            picked.push(i);
        }
    }

    let mut out = Vec::new();
    let mut b = *board;
    for i in picked {
        let (from, to, ..) = steps[i];
        let occ_wo = occ ^ bit(from);
        let mut cands: Vec<(f32, Move)> = Vec::new();
        for arrow in Bits(queen_reach(to, occ_wo)) {
            let m = Move::new(from, to, arrow);
            b.make_move(m);
            let s = if b.has_moves() { -evaluate(&b, w) } else { WIN };
            b.unmake_move(m);
            cands.push((s, m));
        }
        if cands.len() > arrows {
            // A winning arrow is as visible as the step that set it up.
            if let Some(win) = cands.iter().find(|c| c.0 >= WIN).copied() {
                out.push(win);
                continue;
            }
            let amax = cands.iter().map(|c| c.0).fold(f32::NEG_INFINITY, f32::max);
            let aw: Vec<f32> = cands
                .iter()
                .map(|&(s, m)| {
                    let mut sal = 1.0f32;
                    if opp_nbrs & bit(m.arrow) != 0 {
                        sal *= 3.0;
                    }
                    if cheb(m.arrow, to) <= 2 {
                        sal *= 1.5;
                    }
                    if focus.iter().any(|&f| cheb(m.arrow, f) <= 1) {
                        sal *= 1.5;
                    }
                    sal * (sharpness * (s - amax)).max(-30.0).exp()
                })
                .collect();
            for j in sample(&aw, arrows, rng) {
                out.push(cands[j]);
            }
        } else {
            out.extend(cands);
        }
    }
    out
}

/// The value of `m` for its mover after the opponent's best reply among the
/// replies that come to mind (those aimed at the mover's amazons).
fn checked_value(board: &Board, m: Move, lvl: &HumanLevel, w: &EvalWeights, rng: &mut Rng) -> f32 {
    let mut b = *board;
    b.make_move(m);
    if !b.has_moves() {
        return WIN;
    }
    // The opponent's attention: my amazons (where it would hurt me).
    let mine: Vec<u8> =
        Bits((board.amazons[board.side.index()] ^ bit(m.from)) | bit(m.to)).collect();
    let replies = noticed_moves(
        &b,
        &mine,
        None,
        lvl.reply_vision,
        lvl.reply_arrows,
        lvl.sharpness,
        1.0,
        w,
        rng,
    );
    // Replies are scored for the opponent; my value is minus the best one.
    match replies
        .iter()
        .map(|r| r.0)
        .fold(f32::NEG_INFINITY, f32::max)
    {
        x if x.is_finite() => -x,
        _ => -evaluate(&b, w),
    }
}

/// Chooses a move like a player of level `lvl` with per-game weights `w`.
/// `history` is the game so far (used for what draws attention).
pub fn human_move(
    board: &Board,
    history: &[Move],
    lvl: &HumanLevel,
    w: &EvalWeights,
    rng: &mut Rng,
) -> Option<Move> {
    let last = history.last();
    let focus: Vec<u8> = last.map(|m| vec![m.to, m.arrow]).unwrap_or_default();
    let same = history
        .len()
        .checked_sub(2)
        .map(|i| history[i].to)
        .filter(|&sq| board.amazons[board.side.index()] & bit(sq) != 0);
    let mut cands = noticed_moves(
        board,
        &focus,
        same,
        lvl.vision,
        lvl.arrows,
        lvl.sharpness,
        lvl.spot_win,
        w,
        rng,
    );
    if cands.is_empty() {
        // Nothing caught the eye (cannot happen while moves exist, but be safe).
        return board.legal_moves().first().copied();
    }
    if let Some(win) = cands.iter().find(|c| c.0 >= WIN) {
        return Some(win.1);
    }
    if lvl.check > 0 {
        cands.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        cands.truncate(lvl.check);
        for c in cands.iter_mut() {
            c.0 = checked_value(board, c.1, lvl, w, rng);
        }
    }
    let best = cands.iter().map(|c| c.0).fold(f32::NEG_INFINITY, f32::max);
    let temp = if regions_separated(board) {
        lvl.endgame_temp
    } else {
        lvl.choice_temp
    };
    let weights: Vec<f32> = cands
        .iter()
        .map(|c| ((c.0 - best) / temp).max(-30.0).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    let mut r = unit(rng) * total;
    for (c, wt) in cands.iter().zip(&weights) {
        r -= wt;
        if r <= 0.0 {
            return Some(c.1);
        }
    }
    Some(cands.last().unwrap().1)
}
