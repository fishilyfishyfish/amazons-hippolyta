//! Self-play training data for the neural network.
//!
//! cargo run --release --example selfplay -- <games> <iterations> <out.bin> [seed=1]
//!
//! Plays games with the current MCTS. The first 8 plies are sampled in
//! proportion to the visit counts (variety), later plies play the best move.
//! Every position is appended to `out.bin` as a fixed-size record
//! (`RECORD` bytes, little endian), in canonical (side-to-move) coordinates:
//!   13 × u128   input planes (nn::planes)
//!   u8          number of candidates n (≤ 40)
//!   40 × (u8 from, u8 to, u8 arrow, u16 visits)   root visit counts
//!   f32         search value for the side to move, in [-1, 1]
//!   i8          game result for the side to move (+1 win / -1 loss)

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use amazons_engine::board::Board;
use amazons_engine::nn::{canonical_move, planes, PLANES};
use amazons_engine::rng::Rng;
use amazons_engine::search::{Mcts, MctsConfig};

const MAX_CANDS: usize = 40;
pub const RECORD: usize = PLANES * 16 + 1 + MAX_CANDS * 5 + 4 + 1;

struct Sample {
    planes: [u128; PLANES],
    cands: Vec<(u8, u8, u8, u16)>,
    q: f32,
    white_to_move: bool,
}

fn play_game(iterations: u32, rng: &mut Rng) -> (Vec<Sample>, bool) {
    let mut board = Board::new();
    let mut samples = Vec::new();
    let mut ply = 0;
    loop {
        if !board.has_moves() {
            break;
        }
        let cfg = MctsConfig {
            seed: rng.next_u64(),
            ..MctsConfig::default()
        };
        let mut t = Mcts::new(board, cfg);
        t.run(iterations);
        let cands = t.candidates();
        let total: u32 = cands.iter().map(|c| c.visits).sum();
        let mv = if ply < 8 && total > 0 && t.root_proven() == 0 {
            let mut r = rng.below(total as usize) as u32;
            let mut pick = cands[0].mv;
            for c in &cands {
                if r < c.visits {
                    pick = c.mv;
                    break;
                }
                r -= c.visits;
            }
            pick
        } else {
            t.best_move().unwrap()
        };
        let scale = (u16::MAX as f32
            / cands.iter().map(|c| c.visits).max().unwrap_or(1).max(1) as f32)
            .min(1.0);
        samples.push(Sample {
            planes: planes(&board),
            cands: cands
                .iter()
                .filter(|c| c.visits > 0)
                .take(MAX_CANDS)
                .map(|c| {
                    let m = canonical_move(&board, c.mv);
                    (
                        m.from,
                        m.to,
                        m.arrow,
                        (c.visits as f32 * scale).round() as u16,
                    )
                })
                .collect(),
            q: t.root_value(),
            white_to_move: board.side == amazons_engine::board::Color::White,
        });
        board.make_move(mv);
        ply += 1;
    }
    // The side to move at the end has lost.
    let white_won = board.side == amazons_engine::board::Color::Black;
    (samples, white_won)
}

fn encode(s: &Sample, white_won: bool, out: &mut Vec<u8>) {
    for p in &s.planes {
        out.extend_from_slice(&p.to_le_bytes());
    }
    out.push(s.cands.len() as u8);
    for i in 0..MAX_CANDS {
        match s.cands.get(i) {
            Some(&(f, t, a, v)) => {
                out.extend_from_slice(&[f, t, a]);
                out.extend_from_slice(&v.to_le_bytes());
            }
            None => out.extend_from_slice(&[0; 5]),
        }
    }
    out.extend_from_slice(&s.q.to_le_bytes());
    out.push(if s.white_to_move == white_won {
        1
    } else {
        (-1i8) as u8
    });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let games: usize = args[0].parse().unwrap();
    let iterations: u32 = args[1].parse().unwrap();
    let path = args[2].clone();
    let seed: u64 = args.get(3).map_or(1, |s| s.parse().unwrap());
    let file = Mutex::new(
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap(),
    );
    let next = AtomicUsize::new(0);
    let positions = AtomicUsize::new(0);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                if g >= games {
                    break;
                }
                let mut rng = Rng::new(seed.wrapping_mul(1_000_003).wrapping_add(g as u64));
                let (samples, white_won) = play_game(iterations, &mut rng);
                let mut buf = Vec::with_capacity(samples.len() * RECORD);
                for smp in &samples {
                    encode(smp, white_won, &mut buf);
                }
                assert_eq!(buf.len(), samples.len() * RECORD);
                file.lock().unwrap().write_all(&buf).unwrap();
                let n = positions.fetch_add(samples.len(), Ordering::Relaxed) + samples.len();
                if g % 10 == 9 {
                    eprintln!(
                        "game {} done, {n} positions, {:.0}s",
                        g + 1,
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    eprintln!(
        "{games} games, {} positions in {:.0}s -> {path}",
        positions.load(Ordering::Relaxed),
        start.elapsed().as_secs_f64()
    );
}
