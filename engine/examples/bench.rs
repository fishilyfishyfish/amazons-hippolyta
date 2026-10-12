//! Move generation benchmark.
//! `cargo run --release --example bench [-- depth3]`

use std::hint::black_box;
use std::time::Instant;

use amazons_engine::board::{Board, Move};
use amazons_engine::eval::{evaluate, EvalWeights};
use amazons_engine::naive::NaiveBoard;
use amazons_engine::rng::Rng;
use amazons_engine::search::{Mcts, MctsConfig};

fn rate(n: u64, secs: f64) -> String {
    format!("{:.1} M/s", n as f64 / secs / 1e6)
}

/// Positions sampled from random games, to measure generation away from the
/// opening too.
fn sample_positions() -> Vec<Board> {
    let mut rng = Rng::new(12345);
    let mut out = Vec::new();
    for _ in 0..50 {
        let mut b = Board::new();
        while let Some(m) = amazons_engine::random_move(&b, &mut rng) {
            out.push(b);
            b.make_move(m);
        }
    }
    out
}

fn main() {
    let depth3 = std::env::args().any(|a| a == "depth3");

    // 1. Full move list generation (into a reused Vec) at the start position.
    let b = Board::new();
    let mut v: Vec<Move> = Vec::with_capacity(2200);
    let iters = 20_000u64;
    let t = Instant::now();
    for _ in 0..iters {
        v.clear();
        black_box(&b).generate_moves(&mut v);
        black_box(&v);
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "generate_moves(start): {:.2} us/call, {} moves",
        s / iters as f64 * 1e6,
        rate(iters * 2176, s)
    );

    // 2. Same over random-game positions.
    let pos = sample_positions();
    let mut total = 0u64;
    let t = Instant::now();
    for _ in 0..20 {
        for p in &pos {
            v.clear();
            black_box(p).generate_moves(&mut v);
            total += v.len() as u64;
        }
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "generate_moves(random games, {} positions): {:.2} us/call, {} moves",
        pos.len(),
        s / (20 * pos.len()) as f64 * 1e6,
        rate(total, s)
    );

    // 3. Perft 2, bitboard vs naive.
    let t = Instant::now();
    let n = Board::new().perft(2);
    let s = t.elapsed().as_secs_f64();
    println!(
        "perft(2) bitboard = {n} in {:.3}s ({} leaves)",
        s,
        rate(n, s)
    );
    let t = Instant::now();
    let n2 = NaiveBoard::new().perft(2);
    let s = t.elapsed().as_secs_f64();
    println!(
        "perft(2) naive    = {n2} in {:.3}s ({} leaves)",
        s,
        rate(n2, s)
    );
    assert_eq!(n, n2);

    // 4. Static evaluation.
    let w = EvalWeights::default();
    let t = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..20 {
        for p in &pos {
            acc += evaluate(black_box(p), &w);
        }
    }
    let s = t.elapsed().as_secs_f64();
    black_box(acc);
    println!(
        "evaluate (random-game positions): {:.2} us/call",
        s / (20 * pos.len()) as f64 * 1e6
    );

    // 5. MCTS iterations at the start and in a midgame position.
    for (label, b) in [("start", Board::new()), ("ply 20", pos[20])] {
        let mut m = Mcts::new(b, MctsConfig::default());
        let t = Instant::now();
        m.run(2000);
        let s = t.elapsed().as_secs_f64();
        println!(
            "mcts {label}: 2000 iterations in {:.2}s ({:.0} it/s, {} nodes)",
            s,
            2000.0 / s,
            m.node_count()
        );
    }

    // 6. Perft 3 (bulk counted), optional.
    if depth3 {
        let t = Instant::now();
        let n = Board::new().perft(3);
        let s = t.elapsed().as_secs_f64();
        println!(
            "perft(3) bitboard = {n} in {:.2}s ({} leaves)",
            s,
            rate(n, s)
        );
    }
}
