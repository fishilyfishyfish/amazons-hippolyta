//! How often the move the search chose is among the top-k candidates of
//! (a) the network policy and (b) the static two-stage ordering.
//! cargo run --release --example nnrecall -- <net.bin> [positions=300]

use amazons_engine::board::Board;
use amazons_engine::nn::{move_logit, Net};
use amazons_engine::rng::Rng;
use amazons_engine::search::{Mcts, MctsConfig};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let net = Net::from_bytes(&std::fs::read(&args[0]).unwrap()).unwrap();
    let n: usize = args.get(1).map_or(300, |s| s.parse().unwrap());
    let mut rng = Rng::new(77);
    let (mut hit_nn, mut hit_static, mut hit_nn_step, mut total) = (0, 0, 0, 0);
    while total < n {
        // A position from a game between searches (plies 4..40).
        let mut b = Board::new();
        let plies = 4 + rng.below(36);
        for _ in 0..plies {
            if !b.has_moves() {
                break;
            }
            let mut t = Mcts::new(
                b,
                MctsConfig {
                    seed: rng.next_u64(),
                    jitter: 0.5,
                    ..MctsConfig::default()
                },
            );
            t.run(100);
            b.make_move(t.best_move().unwrap());
        }
        if !b.has_moves() {
            continue;
        }
        // Reference: a deep search's choice.
        let mut t = Mcts::new(
            b,
            MctsConfig {
                top_k: 80,
                half_k: 32,
                ..MctsConfig::default()
            },
        );
        t.run(4000);
        let best = t.best_move().unwrap();
        // Static ordering, as the engine expands nodes (top 40).
        let mut s = Mcts::new(b, MctsConfig::default());
        s.run(1);
        if s.candidates().iter().any(|c| c.mv == best) {
            hit_static += 1;
        }
        // Network ordering.
        let o = net.eval(&b);
        let mut scored: Vec<(f32, _)> = b
            .legal_moves()
            .into_iter()
            .map(|m| (move_logit(&o, &b, m), m))
            .collect();
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        if scored.iter().take(40).any(|(_, m)| *m == best) {
            hit_nn += 1;
        }
        if scored
            .iter()
            .take(40)
            .any(|(_, m)| m.from == best.from && m.to == best.to)
        {
            hit_nn_step += 1;
        }
        total += 1;
    }
    println!(
        "{total} positions: best move in top-40 — static {:.0}%, network {:.0}% (network, same amazon step: {:.0}%)",
        100.0 * hit_static as f64 / total as f64,
        100.0 * hit_nn as f64 / total as f64,
        100.0 * hit_nn_step as f64 / total as f64
    );
}
