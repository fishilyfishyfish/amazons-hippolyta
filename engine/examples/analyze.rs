//! Prints the explanation of every move of a game.
//! cargo run --release --example analyze -- "d1-d7(g7) a7-e3(e6) ..." [iterations]
//! Without moves, plays a game between mcts:300 and greedy and analyzes it.

use amazons_engine::board::{Board, Move};
use amazons_engine::explain::analyze_move;
use amazons_engine::players::{Player, PlayerSpec};
use amazons_engine::search::MctsConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let iters: u32 = args.get(1).map_or(1500, |s| s.parse().unwrap());
    let moves: Vec<Move> = match args.first() {
        Some(text) if !text.is_empty() => text
            .split_whitespace()
            .map(|t| t.parse().unwrap())
            .collect(),
        _ => {
            let mut b = Board::new();
            let mut a = Player::new(PlayerSpec::parse("mcts:300").unwrap(), 1);
            let mut g = Player::new(PlayerSpec::parse("greedy").unwrap(), 2);
            let mut v = Vec::new();
            for ply in 0.. {
                let m = if ply % 2 == 0 {
                    a.choose(&b)
                } else {
                    g.choose(&b)
                };
                let Some(m) = m else { break };
                b.make_move(m);
                v.push(m);
            }
            v
        }
    };
    let mut b = Board::new();
    for (i, &m) in moves.iter().enumerate() {
        let e = analyze_move(&b, m, MctsConfig::default(), iters);
        println!(
            "{:>3}. {} {}  [{}] {}",
            i / 2 + 1,
            if i % 2 == 0 { "백" } else { "흑" },
            m,
            e.grade.label(),
            e.summary
        );
        for p in &e.points {
            println!("        {} {}", if p.positive { "+" } else { "-" }, p.text);
        }
        for p in &e.reply_points {
            println!("        ! {}", p.text);
        }
        if let Some(best) = e.best {
            println!("        추천: {best}");
            for p in &e.best_points {
                println!("          · {}", p.text);
            }
        }
        println!("        ({})", e.balance);
        b.make_move(m);
    }
}
