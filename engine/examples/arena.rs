//! Self-play matches between two player specs.
//!
//! cargo run --release --example arena -- <A> <B> [pairs=20] [opening_plies=2] [seed=1]
//!
//! Each pair plays the same random opening twice with colors swapped. Prints
//! A's score, a 95% confidence interval and the implied Elo difference.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use amazons_engine::board::{Board, Color};
use amazons_engine::players::{Player, PlayerSpec};
use amazons_engine::rng::Rng;

fn play(
    a: PlayerSpec,
    b: PlayerSpec,
    a_white: bool,
    opening: &[amazons_engine::Move],
    seed: u64,
) -> (bool, usize) {
    let mut board = Board::new();
    for &m in opening {
        board.make_move(m);
    }
    let mut pa = Player::new(a, seed);
    let mut pb = Player::new(b, seed ^ 0xFFFF);
    let mut plies = opening.len();
    loop {
        let a_to_move = (board.side == Color::White) == a_white;
        let m = if a_to_move {
            pa.choose(&board)
        } else {
            pb.choose(&board)
        };
        let Some(m) = m else { break };
        assert!(board.is_legal(m));
        board.make_move(m);
        plies += 1;
    }
    let winner = board.winner().unwrap();
    ((winner == Color::White) == a_white, plies)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: arena <A> <B> [pairs] [opening_plies] [seed]");
        std::process::exit(2);
    }
    let a = PlayerSpec::parse(&args[0]).unwrap();
    let b = PlayerSpec::parse(&args[1]).unwrap();
    let pairs: usize = args.get(2).map_or(20, |s| s.parse().unwrap());
    let open_plies: usize = args.get(3).map_or(2, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(1, |s| s.parse().unwrap());

    // Openings: random plies, identical for both games of a pair.
    let mut rng = Rng::new(seed);
    let openings: Vec<Vec<_>> = (0..pairs)
        .map(|_| {
            let mut bd = Board::new();
            let mut v = Vec::new();
            for _ in 0..open_plies {
                let m = amazons_engine::random_move(&bd, &mut rng).unwrap();
                bd.make_move(m);
                v.push(m);
            }
            v
        })
        .collect();

    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::new());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let g = next.fetch_add(1, Ordering::Relaxed);
                if g >= 2 * pairs {
                    break;
                }
                let a_white = g % 2 == 0;
                let r = play(a, b, a_white, &openings[g / 2], seed * 1000 + g as u64);
                results.lock().unwrap().push((g, a_white, r.0, r.1));
            });
        }
    });
    let res = results.into_inner().unwrap();
    let n = res.len() as f64;
    let wins = res.iter().filter(|r| r.2).count() as f64;
    let white_wins = res.iter().filter(|r| r.1 == r.2).count();
    let avg_plies = res.iter().map(|r| r.3).sum::<usize>() as f64 / n;
    let p = wins / n;
    let se = (p * (1.0 - p) / n).sqrt();
    let elo = |p: f64| -400.0 * (1.0 / p.clamp(1e-3, 1.0 - 1e-3) - 1.0).log10();
    println!("A = {}\nB = {}", args[0], args[1]);
    println!(
        "games {n}, A wins {wins} ({:.1}% ± {:.1}), Elo(A-B) ≈ {:+.0} [{:+.0}, {:+.0}]",
        100.0 * p,
        196.0 * se,
        elo(p),
        elo(p - 1.96 * se),
        elo(p + 1.96 * se)
    );
    println!(
        "white won {white_wins}/{n}, avg game length {avg_plies:.1} plies, {:.1}s on {threads} threads",
        start.elapsed().as_secs_f64()
    );
}
