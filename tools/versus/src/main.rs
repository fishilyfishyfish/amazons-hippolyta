//! Cross-version self-play. Specs use the current engine's syntax; a `v1:`
//! prefix selects the engine pinned in Cargo.toml (stage 4, commit 31df4b9).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use cur::board::{Board, Color, Move};
use cur::rng::Rng;

enum Engine {
    Cur(cur::players::Player),
    V1(v1::players::Player),
}

impl Engine {
    fn new(spec: &str, seed: u64) -> Engine {
        match spec.strip_prefix("v1:") {
            Some(s) => Engine::V1(v1::players::Player::new(
                v1::players::PlayerSpec::parse(s).unwrap(),
                seed,
            )),
            None => Engine::Cur(cur::players::Player::new(
                cur::players::PlayerSpec::parse(spec).unwrap(),
                seed,
            )),
        }
    }

    fn choose(&mut self, b: &Board) -> Option<Move> {
        match self {
            Engine::Cur(p) => p.choose(b),
            Engine::V1(p) => {
                let old = v1::board::Board {
                    amazons: b.amazons,
                    arrows: b.arrows,
                    side: if b.side == Color::White {
                        v1::board::Color::White
                    } else {
                        v1::board::Color::Black
                    },
                };
                p.choose(&old).map(|m| Move::new(m.from, m.to, m.arrow))
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (a, b) = (args[0].clone(), args[1].clone());
    let pairs: usize = args.get(2).map_or(20, |s| s.parse().unwrap());
    let open_plies: usize = args.get(3).map_or(2, |s| s.parse().unwrap());
    let seed: u64 = args.get(4).map_or(1, |s| s.parse().unwrap());
    // Validate specs up front.
    Engine::new(&a, 0);
    Engine::new(&b, 0);

    let mut rng = Rng::new(seed);
    let openings: Vec<Vec<Move>> = (0..pairs)
        .map(|_| {
            let mut bd = Board::new();
            (0..open_plies)
                .map(|_| {
                    let m = cur::random_move(&bd, &mut rng).unwrap();
                    bd.make_move(m);
                    m
                })
                .collect()
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
                let mut board = Board::new();
                for &m in &openings[g / 2] {
                    board.make_move(m);
                }
                let gs = seed * 1000 + g as u64;
                let mut pa = Engine::new(&a, gs);
                let mut pb = Engine::new(&b, gs ^ 0xFFFF);
                loop {
                    let a_to_move = (board.side == Color::White) == a_white;
                    let m = if a_to_move { pa.choose(&board) } else { pb.choose(&board) };
                    let Some(m) = m else { break };
                    assert!(board.is_legal(m), "illegal move {m}");
                    board.make_move(m);
                }
                let a_won = (board.winner() == Some(Color::White)) == a_white;
                results.lock().unwrap().push(a_won);
            });
        }
    });
    let res = results.into_inner().unwrap();
    let n = res.len() as f64;
    let wins = res.iter().filter(|&&w| w).count() as f64;
    let p = wins / n;
    let se = (p * (1.0 - p) / n).sqrt();
    let elo = |p: f64| -400.0 * (1.0 / p.clamp(1e-3, 1.0 - 1e-3) - 1.0).log10();
    println!("A = {a}\nB = {b}");
    println!(
        "games {n}, A wins {wins} ({:.1}% ± {:.1}), Elo(A-B) ≈ {:+.0} [{:+.0}, {:+.0}], {:.0}s",
        100.0 * p,
        196.0 * se,
        elo(p),
        elo(p - 1.96 * se),
        elo(p + 1.96 * se),
        start.elapsed().as_secs_f64()
    );
}
