//! How the difficulty levels play: self-play games of one level, every move
//! graded by the explanation engine (full-strength search), summarized by
//! game phase, with a few sample mistakes.
//!
//! cargo run --release --example levels -- [levels=1,2,3,4] [games=4] [iterations=800]

use std::sync::Mutex;

use amazons_engine::board::{Board, Move};
use amazons_engine::eval::regions_separated;
use amazons_engine::explain::{analyze_move, Grade};
use amazons_engine::players::{level_spec, Player};
use amazons_engine::rng::Rng;
use amazons_engine::search::MctsConfig;

const GRADES: [Grade; 5] = [
    Grade::Best,
    Grade::Good,
    Grade::Inaccuracy,
    Grade::Mistake,
    Grade::Blunder,
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let levels: Vec<u8> = args
        .first()
        .map_or("1,2,3,4", |s| s.as_str())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    let games: u64 = args.get(1).map_or(4, |s| s.parse().unwrap());
    let iters: u32 = args.get(2).map_or(800, |s| s.parse().unwrap());

    for &lv in &levels {
        // [phase][grade] counts; phases: opening (< 20 plies), middle, separated.
        let counts = Mutex::new([[0u32; 5]; 3]);
        let samples = Mutex::new(Vec::<String>::new());
        let lengths = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for g in 0..games {
                let (counts, samples, lengths) = (&counts, &samples, &lengths);
                s.spawn(move || {
                    let mut rng = Rng::new(1000 + g);
                    let mut b = Board::new();
                    let mut hist: Vec<Move> = Vec::new();
                    // Two random opening plies so games differ.
                    for _ in 0..2 {
                        let m = amazons_engine::random_move(&b, &mut rng).unwrap();
                        b.make_move(m);
                        hist.push(m);
                    }
                    let mut p = [
                        Player::new(level_spec(lv), 7 + g),
                        Player::new(level_spec(lv), 99 + g),
                    ];
                    while let Some(m) = p[hist.len() % 2].choose_in_game(&b, &hist) {
                        let phase = if regions_separated(&b) {
                            2
                        } else if hist.len() < 20 {
                            0
                        } else {
                            1
                        };
                        let e = analyze_move(&b, m, MctsConfig::default(), iters);
                        let gi = GRADES.iter().position(|&x| x == e.grade).unwrap();
                        counts.lock().unwrap()[phase][gi] += 1;
                        if gi >= 3 && samples.lock().unwrap().len() < 12 {
                            let mut t = format!(
                                "  ply {:>2} {} [{}] {}",
                                hist.len() + 1,
                                m,
                                e.grade.label(),
                                e.summary
                            );
                            for pt in e.points.iter().chain(&e.reply_points).take(3) {
                                t += &format!(
                                    "\n      {} {}",
                                    if pt.positive { "+" } else { "-" },
                                    pt.text
                                );
                            }
                            if let Some(best) = e.best {
                                t += &format!("\n      추천 {best}");
                            }
                            samples.lock().unwrap().push(t);
                        }
                        b.make_move(m);
                        hist.push(m);
                    }
                    lengths.lock().unwrap().push(hist.len());
                });
            }
        });
        let c = counts.into_inner().unwrap();
        let l = lengths.into_inner().unwrap();
        println!(
            "== level {lv}: {games} games, avg {:.0} plies",
            l.iter().sum::<usize>() as f32 / l.len() as f32
        );
        for (pi, name) in ["opening", "middle", "separated"].iter().enumerate() {
            let tot: u32 = c[pi].iter().sum();
            if tot == 0 {
                continue;
            }
            let pct: Vec<String> = c[pi]
                .iter()
                .zip(GRADES)
                .map(|(&n, g)| format!("{} {:>4.1}%", g.label(), 100.0 * n as f32 / tot as f32))
                .collect();
            println!("  {name:<9} ({tot:>3} moves): {}", pct.join("  "));
        }
        for s in samples.into_inner().unwrap() {
            println!("{s}");
        }
    }
}
