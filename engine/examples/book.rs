//! Builds the opening book (web/book.json).
//!
//! cargo run --release --example book -- [depth=4] [width=4] [iterations=80000] [out=../web/book.json]
//!
//! Breadth-first from the start position: every position reached by the
//! first `depth - 1` plies of book moves gets a deep MCTS; its `width` most
//! visited moves are stored with win rates (from the same tree, so they are
//! comparable), the expected continuation and pros/cons. Positions of one
//! depth are searched in parallel.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use amazons_engine::board::{Board, Color, Move};
use amazons_engine::explain::book_notes;
use amazons_engine::search::{Mcts, MctsConfig};
use serde_json::{json, Value};

fn mirror_sq(sq: u8) -> u8 {
    sq / 10 * 10 + (9 - sq % 10)
}

/// Left-right mirror image of a move (files a <-> j).
fn mirror_move(m: Move) -> Move {
    Move::new(mirror_sq(m.from), mirror_sq(m.to), mirror_sq(m.arrow))
}

fn mirror_board(b: &Board) -> Board {
    let f = |bb: u128| {
        (0..100u8)
            .filter(|&s| bb >> s & 1 == 1)
            .fold(0u128, |acc, s| acc | 1 << mirror_sq(s))
    };
    Board {
        amazons: [f(b.amazons[0]), f(b.amazons[1])],
        arrows: f(b.arrows),
        side: b.side,
    }
}

/// One decimal, as f64 so the JSON stays short.
fn r1(x: f32) -> f64 {
    (x as f64 * 10.0).round() / 10.0
}

fn notation(moves: &[Move]) -> String {
    moves
        .iter()
        .map(|m| m.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Searches one position and returns (its JSON entry, chosen moves).
fn analyse(line: &[Move], width: usize, iterations: u32) -> Option<(Value, Vec<Move>)> {
    let mut board = Board::new();
    for &m in line {
        board.make_move(m);
    }
    if !board.has_moves() {
        return None;
    }
    let mut t = Mcts::new(board, MctsConfig::default());
    t.run(iterations);
    // In a left-right symmetric position a move and its mirror image are the
    // same move: keep one, remember the twin.
    let symmetric = mirror_board(&board) == board;
    let mut cands = Vec::new();
    for c in t.candidates().into_iter().filter(|c| c.visits > 0) {
        if cands.len() == width {
            break;
        }
        if symmetric
            && cands
                .iter()
                .any(|k: &amazons_engine::search::Candidate| k.mv == mirror_move(c.mv))
        {
            continue;
        }
        cands.push(c);
    }
    let best = cands.first()?;
    let total: u32 = t.candidates().iter().map(|c| c.visits).sum();
    let pct = |q: f32| (q + 1.0) * 50.0;
    let mut moves = Vec::new();
    let mut chosen = Vec::new();
    for c in &cands {
        let cont = t.line_after(c.mv, 7);
        let gap = pct(best.q) - pct(c.q);
        let notes = book_notes(&board, c.mv, best.mv, gap.max(0.0), cont.get(1).copied());
        let twin = (symmetric && mirror_move(c.mv) != c.mv).then(|| mirror_move(c.mv).to_string());
        moves.push(json!({
            "move": c.mv.to_string(),
            "sq": [c.mv.from, c.mv.to, c.mv.arrow],
            "mirror": twin,
            "win": r1(pct(c.q)),
            "share": r1(100.0 * c.visits as f32 / total.max(1) as f32),
            "line": cont.iter().map(|m| m.to_string()).collect::<Vec<_>>(),
            "summary": notes.summary,
            "pros": notes.pros,
            "cons": notes.cons,
        }));
        chosen.push(c.mv);
    }
    let entry = json!({
        "ply": line.len(),
        "side": if board.side == Color::White { "백" } else { "흑" },
        "win": r1(pct(best.q)),
        "moves": moves,
    });
    Some((entry, chosen))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let depth: usize = args.first().map_or(4, |s| s.parse().unwrap());
    let width: usize = args.get(1).map_or(4, |s| s.parse().unwrap());
    let iterations: u32 = args.get(2).map_or(80_000, |s| s.parse().unwrap());
    let out = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "../web/book.json".into());
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());

    let start = Instant::now();
    let mut nodes = serde_json::Map::new();
    let mut level: Vec<Vec<Move>> = vec![Vec::new()];
    for d in 0..depth {
        let next = AtomicUsize::new(0);
        let results = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(line) = level.get(i) else { break };
                    if let Some(r) = analyse(line, width, iterations) {
                        results.lock().unwrap().push((line.clone(), r));
                    }
                });
            }
        });
        let mut results = results.into_inner().unwrap();
        results.sort_by_key(|(l, _)| notation(l));
        let mut children = Vec::new();
        for (line, (entry, chosen)) in results {
            for m in chosen {
                let mut l = line.clone();
                l.push(m);
                children.push(l);
            }
            nodes.insert(notation(&line), entry);
        }
        eprintln!(
            "ply {d}: {} positions, {:.0}s elapsed",
            level.len(),
            start.elapsed().as_secs_f64()
        );
        level = children;
    }
    let book = json!({
        "version": 1,
        "depth": depth,
        "width": width,
        "iterations": iterations,
        "positions": nodes.len(),
        "nodes": Value::Object(nodes),
    });
    std::fs::write(&out, serde_json::to_string(&book).unwrap() + "\n").unwrap();
    eprintln!(
        "wrote {out}: {} positions in {:.0}s",
        book["positions"],
        start.elapsed().as_secs_f64()
    );
}
