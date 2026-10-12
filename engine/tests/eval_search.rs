use amazons_engine::board::{parse_square, Bitboard, Bits, Board, Color, Move};
use amazons_engine::eval::{
    evaluate, evaluate_terms, king_levels, queen_levels, reference_distances, EvalWeights,
};
use amazons_engine::rng::Rng;
use amazons_engine::search::{greedy_move, Mcts, MctsConfig};

fn sq(s: &str) -> u8 {
    parse_square(s).unwrap()
}

fn random_positions(seed: u64, games: usize) -> Vec<Board> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    for _ in 0..games {
        let mut b = Board::new();
        while let Some(m) = amazons_engine::random_move(&b, &mut rng) {
            out.push(b);
            b.make_move(m);
        }
        out.push(b);
    }
    out
}

fn mirror_bb(b: Bitboard) -> Bitboard {
    Bits(b).fold(0, |acc, s| acc | 1u128 << ((9 - s / 10) * 10 + s % 10))
}

/// Flip the board vertically and swap colors: same position from the other
/// side's point of view.
fn mirror(b: &Board) -> Board {
    Board {
        amazons: [mirror_bb(b.amazons[1]), mirror_bb(b.amazons[0])],
        arrows: mirror_bb(b.arrows),
        side: b.side.opposite(),
    }
}

#[test]
fn distances_match_reference_bfs() {
    for b in random_positions(3, 6) {
        let occ = b.occupied();
        for c in 0..2 {
            let q = queen_levels(b.amazons[c], occ).to_array();
            let k = king_levels(b.amazons[c], occ).to_array();
            assert_eq!(
                q,
                reference_distances(&b, b.amazons[c], false),
                "queen\n{b}"
            );
            assert_eq!(k, reference_distances(&b, b.amazons[c], true), "king\n{b}");
        }
    }
}

#[test]
fn start_distances() {
    let b = Board::new();
    let q = queen_levels(b.amazons[0], b.occupied()).to_array();
    assert_eq!(q[sq("d2") as usize], 1);
    assert_eq!(q[sq("e2") as usize], 1); // d1 diagonal
    assert_eq!(q[sq("e5") as usize], 2); // on no white queen line
    assert_eq!(q[sq("a4") as usize], 255); // occupied
}

#[test]
fn evaluation_is_color_symmetric() {
    let w = EvalWeights::default();
    let start = Board::new();
    assert!((evaluate(&start, &w) - evaluate(&mirror(&start), &w)).abs() < 1e-4);
    for b in random_positions(11, 4) {
        let e1 = evaluate(&b, &w);
        let e2 = evaluate(&mirror(&b), &w);
        assert!((e1 - e2).abs() < 1e-3, "{e1} vs {e2}\n{b}");
    }
}

#[test]
fn territory_counts_separated_regions() {
    // White a1 owns a 3-square pocket; black j10 owns the rest. A wall of
    // arrows on rank 2 / file b seals the pocket {a1? occupied, b1? ...}.
    let mut b = Board {
        amazons: [0, 0],
        arrows: 0,
        side: Color::White,
    };
    b.amazons[0] |= 1 << sq("a1");
    b.amazons[1] |= 1 << sq("j10");
    for s in ["a3", "b3", "c3", "c2", "c1"] {
        b.arrows |= 1 << sq(s);
    }
    let t = evaluate_terms(&b, &EvalWeights::default());
    // White reaches a2, b2, b1 (3); black reaches the other 100-2-5-3 = 90.
    assert_eq!(t.t1, 3.0 - 90.0);
    assert!(t.score < -50.0);
}

#[test]
fn greedy_takes_immediate_win() {
    // Black j10 has one escape square i10; white can seal it and win.
    let mut b = Board {
        amazons: [0, 0],
        arrows: 0,
        side: Color::White,
    };
    b.amazons[0] |= 1 << sq("a1");
    b.amazons[1] |= 1 << sq("j10");
    for s in ["i9", "j9"] {
        b.arrows |= 1 << sq(s);
    }
    let m = greedy_move(&b, &EvalWeights::default()).unwrap();
    let mut after = b;
    after.make_move(m);
    assert!(!after.has_moves(), "greedy played {m}");
}

#[test]
fn mcts_takes_immediate_win_and_is_consistent() {
    let mut b = Board {
        amazons: [0, 0],
        arrows: 0,
        side: Color::White,
    };
    b.amazons[0] |= 1 << sq("a1");
    b.amazons[1] |= 1 << sq("j10");
    for s in ["i9", "j9"] {
        b.arrows |= 1 << sq(s);
    }
    let mut t = Mcts::new(b, MctsConfig::default());
    t.run(200);
    assert_eq!(t.iterations(), 200);
    let m = t.best_move().unwrap();
    let mut after = b;
    after.make_move(m);
    assert!(!after.has_moves(), "mcts played {m}");
    assert!(t.root_value() > 0.9);
    // Visits of root edges sum to iterations - 1 (the first one expands).
    let sum: u32 = t.candidates().iter().map(|c| c.visits).sum();
    assert_eq!(sum, 199);
}

#[test]
fn mcts_lost_position() {
    // White to move with no moves at all.
    let mut b = Board {
        amazons: [0, 0],
        arrows: 0,
        side: Color::White,
    };
    b.amazons[0] |= 1 << sq("a1");
    b.amazons[1] |= 1 << sq("j10");
    for s in ["a2", "b1", "b2"] {
        b.arrows |= 1 << sq(s);
    }
    let mut t = Mcts::new(b, MctsConfig::default());
    t.run(10);
    assert_eq!(t.best_move(), None);
}

#[test]
fn mcts_plays_legal_moves_through_a_game() {
    let mut b = Board::new();
    let cfg = MctsConfig::default();
    let mut rng = Rng::new(5);
    let mut plies = 0;
    loop {
        let m: Option<Move> = if b.side == Color::White {
            let mut t = Mcts::new(b, cfg);
            t.run(60);
            let pv = t.pv(5);
            assert!(pv.is_empty() || pv[0] == t.best_move().unwrap());
            t.best_move()
        } else {
            amazons_engine::random_move(&b, &mut rng)
        };
        let Some(m) = m else { break };
        assert!(b.is_legal(m));
        b.make_move(m);
        plies += 1;
    }
    // MCTS (white) should beat a random player.
    assert_eq!(b.winner(), Some(Color::White), "after {plies} plies");
}

#[test]
fn wasm_analysis_api() {
    use amazons_engine::wasm::Game;
    let mut g = Game::new(1.0);
    // Start position is symmetric and the display score has no tempo bonus.
    let e = g.evaluation();
    let score: f32 = e["{\"score\":".len()..e.find(',').unwrap()]
        .parse()
        .unwrap();
    assert!(score.abs() < 1e-3, "{e}");
    let d = g.queen_distances();
    assert_eq!(d.len(), 200);
    assert_eq!(d[sq("d2") as usize], 1);
    assert_eq!(d[100 + sq("d9") as usize], 1);
    assert_eq!(g.greedy_move().len(), 3);
    assert_eq!(g.search_run(10), 0); // no search started
    g.search_start();
    assert_eq!(g.search_run(50), 50);
    let info = g.search_info(3);
    assert!(info.contains("\"iterations\":50"), "{info}");
    let best = g.search_best();
    assert!(g.play(best[0], best[1], best[2]));
    // Stale search no longer runs after the position changed.
    assert_eq!(g.search_run(10), 0);
}
