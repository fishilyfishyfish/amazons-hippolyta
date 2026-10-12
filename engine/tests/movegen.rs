use amazons_engine::board::{parse_square, Board, Color, Move};
use amazons_engine::naive::NaiveBoard;
use amazons_engine::rng::Rng;

fn sq(s: &str) -> u8 {
    parse_square(s).unwrap()
}

/// Board with only the given pieces; white to move unless stated.
fn custom(white: &[&str], black: &[&str], arrows: &[&str], side: Color) -> (Board, NaiveBoard) {
    let mut b = Board {
        amazons: [0, 0],
        arrows: 0,
        side,
    };
    let mut n = NaiveBoard {
        cells: [[0; 10]; 10],
        side: side as u8 + 1,
    };
    for (list, code) in [(white, 1u8), (black, 2), (arrows, 3)] {
        for s in list {
            let q = sq(s);
            match code {
                1 => b.amazons[0] |= 1 << q,
                2 => b.amazons[1] |= 1 << q,
                _ => b.arrows |= 1 << q,
            }
            n.cells[q as usize / 10][q as usize % 10] = code;
        }
    }
    (b, n)
}

fn sorted(mut v: Vec<Move>) -> Vec<Move> {
    v.sort();
    v
}

#[test]
fn initial_position_has_2176_moves() {
    let b = Board::new();
    assert_eq!(b.legal_moves().len(), 2176);
    assert_eq!(b.count_moves(), 2176);
    assert_eq!(NaiveBoard::new().legal_moves().len(), 2176);
}

#[test]
fn initial_move_sets_match() {
    assert_eq!(
        sorted(Board::new().legal_moves()),
        sorted(NaiveBoard::new().legal_moves())
    );
}

#[test]
fn moves_are_unique() {
    let mut v = sorted(Board::new().legal_moves());
    let n = v.len();
    v.dedup();
    assert_eq!(v.len(), n);
}

#[test]
fn perft_depth2_matches_naive() {
    let fast = Board::new().perft(2);
    let slow = NaiveBoard::new().perft(2);
    assert_eq!(fast, slow);
    assert_eq!(fast, 4_307_152);
}

/// Depth 3 is ~8.4e9 leaves (about a minute for the naive side in release),
/// so it is opt-in: `cargo test --release -- --ignored`.
#[test]
#[ignore]
fn perft_depth3_matches_naive() {
    let fast = Board::new().perft(3);
    assert_eq!(fast, NaiveBoard::new().perft(3));
    assert_eq!(fast, 8_350_439_170);
}

fn cells_match(b: &Board, n: &NaiveBoard) {
    for s in 0..100 {
        assert_eq!(b.cell(s), n.cell(s), "cell {s}");
    }
    assert_eq!(b.side as u8 + 1, n.side);
}

/// Random games: at every ply the full move sets must agree, and every few
/// plies a depth-2 perft must agree. Covers mid- and endgame positions.
#[test]
fn random_games_cross_check() {
    let mut rng = Rng::new(0xA11A_2055);
    for game in 0..12 {
        let mut b = Board::new();
        let mut n = NaiveBoard::new();
        let mut ply = 0;
        loop {
            cells_match(&b, &n);
            let fast = sorted(b.legal_moves());
            let slow = sorted(n.legal_moves());
            assert_eq!(fast, slow, "game {game} ply {ply}\n{b}");
            assert_eq!(b.count_moves() as usize, fast.len());
            assert_eq!(
                b.has_moves(),
                !fast.is_empty(),
                "game {game} ply {ply}\n{b}"
            );
            if ply % 7 == 3 && fast.len() < 1500 {
                assert_eq!(
                    b.clone().perft(2),
                    n.clone().perft(2),
                    "game {game} ply {ply}"
                );
            }
            if fast.is_empty() {
                assert_eq!(b.winner(), Some(b.side.opposite()));
                break;
            }
            let m = fast[rng.below(fast.len())];
            assert!(b.is_legal(m));
            b.make_move(m);
            n.make_move(m);
            ply += 1;
        }
        assert!(ply >= 20, "game {game} ended too early: {ply}");
    }
}

#[test]
fn make_unmake_restores_position() {
    let mut rng = Rng::new(7);
    let mut b = Board::new();
    let mut stack = Vec::new();
    let mut snapshots = vec![b];
    while let Some(m) = amazons_engine::random_move(&b, &mut rng) {
        b.make_move(m);
        stack.push(m);
        snapshots.push(b);
    }
    while let Some(m) = stack.pop() {
        snapshots.pop();
        b.unmake_move(m);
        assert_eq!(&b, snapshots.last().unwrap());
    }
    assert_eq!(b, Board::new());
}

#[test]
fn arrow_may_land_on_vacated_square() {
    let b = Board::new();
    assert!(b.is_legal(Move::new(sq("d1"), sq("d2"), sq("d1"))));
}

#[test]
fn cannot_pass_through_pieces() {
    let b = Board::new();
    // d1 -> d10 is occupied; d1 cannot jump to a square beyond a blocker.
    assert!(!b.is_legal(Move::new(sq("d1"), sq("d10"), sq("d9"))));
    // a4 -> a8 would pass through the black amazon on a7.
    assert!(!b.is_legal(Move::new(sq("a4"), sq("a8"), sq("a9"))));
    // Arrow cannot pass through: from d2 the arrow may not cross d10 / reach beyond.
    let (b, _) = custom(&["a1"], &["j10"], &["a3"], Color::White);
    assert!(!b.is_legal(Move::new(sq("a1"), sq("a4"), sq("a5"))));
    assert!(b.is_legal(Move::new(sq("a1"), sq("a2"), sq("b2"))));
    assert!(!b.is_legal(Move::new(sq("a1"), sq("a2"), sq("a4"))));
    // Cannot move the opponent's amazon or a non-amazon square.
    assert!(!Board::new().is_legal(Move::new(sq("a7"), sq("a6"), sq("a5"))));
    assert!(!Board::new().is_legal(Move::new(sq("e5"), sq("e6"), sq("e7"))));
}

#[test]
fn edges_do_not_wrap() {
    // A lone amazon on j1 must not "wrap" to a2 etc.
    let (b, n) = custom(&["j1"], &["a10"], &[], Color::White);
    assert_eq!(sorted(b.legal_moves()), sorted(n.legal_moves()));
    for m in b.legal_moves() {
        assert_ne!(m.to, sq("a2"));
    }
}

#[test]
fn terminal_detection() {
    // White on a1 boxed in by arrows; black free.
    let (b, n) = custom(&["a1"], &["j10"], &["a2", "b1", "b2"], Color::White);
    assert!(n.legal_moves().is_empty());
    assert!(!b.has_moves());
    assert!(b.is_game_over());
    assert_eq!(b.winner(), Some(Color::Black));
    // One free neighbor is enough: move there, shoot back.
    let (b, _) = custom(&["a1"], &["j10"], &["a2", "b1"], Color::White);
    assert!(b.has_moves());
    assert_eq!(b.legal_moves().len(), b.count_moves() as usize);
    assert!(b.is_legal(Move::new(sq("a1"), sq("b2"), sq("a1"))));
    // Boxed in by an enemy amazon counts too.
    let (b, _) = custom(&["a1"], &["a2", "b1", "b2"], &[], Color::White);
    assert_eq!(b.winner(), Some(Color::Black));
}

#[test]
fn notation_roundtrip() {
    let m: Move = "d1-d7(g7)".parse().unwrap();
    assert_eq!(m, Move::new(sq("d1"), sq("d7"), sq("g7")));
    assert_eq!(m.to_string(), "d1-d7(g7)");
    let m: Move = "j10-a1(a10)".parse().unwrap();
    assert_eq!(m.to_string(), "j10-a1(a10)");
    for bad in [
        "",
        "d1d7(g7)",
        "d1-d7g7",
        "k1-d7(g7)",
        "d0-d7(g7)",
        "d11-d7(g7)",
        "d01-d7(g7)",
    ] {
        assert!(bad.parse::<Move>().is_err(), "{bad}");
    }
    for m in Board::new().legal_moves() {
        assert_eq!(m.to_string().parse::<Move>().unwrap(), m);
    }
}

#[test]
fn wasm_game_api() {
    use amazons_engine::wasm::Game;
    let mut g = Game::new(1.0);
    assert_eq!(g.legal_moves().len(), 2176 * 3);
    assert!(!g.play(sq("a7"), sq("a6"), sq("a5")));
    assert!(g.play_notation("d1-d7(g7)"));
    assert_eq!(g.side_to_move(), 1);
    assert_eq!(g.history_notation(), "d1-d7(g7)");
    assert_eq!(g.cells()[sq("g7") as usize], 3);
    let r = g.random_move();
    assert_eq!(r.len(), 3);
    assert!(g.play(r[0], r[1], r[2]));
    assert!(g.undo() && g.undo() && !g.undo());
    assert_eq!(g.cells(), Game::new(2.0).cells());
    assert_eq!(g.destinations(sq("d1")).len(), 20);
    assert!(g.destinations(sq("a7")).is_empty());
}
