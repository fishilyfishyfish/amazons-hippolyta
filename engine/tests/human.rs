use amazons_engine::board::{Board, Move};
use amazons_engine::human::{game_style, human_level, human_move};
use amazons_engine::players::{level_spec, Budget, Player, PlayerSpec, LEVEL4_ITERATIONS};
use amazons_engine::rng::Rng;

/// Plays `a` (White) against `b` and returns (white won, history).
fn play(a: PlayerSpec, b: PlayerSpec, seed: u64) -> (bool, Vec<Move>) {
    let mut board = Board::new();
    let mut hist = Vec::new();
    let mut p = [Player::new(a, seed), Player::new(b, seed + 1)];
    while let Some(m) = p[hist.len() % 2].choose_in_game(&board, &hist) {
        assert!(board.is_legal(m), "illegal {m} after {hist:?}");
        board.make_move(m);
        hist.push(m);
    }
    assert!(!board.has_moves(), "player gave up with moves left");
    (board.winner() == Some(amazons_engine::Color::White), hist)
}

#[test]
fn levels_play_legal_games_and_beat_random() {
    for lv in 1..=4u8 {
        for seed in 0..2 {
            let (w, _) = play(level_spec(lv), PlayerSpec::Random, 10 * lv as u64 + seed);
            assert!(w, "level {lv} lost to random as White");
            let (w, _) = play(PlayerSpec::Random, level_spec(lv), 20 * lv as u64 + seed);
            assert!(!w, "level {lv} lost to random as Black");
        }
    }
}

#[test]
fn level_specs() {
    assert_eq!(level_spec(2), PlayerSpec::Human(human_level(2)));
    assert!(matches!(
        level_spec(4),
        PlayerSpec::Mcts(_, Budget::Iterations(n), 1) if n == LEVEL4_ITERATIONS
    ));
    assert_eq!(PlayerSpec::parse("level:3").unwrap(), level_spec(3));
    match PlayerSpec::parse("human:2,hv=5,het=3").unwrap() {
        PlayerSpec::Human(h) => {
            assert_eq!(h.vision, 5);
            assert_eq!(h.endgame_temp, 3.0);
            assert_eq!(h.check, human_level(2).check);
        }
        s => panic!("{s:?}"),
    }
}

#[test]
fn style_depends_only_on_game_seed() {
    let l = human_level(1);
    assert_eq!(
        game_style(&l, &mut Rng::new(5)),
        game_style(&l, &mut Rng::new(5))
    );
    assert_ne!(
        game_style(&l, &mut Rng::new(5)),
        game_style(&l, &mut Rng::new(6))
    );
}

/// Positions from random games where some move wins on the spot.
fn winning_positions() -> Vec<(Board, Vec<Move>)> {
    let mut rng = Rng::new(3);
    let mut out = Vec::new();
    for _ in 0..40 {
        let mut b = Board::new();
        let mut hist = Vec::new();
        while let Some(m) = amazons_engine::random_move(&b, &mut rng) {
            let wins = b.legal_moves().into_iter().any(|x| {
                let mut c = b;
                c.make_move(x);
                !c.has_moves()
            });
            if wins {
                out.push((b, hist.clone()));
                break;
            }
            b.make_move(m);
            hist.push(m);
        }
    }
    out
}

#[test]
fn level3_always_sees_an_immediate_win() {
    let positions = winning_positions();
    assert!(positions.len() >= 10);
    let lvl = human_level(3);
    let mut rng = Rng::new(9);
    for (b, hist) in &positions {
        let w = game_style(&lvl, &mut rng);
        let m = human_move(b, hist, &lvl, &w, &mut rng).unwrap();
        let mut c = *b;
        c.make_move(m);
        assert!(!c.has_moves(), "missed the win in\n{b}");
    }
}
