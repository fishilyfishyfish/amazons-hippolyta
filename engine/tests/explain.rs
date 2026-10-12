use amazons_engine::board::{parse_square, Board, Color, Move};
use amazons_engine::explain::{describe, explain, move_eun, move_facts, ro, territory, Grade};
use amazons_engine::rng::Rng;

fn sq(s: &str) -> u8 {
    parse_square(s).unwrap()
}

fn mv(s: &str) -> Move {
    s.parse().unwrap()
}

fn custom(white: &[&str], black: &[&str], arrows: &[&str], side: Color) -> Board {
    let mut b = Board {
        amazons: [0, 0],
        arrows: 0,
        side,
    };
    for s in white {
        b.amazons[0] |= 1 << sq(s);
    }
    for s in black {
        b.amazons[1] |= 1 << sq(s);
    }
    for s in arrows {
        b.arrows |= 1 << sq(s);
    }
    b
}

#[test]
fn territory_partitions_empty_squares() {
    let mut rng = Rng::new(4);
    let mut b = Board::new();
    while let Some(m) = amazons_engine::random_move(&b, &mut rng) {
        let t = territory(&b);
        let parts = [t.owned[0], t.owned[1], t.contested, t.dead];
        let mut union = 0u128;
        for (i, p) in parts.iter().enumerate() {
            for q in &parts[i + 1..] {
                assert_eq!(p & q, 0, "parts overlap\n{b}");
            }
            union |= p;
        }
        assert_eq!(
            union,
            b.empty(),
            "parts do not cover the empty squares\n{b}"
        );
        assert_eq!(t.secure[0] & !t.owned[0], 0);
        assert_eq!(t.secure[1] & !t.owned[1], 0);
        b.make_move(m);
    }
}

#[test]
fn start_position_territory_is_symmetric() {
    let t = territory(&Board::new());
    assert_eq!(t.owned[0].count_ones(), t.owned[1].count_ones());
}

#[test]
fn trapping_an_amazon_is_explained() {
    // Black j10 can only leave through i10 / i9 / j9. White on a10 steps to
    // h10 and shoots j9 ... we construct it so that white seals j10 fully:
    // black j10 with neighbours i9 (arrow), j9 (arrow); white moves to h10?
    // Simpler: white amazon slides next to it and the arrow takes the last
    // free neighbour.
    let b = custom(&["a10", "a1"], &["j10", "j1"], &["i9", "j9"], Color::White);
    // a10-h10 leaves i10 as the only liberty; arrow h10->i10 seals it.
    let m = mv("a10-h10(i10)");
    assert!(b.is_legal(m));
    let f = move_facts(&b, m);
    let texts: Vec<String> = describe(&f).into_iter().map(|p| p.text).collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("j10 아마존을 완전히 가뒀어요")),
        "{texts:?}"
    );
}

#[test]
fn self_trap_is_flagged() {
    // White moves its amazon into a corner pocket and shoots the exit.
    // d4 slides down the diagonal to b2 and shoots c3 behind itself: b2 is
    // left with a1 only.
    let b = custom(
        &["d4", "j5"],
        &["h10", "a10"],
        &["a2", "a3", "b1", "b3", "c1", "c2"],
        Color::White,
    );
    let m = mv("d4-b2(c3)");
    assert!(b.is_legal(m), "{b}");
    let texts: Vec<String> = describe(&move_facts(&b, m))
        .into_iter()
        .map(|p| p.text)
        .collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("내 b2 아마존") && t.contains("갇")),
        "{texts:?}"
    );
}

#[test]
fn endgame_waste_is_detected() {
    // Regions sealed by a wall of arrows on rank 5. White owns ranks 1-4,
    // black ranks 6-10. A white arrow that cuts off a corner wastes squares.
    let wall = ["a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5", "i5", "j5"];
    let b = custom(&["a1"], &["j10"], &wall, Color::White);
    // a1-a2 shooting b1 keeps everything usable.
    let good = describe(&move_facts(&b, mv("a1-a2(a1)")));
    assert!(good.iter().all(|p| p.positive), "{good:?}");
    // Moving to c1 and shooting b2... build a pocket: arrow at a2 after
    // moving to b1 isolates a1? a1 stays reachable from b1. Use a real cut:
    // white at b2 shooting... the amazon goes d1, arrow a1? simplest
    // detectable waste: walling off a whole column region.
    let b2 = custom(
        &["d1"],
        &["j10"],
        &[&wall[..], &["b2", "b3", "b4", "a3"]].concat(),
        Color::White,
    );
    // Pocket a1, a2, b1, c1... d1-c2(b1): arrow b1 cuts a1/a2 off (a2 is
    // only adjacent to a1, b1, b2(x), b3(x), a3(x)).
    let m = mv("d1-c2(b1)");
    assert!(b2.is_legal(m), "{b2}");
    let pts = describe(&move_facts(&b2, m));
    assert!(
        pts.iter()
            .any(|p| !p.positive && p.text.contains("못 쓰게")),
        "{pts:?}"
    );
}

#[test]
fn grades_follow_loss() {
    assert_eq!(Grade::from_loss(0.3, true), Grade::Best);
    assert_eq!(Grade::from_loss(0.01, false), Grade::Best);
    assert_eq!(Grade::from_loss(0.04, false), Grade::Good);
    assert_eq!(Grade::from_loss(0.1, false), Grade::Inaccuracy);
    assert_eq!(Grade::from_loss(0.2, false), Grade::Mistake);
    assert_eq!(Grade::from_loss(0.4, false), Grade::Blunder);
}

#[test]
fn korean_particles() {
    assert_eq!(ro(sq("c10")), "으로");
    assert_eq!(ro(sq("c1")), "로");
    assert_eq!(ro(sq("c2")), "로");
    assert_eq!(ro(sq("c3")), "으로");
    assert_eq!(ro(sq("c6")), "으로");
    assert_eq!(ro(sq("c7")), "로");
    assert_eq!(move_eun(mv("a7-e3(e6)")), "은");
    assert_eq!(move_eun(mv("a7-e3(e5)")), "는");
    assert_eq!(move_eun(mv("a7-e3(e10)")), "은");
    assert_eq!(move_eun(mv("a7-e3(e2)")), "는");
}

#[test]
fn explanation_json_is_valid() {
    let b = Board::new();
    let played = mv("d1-d2(d1)");
    let best = mv("d1-d7(g7)");
    let e = explain(&b, played, Some(best), Some(mv("a7-a6(a5)")), 0.55, 0.30);
    assert_eq!(e.grade, Grade::Blunder);
    assert_eq!(e.best, Some(best));
    let v: serde_json::Value = serde_json::from_str(&e.to_json()).unwrap();
    assert_eq!(v["grade"], "blunder");
    assert_eq!(v["label"], "대실수");
    assert_eq!(v["best"]["move"], "d1-d7(g7)");
    assert!(v["points"].as_array().unwrap().len() >= 1);
    assert!(v["balance"].as_str().unwrap().contains("백"));
    // A best move hides the recommendation.
    let e = explain(&b, best, Some(best), None, 0.55, 0.55);
    assert_eq!(e.grade, Grade::Best);
    assert_eq!(e.best, None);
    assert!(!e.points.is_empty());
}

#[test]
fn wasm_explain_move() {
    use amazons_engine::wasm::explain_move;
    let h = [3u8, 63, 66]; // d1-d7(g7)
    let j = explain_move(&h, 0, &[3, 13, 3], &[], 0.6, 0.6);
    let v: serde_json::Value = serde_json::from_str(&j).unwrap();
    assert!(v["summary"].is_string());
    assert_eq!(explain_move(&h, 1, &[], &[], 0.5, 0.5), "null"); // no such ply
    assert_eq!(explain_move(&[3, 99, 66], 0, &[], &[], 0.5, 0.5), "null"); // illegal
                                                                           // Illegal "best" is ignored rather than trusted.
    let j = explain_move(&h, 0, &[0, 1, 2], &[], 0.6, 0.3);
    let v: serde_json::Value = serde_json::from_str(&j).unwrap();
    assert!(v["best"].is_null());
}

#[test]
fn winning_move_is_not_described_as_a_counting_fight() {
    let b = custom(&["a10", "a1"], &["j10"], &["i9", "j9"], Color::White);
    let m = mv("a10-h10(i10)");
    let pts = describe(&move_facts(&b, m));
    assert!(pts[0].text.contains("이겨요"), "{pts:?}");
    assert!(pts.iter().all(|p| !p.text.contains("칸 수")), "{pts:?}");
}
