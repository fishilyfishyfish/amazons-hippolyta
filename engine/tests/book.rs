use amazons_engine::board::{Board, Move};
use amazons_engine::explain::book_notes;

fn mv(s: &str) -> Move {
    s.parse().unwrap()
}

#[test]
fn notes_have_both_sides() {
    let b = Board::new();
    let best = mv("d1-d9(g9)");
    let n = book_notes(&b, best, best, 0.0, Some(mv("a7-d4(f2)")));
    assert_eq!(n.summary, "엔진의 최선수");
    assert!(!n.pros.is_empty() && !n.cons.is_empty(), "{n:?}");
    // A worse move mentions the gap to the best move first among its cons.
    let n = book_notes(&b, mv("d1-d2(d1)"), best, 20.0, None);
    assert_eq!(n.summary, "덜 좋은 선택");
    assert!(
        n.cons[0].contains("최선수 d1-d9(g9)보다 승률이 20%p 낮아요"),
        "{n:?}"
    );
    // An arrow that shuts in my own amazons is a con.
    let n = book_notes(&b, mv("d1-d7(g4)"), best, 0.5, None);
    assert!(n.cons.iter().any(|c| c.contains("답답해져요")), "{n:?}");
}

/// The committed book must be well-formed and every stored move legal.
#[test]
fn committed_book_is_valid() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/book.json");
    let text = std::fs::read_to_string(path).expect("web/book.json missing");
    let book: serde_json::Value = serde_json::from_str(&text).unwrap();
    let nodes = book["nodes"].as_object().unwrap();
    assert!(nodes.contains_key(""), "no start position");
    assert_eq!(book["positions"].as_u64().unwrap() as usize, nodes.len());
    for (key, node) in nodes {
        let mut b = Board::new();
        for t in key.split_whitespace() {
            let m = mv(t);
            assert!(b.is_legal(m), "illegal key move {t} in {key:?}");
            b.make_move(m);
        }
        assert_eq!(
            node["ply"].as_u64().unwrap() as usize,
            key.split_whitespace().count()
        );
        let moves = node["moves"].as_array().unwrap();
        assert!(!moves.is_empty(), "{key:?}");
        for m in moves {
            let text = m["move"].as_str().unwrap();
            let parsed = mv(text);
            assert!(b.is_legal(parsed), "illegal book move {text} after {key:?}");
            let sq: Vec<u64> = m["sq"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect();
            assert_eq!(
                sq,
                vec![parsed.from as u64, parsed.to as u64, parsed.arrow as u64]
            );
            let win = m["win"].as_f64().unwrap();
            assert!((0.0..=100.0).contains(&win));
            assert!(!m["pros"].as_array().unwrap().is_empty());
            assert!(!m["cons"].as_array().unwrap().is_empty());
            // The expected continuation starts with the move and is legal.
            let mut lb = b;
            let line = m["line"].as_array().unwrap();
            assert_eq!(line[0].as_str().unwrap(), text);
            for l in line {
                let lm = mv(l.as_str().unwrap());
                assert!(
                    lb.is_legal(lm),
                    "illegal line move {lm} after {key:?} {text}"
                );
                lb.make_move(lm);
            }
        }
    }
}
