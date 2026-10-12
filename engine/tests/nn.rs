use amazons_engine::board::{Bits, Board, Color, Move};
use amazons_engine::nn::{canonical, canonical_move, flip_sq, planes, Net, PLANES};
use amazons_engine::rng::Rng;

fn flip_bb(b: u128) -> u128 {
    Bits(b).fold(0, |acc, s| acc | 1u128 << flip_sq(s))
}

/// The same position seen by the other color: flipped vertically, colors
/// swapped, other side to move.
fn color_swapped(b: &Board) -> Board {
    Board {
        amazons: [flip_bb(b.amazons[1]), flip_bb(b.amazons[0])],
        arrows: flip_bb(b.arrows),
        side: b.side.opposite(),
    }
}

#[test]
fn planes_depend_only_on_the_side_to_move_view() {
    let mut rng = Rng::new(21);
    let mut b = Board::new();
    while let Some(m) = amazons_engine::random_move(&b, &mut rng) {
        assert_eq!(planes(&b), planes(&color_swapped(&b)));
        assert_eq!(canonical(&b).side, Color::White);
        // Canonical moves are legal in the canonical position.
        let c = canonical(&b);
        assert!(c.is_legal(canonical_move(&b, m)));
        b.make_move(m);
    }
}

#[test]
fn start_planes() {
    let p = planes(&Board::new());
    assert_eq!(p[0].count_ones(), 4); // my amazons
    assert_eq!(p[1].count_ones(), 4);
    assert_eq!(p[2], 0); // no arrows
                         // Symmetric start: my and the opponent's "closer" squares match in count.
    assert_eq!(p[11].count_ones(), p[12].count_ones());
    assert_eq!(PLANES, p.len());
}

fn weights(c: usize, blocks: usize, fill: impl Fn(usize) -> f32) -> Vec<u8> {
    let mut n = 9 * 13 * c + c;
    n += blocks * 2 * (9 * c * c + c);
    n += 2 * c + 2 + 64 * 200 + 64 + 64 + 1 + 3 * c + 3;
    let mut out = b"AMNN".to_vec();
    for v in [1u32, c as u32, blocks as u32] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for i in 0..n {
        out.extend_from_slice(&fill(i).to_le_bytes());
    }
    out
}

#[test]
fn weight_file_parsing_and_eval() {
    // All-zero weights: value tanh(0) = 0, all logits 0.
    let net = Net::from_bytes(&weights(4, 1, |_| 0.0)).unwrap();
    let o = net.eval(&Board::new());
    assert_eq!(o.value, 0.0);
    assert!(o
        .from
        .iter()
        .chain(&o.to)
        .chain(&o.arrow)
        .all(|&x| x == 0.0));
    // Truncated / trailing data and wrong magic are rejected.
    let w = weights(4, 1, |i| (i % 7) as f32 * 0.01);
    assert!(Net::from_bytes(&w[..w.len() - 4]).is_err());
    let mut longer = w.clone();
    longer.extend_from_slice(&[0; 4]);
    assert!(Net::from_bytes(&longer).is_err());
    assert!(Net::from_bytes(b"XXXX").is_err());
    // Non-trivial weights give finite outputs in range.
    let net = Net::from_bytes(&w).unwrap();
    let o = net.eval(&Board::new());
    assert!(o.value.is_finite() && o.value.abs() <= 1.0);
}

#[test]
fn mcts_with_network_plays_legal_moves() {
    use amazons_engine::nn::register_net;
    use amazons_engine::search::{Mcts, MctsConfig};
    let id = register_net(
        Net::from_bytes(&weights(4, 1, |i| ((i * 37) % 11) as f32 * 0.01 - 0.05)).unwrap(),
    );
    for mode in 0..3u8 {
        let cfg = MctsConfig {
            net: id,
            net_value: mode,
            ..MctsConfig::default()
        };
        let mut b = Board::new();
        for _ in 0..6 {
            let mut t = Mcts::new(b, cfg);
            t.run(50);
            let m: Move = t.best_move().unwrap();
            assert!(b.is_legal(m));
            b.make_move(m);
        }
    }
}
