//! Human-readable move explanations.
//!
//! A move is compared in three snapshots: before it, after the amazon step
//! (arrow not yet shot) and after the arrow. From these we extract facts a
//! player thinks in — territory won or lost, enemy amazons hemmed in, own
//! amazons put in danger, what the arrow blocked or cost, regions sealed off —
//! and turn the most important ones into Korean sentences. The search
//! decides *how good* a move is (grade from win-probability loss); the facts
//! say *why*.

use crate::board::{queen_reach, square_name, Bitboard, Bits, Board, Color, Move};
use crate::eval::{evaluate_terms, queen_levels, EvalWeights};
use crate::search::{Mcts, MctsConfig};

/// Queen-distance ownership of empty squares.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Territory {
    /// Squares each color reaches strictly first (`[white, black]`).
    pub owned: [Bitboard; 2],
    /// Subset of `owned` that is safe: the other side is at least two queen
    /// moves further away or cannot reach it at all.
    pub secure: [Bitboard; 2],
    /// Squares both reach equally fast.
    pub contested: Bitboard,
    /// Squares nobody can reach.
    pub dead: Bitboard,
}

pub fn territory(board: &Board) -> Territory {
    let occ = board.occupied();
    let lv = [
        queen_levels(board.amazons[0], occ),
        queen_levels(board.amazons[1], occ),
    ];
    let mut t = Territory::default();
    let mut cum = [0u128; 2];
    let depth = lv[0].depth.max(lv[1].depth);
    // `near[c]`: squares color c reaches within distance d (for "secure").
    for d in 1..=depth {
        let l = [lv[0].levels[d], lv[1].levels[d]];
        cum[0] |= l[0];
        cum[1] |= l[1];
        for c in 0..2 {
            let mine = l[c] & !cum[1 - c];
            t.owned[c] |= mine;
            // Opponent not within d + 1.
            let opp_next = if d < depth {
                lv[1 - c].levels[d + 1]
            } else {
                0
            };
            t.secure[c] |= mine & !opp_next;
        }
        t.contested |= l[0] & l[1];
    }
    t.dead = board.empty() & !(lv[0].reached | lv[1].reached);
    t
}

pub use crate::eval::regions_separated;

/// Squares reachable at all (any number of moves) from one amazon.
fn region_size(sq: u8, occ: Bitboard) -> u32 {
    queen_levels(1u128 << sq, occ).reached.count_ones()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmazonState {
    pub sq: u8,
    /// Squares reachable in one queen move.
    pub mobility: u32,
    /// Squares reachable in any number of moves.
    pub region: u32,
}

fn amazons(board: &Board, color: usize) -> Vec<AmazonState> {
    let occ = board.occupied();
    Bits(board.amazons[color])
        .map(|sq| AmazonState {
            sq,
            mobility: queen_reach(sq, occ).count_ones(),
            region: region_size(sq, occ),
        })
        .collect()
}

/// Facts about one move, from the mover's point of view.
#[derive(Clone, Debug, PartialEq)]
pub struct MoveFacts {
    pub mv: Move,
    pub mover: Color,
    /// Owned squares (strictly closer) before / after, mine and the opponent's.
    pub mine_before: u32,
    pub mine_after: u32,
    pub opp_before: u32,
    pub opp_after: u32,
    pub contested_before: u32,
    pub contested_after: u32,
    pub secure_mine_after: u32,
    pub secure_opp_after: u32,
    /// Territory change caused by the arrow alone (after step vs after arrow).
    pub arrow_took_from_opp: i32,
    pub arrow_cost_me: i32,
    /// Opponent amazons before -> after (same squares; they did not move).
    pub opp_amazons: Vec<(AmazonState, AmazonState)>,
    /// My amazons before -> after (the moved one is listed from -> to).
    pub my_amazons: Vec<(AmazonState, AmazonState)>,
    pub separated_after: bool,
    pub separated_before: bool,
    pub opponent_has_no_moves: bool,
    /// Squares whose owner changed (for highlighting): now mine / now the
    /// opponent's / now contested or dead.
    pub gained: Bitboard,
    pub lost: Bitboard,
}

pub fn move_facts(board: &Board, m: Move) -> MoveFacts {
    let me = board.side.index();
    let opp = 1 - me;
    let before = territory(board);

    let mut step = *board;
    step.amazons[me] ^= (1u128 << m.from) | (1u128 << m.to);
    let mid = territory(&step);

    let mut after_b = *board;
    after_b.make_move(m);
    let after = territory(&after_b);

    let my_before = amazons(board, me);
    let my_after = amazons(&after_b, me);
    let my_amazons = my_before
        .iter()
        .map(|b| {
            let sq = if b.sq == m.from { m.to } else { b.sq };
            (*b, *my_after.iter().find(|a| a.sq == sq).unwrap())
        })
        .collect();
    let opp_before = amazons(board, opp);
    let opp_after = amazons(&after_b, opp);
    let opp_amazons = opp_before
        .iter()
        .copied()
        .zip(opp_after.iter().copied())
        .collect();

    let cnt = |b: Bitboard| b.count_ones();
    MoveFacts {
        mv: m,
        mover: board.side,
        mine_before: cnt(before.owned[me]),
        mine_after: cnt(after.owned[me]),
        opp_before: cnt(before.owned[opp]),
        opp_after: cnt(after.owned[opp]),
        contested_before: cnt(before.contested),
        contested_after: cnt(after.contested),
        secure_mine_after: cnt(after.secure[me]),
        secure_opp_after: cnt(after.secure[opp]),
        arrow_took_from_opp: cnt(mid.owned[opp]) as i32 - cnt(after.owned[opp]) as i32,
        arrow_cost_me: cnt(mid.owned[me]) as i32 - cnt(after.owned[me]) as i32,
        opp_amazons,
        my_amazons,
        separated_before: regions_separated(board),
        separated_after: regions_separated(&after_b),
        opponent_has_no_moves: !after_b.has_moves(),
        gained: after.owned[me] & !before.owned[me],
        lost: (before.owned[me] & !after.owned[me]) | (after.owned[opp] & !before.owned[opp]),
    }
}

/// Territory swing of a move for its mover (mine gained + theirs lost).
pub fn net_territory(f: &MoveFacts) -> i32 {
    (f.mine_after as i32 - f.mine_before as i32) - (f.opp_after as i32 - f.opp_before as i32)
}

/// What the opponent's best reply would do to us, as sentences from our
/// point of view. `f` are the reply's facts (its mover is the opponent).
pub fn describe_threat(f: &MoveFacts) -> Vec<Point> {
    let mut out = Vec::new();
    let reply = f.mv;
    let lead = format!("상대가 {reply}{} 응수하면", move_ro(reply));
    if f.opponent_has_no_moves {
        out.push(Point {
            weight: 100,
            positive: false,
            text: format!("{lead} 내가 둘 곳이 없어져서 져요."),
        });
        return out;
    }
    if !f.separated_before {
        // The reply's "opponent amazons" are ours.
        for (b, a) in &f.opp_amazons {
            let name = square_name(a.sq);
            let text = if a.mobility == 0 && b.mobility > 0 {
                Some((30, format!("{lead} 내 {name} 아마존이 완전히 갇혀요.")))
            } else if a.region <= 6 && b.region > a.region && b.region >= 10 {
                Some((
                    22,
                    format!(
                        "{lead} 내 {name} 아마존이 {}칸짜리 구역에 갇혀요.",
                        a.region
                    ),
                ))
            } else if a.mobility <= 3 && b.mobility >= a.mobility + 3 {
                Some((
                    16,
                    format!(
                        "{lead} 내 {name} 아마존의 길이 막혀요 ({} → {}칸).",
                        b.mobility, a.mobility
                    ),
                ))
            } else if b.mobility >= a.mobility + 6 {
                Some((
                    9,
                    format!(
                        "{lead} 내 {name} 아마존의 활동 범위가 크게 줄어요 ({} → {}칸).",
                        b.mobility, a.mobility
                    ),
                ))
            } else {
                None
            };
            if let Some((w, t)) = text {
                out.push(Point {
                    weight: w,
                    positive: false,
                    text: t,
                });
            }
        }
    }
    let net = net_territory(f);
    if net >= 8 {
        out.push(Point {
            weight: net.min(20),
            positive: false,
            text: format!("{lead} 상대가 영역을 {net}칸가량 크게 가져가요."),
        });
    }
    if f.separated_after && !f.separated_before {
        out.push(Point {
            weight: 25,
            positive: false,
            text: format!(
                "{lead} 영역이 갈리면서 칸 수 싸움이 돼요: 상대 {}칸, 내 쪽 {}칸.",
                f.mine_after, f.opp_after
            ),
        });
    }
    out.sort_by(|a, b| b.weight.cmp(&a.weight));
    out
}

/// One explanation sentence with its weight; `positive` = good for the mover.
#[derive(Clone, Debug, PartialEq)]
pub struct Point {
    pub weight: i32,
    pub positive: bool,
    pub text: String,
}

/// "로" or "으로" after a square name, by how its last digit is read
/// (1 일, 7 칠, 8 팔 end in ㄹ and take 로; 0 as in 10 십, 3 삼, 6 육 take 으로).
pub fn ro(sq: u8) -> &'static str {
    match square_name(sq).as_bytes().last() {
        Some(b'0' | b'3' | b'6') => "으로",
        _ => "로",
    }
}

/// Particle for a move in notation, which is read ending with its arrow
/// square: "d1-d7(g7)로", "d1-d7(g10)으로".
pub fn move_ro(m: Move) -> &'static str {
    ro(m.arrow)
}

/// "은" or "는" after a move in notation (by the arrow square's last digit:
/// 0 십, 1 일, 3 삼, 6 육, 7 칠, 8 팔 end in a consonant).
pub fn move_eun(m: Move) -> &'static str {
    match square_name(m.arrow).as_bytes().last() {
        Some(b'0' | b'1' | b'3' | b'6' | b'7' | b'8') => "은",
        _ => "는",
    }
}

fn color_name(c: Color) -> &'static str {
    match c {
        Color::White => "백",
        Color::Black => "흑",
    }
}

/// Facts turned into sentences, most important first.
pub fn describe(f: &MoveFacts) -> Vec<Point> {
    let mut out = Vec::new();
    let mut push = |weight: i32, positive: bool, text: String| {
        out.push(Point {
            weight,
            positive,
            text,
        });
    };

    if f.opponent_has_no_moves {
        push(
            100,
            true,
            "상대가 더 이상 둘 곳이 없어 이 수로 이겨요.".into(),
        );
    }

    // Once the regions are sealed, mobility details no longer matter: only
    // the square count does.
    let fight = !f.separated_before;

    // Enemy amazons hemmed in.
    for (b, a) in f.opp_amazons.iter().filter(|_| fight) {
        let name = square_name(a.sq);
        if a.mobility == 0 && b.mobility > 0 {
            push(
                30,
                true,
                format!("상대 {name} 아마존을 완전히 가뒀어요. 이제 그 아마존은 움직일 수 없어요."),
            );
        } else if a.region <= 6 && b.region > a.region && b.region >= 10 {
            push(
                22,
                true,
                format!(
                    "상대 {name} 아마존을 {}칸짜리 좁은 구역에 가뒀어요.",
                    a.region
                ),
            );
        } else if a.mobility <= 3 && b.mobility >= a.mobility + 3 {
            push(
                16,
                true,
                format!(
                    "상대 {name} 아마존의 길을 막았어요 (한 번에 갈 수 있는 칸 {} → {}).",
                    b.mobility, a.mobility
                ),
            );
        } else if b.mobility >= a.mobility + 6 {
            push(
                8,
                true,
                format!(
                    "상대 {name} 아마존의 활동 범위를 줄였어요 ({} → {}칸).",
                    b.mobility, a.mobility
                ),
            );
        }
    }

    // Own amazons: danger or relief.
    for (b, a) in f.my_amazons.iter().filter(|_| fight) {
        let moved = b.sq != a.sq;
        let name = square_name(a.sq);
        if a.mobility == 0 && !f.opponent_has_no_moves {
            push(
                28,
                false,
                format!("내 {name} 아마존이 스스로 갇혀서 더 이상 움직일 수 없어요."),
            );
        } else if a.region <= 6 && b.region >= 10 && !f.separated_after {
            push(
                20,
                false,
                format!(
                    "내 {name} 아마존이 {}칸짜리 좁은 구역에 갇혔어요.",
                    a.region
                ),
            );
        } else if a.mobility <= 3 && b.mobility >= a.mobility + 3 {
            push(
                14,
                false,
                format!(
                    "내 {name} 아마존이 위험해졌어요 (한 번에 갈 수 있는 칸 {} → {}). 상대가 가두기 쉬워요.",
                    b.mobility, a.mobility
                ),
            );
        } else if moved && b.mobility <= 4 && a.mobility >= b.mobility + 6 {
            push(
                12,
                true,
                format!(
                    "답답하던 아마존을 {}에서 {}{} 꺼내 활동 범위를 넓혔어요 ({} → {}칸).",
                    square_name(b.sq),
                    name,
                    ro(a.sq),
                    b.mobility,
                    a.mobility
                ),
            );
        }
    }

    // Territory balance. Almost every move gains a little (it is the mover's
    // turn), so only large swings are worth a sentence.
    let net = net_territory(f);
    let mine = f.mine_after as i32 - f.mine_before as i32;
    let theirs = f.opp_after as i32 - f.opp_before as i32;
    if net >= 8 {
        push(
            net.min(20),
            true,
            format!("영역을 크게 넓혔어요 (먼저 닿는 칸: 내 땅 {mine:+}칸, 상대 땅 {theirs:+}칸)."),
        );
    } else if net <= -2 {
        push(
            (-net).min(15) + 4,
            false,
            format!("오히려 영역을 잃었어요 (내 땅 {mine:+}칸, 상대 땅 {theirs:+}칸)."),
        );
    }

    // What the arrow did.
    if f.arrow_took_from_opp >= 3 {
        push(
            f.arrow_took_from_opp.min(12) + 1,
            true,
            format!(
                "{} 화살이 상대 쪽 길을 막아 상대 땅을 {}칸 줄였어요.",
                square_name(f.mv.arrow),
                f.arrow_took_from_opp
            ),
        );
    }
    if f.arrow_cost_me >= 3 {
        push(
            f.arrow_cost_me.min(12) + 1,
            false,
            format!(
                "{} 화살이 오히려 내 땅을 {}칸 깎았어요. 화살을 상대 쪽에 쏘는 게 나았어요.",
                square_name(f.mv.arrow),
                f.arrow_cost_me
            ),
        );
    }

    // Regions sealed.
    if f.separated_after && !f.separated_before && !f.opponent_has_no_moves {
        let me = f.mine_after;
        let opp = f.opp_after;
        let verdict = if me > opp {
            "남은 칸이 더 많아서 유리해요"
        } else if me < opp {
            "남은 칸이 더 적어서 불리해요"
        } else {
            "칸 수가 같아서 마지막까지 정확히 채워야 해요"
        };
        push(
            25,
            me >= opp,
            format!("이 수로 영역이 완전히 갈렸어요. 이제 각자 자기 구역을 채우는 싸움이에요: 내 쪽 {me}칸, 상대 쪽 {opp}칸 — {verdict}."),
        );
    }

    // Sealed endgame: every move uses up one square of your own region; a
    // move that makes more of it unusable wastes moves.
    if f.separated_before {
        let waste = f.mine_before as i32 - 1 - f.mine_after as i32;
        if waste >= 1 {
            push(
                12 + waste.min(10),
                false,
                format!(
                    "영역이 갈린 뒤라 칸 수 싸움인데, 내 구역에서 {waste}칸을 더 못 쓰게 만들었어요. 구역을 끊지 않도록 가장자리부터 채우는 게 좋아요."
                ),
            );
        } else {
            push(
                3,
                true,
                "영역이 갈린 뒤에는 자기 구역을 빈틈없이 채우는 게 전부예요. 칸을 낭비하지 않고 잘 채웠어요.".into(),
            );
        }
    }

    if f.contested_after + 6 <= f.contested_before && net >= 0 && !f.separated_after {
        push(
            4,
            true,
            format!(
                "경합 중이던 칸을 정리했어요 (양쪽이 똑같이 닿는 칸 {} → {}).",
                f.contested_before, f.contested_after
            ),
        );
    }

    out.sort_by(|a, b| b.weight.cmp(&a.weight));
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    Best,
    Good,
    Inaccuracy,
    Mistake,
    Blunder,
}

impl Grade {
    /// Grade from the mover's win-probability loss versus the best move.
    pub fn from_loss(loss: f32, is_best: bool) -> Grade {
        if is_best || loss < 0.02 {
            Grade::Best
        } else if loss < 0.06 {
            Grade::Good
        } else if loss < 0.12 {
            Grade::Inaccuracy
        } else if loss < 0.25 {
            Grade::Mistake
        } else {
            Grade::Blunder
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Grade::Best => "best",
            Grade::Good => "good",
            Grade::Inaccuracy => "inaccuracy",
            Grade::Mistake => "mistake",
            Grade::Blunder => "blunder",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Grade::Best => "최선",
            Grade::Good => "좋음",
            Grade::Inaccuracy => "부정확",
            Grade::Mistake => "실수",
            Grade::Blunder => "대실수",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Explanation {
    pub grade: Grade,
    /// Mover's win probability before (with best play) and after the move.
    pub p_before: f32,
    pub p_after: f32,
    pub summary: String,
    /// Why the played move is good / bad (most important first).
    pub points: Vec<Point>,
    /// What the recommended move would have achieved instead.
    pub best: Option<Move>,
    pub best_points: Vec<Point>,
    /// The opponent's best answer to the played move and what it threatens.
    pub reply: Option<Move>,
    pub reply_points: Vec<Point>,
    /// Neutral one-line territory status after the move.
    pub balance: String,
    pub gained: Bitboard,
    pub lost: Bitboard,
}

/// Explains `played` in `board`. `p_before` is the mover's win probability
/// with best play (from a search of `board`), `p_after` the mover's win
/// probability after `played` (from a search of the resulting position).
/// `reply` is the opponent's best answer found by that second search.
pub fn explain(
    board: &Board,
    played: Move,
    best: Option<Move>,
    reply: Option<Move>,
    p_before: f32,
    p_after: f32,
) -> Explanation {
    let loss = (p_before - p_after).max(0.0);
    let grade = Grade::from_loss(loss, best == Some(played));
    let weak = matches!(grade, Grade::Inaccuracy | Grade::Mistake | Grade::Blunder);
    let facts = move_facts(board, played);
    let mut points = describe(&facts);

    let mut after = *board;
    after.make_move(played);

    // What the opponent can do next (only worth saying for weaker moves,
    // or when the threat is severe).
    let reply = reply.filter(|m| after.is_legal(*m));
    let reply_points: Vec<Point> = match reply {
        Some(r) => describe_threat(&move_facts(&after, r))
            .into_iter()
            // For sound moves only a severe threat in a position that is
            // not already won is worth a warning.
            .filter(|p| weak || (p.weight >= 20 && p_after < 0.6))
            .take(2)
            .collect(),
        None => Vec::new(),
    };

    // The recommended move: shown only when the played move was not (as
    // good as) the best.
    let best = best.filter(|b| *b != played && grade != Grade::Best && board.is_legal(*b));
    let mut best_points = Vec::new();
    if let Some(b) = best {
        let bf = move_facts(board, b);
        if b.from == played.from && b.to == played.to {
            let gain = bf.arrow_took_from_opp - bf.arrow_cost_me;
            let had = facts.arrow_took_from_opp - facts.arrow_cost_me;
            let why = if gain > had {
                format!(" 그 화살은 상대 땅을 {}칸 더 줄여요.", gain - had)
            } else {
                String::new()
            };
            best_points.push(Point {
                weight: 30,
                positive: true,
                text: format!(
                    "아마존 이동은 좋았어요. 화살을 {} 대신 {}에 쏘는 게 더 나았어요.{why}",
                    square_name(played.arrow),
                    square_name(b.arrow)
                ),
            });
        }
        // Where the two resulting positions differ most in evaluation terms.
        let w = EvalWeights::default();
        let mut pb = *board;
        pb.make_move(b);
        let tb = evaluate_terms(&pb, &w);
        let ta = evaluate_terms(&after, &w);
        // Terms are for the side to move (the opponent): negate for the mover.
        // t2 = king-distance territory: squares reached first walking one
        // step at a time, i.e. land that can actually be held soon.
        let near = -tb.t2 + ta.t2;
        let mob = -tb.mobility + ta.mobility;
        if near >= 3.0 {
            best_points.push(Point {
                weight: 7 + near.min(10.0) as i32,
                positive: true,
                text: format!(
                    "추천 수 {b}{} 아마존 가까이의 땅을 더 단단히 장악해요 (한 칸씩 걸어서 먼저 닿는 칸이 {}칸 더 많아요). 멀리 퀸으로 닿는 땅보다 곧 굳힐 수 있는 가까운 땅이 중요해요.",
                    move_eun(b),
                    near.round() as i32
                ),
            });
        }
        if mob >= 2.0 {
            best_points.push(Point {
                weight: 7 + (2.0 * mob).min(10.0) as i32,
                positive: true,
                text: format!(
                    "추천 수 {b}{} 내 아마존들이 갇히지 않게 활동 공간을 더 잘 지켜요.",
                    move_eun(b)
                ),
            });
        }
        let diff = net_territory(&bf) - net_territory(&facts);
        if diff >= 3 {
            best_points.push(Point {
                weight: diff.min(20),
                positive: true,
                text: format!(
                    "추천 수 {b}{} 둔 수보다 영역 차이를 {diff}칸 더 벌려요.",
                    move_eun(b)
                ),
            });
        }
        for p in describe(&bf) {
            if p.positive && p.weight >= 8 && !points.iter().any(|q| q.text == p.text) {
                best_points.push(p);
            }
        }
        best_points.sort_by(|a, b| b.weight.cmp(&a.weight));
        best_points.truncate(2);
    }

    let pct = |p: f32| (p * 100.0).round() as i32;
    let summary = match grade {
        Grade::Best => {
            if facts.opponent_has_no_moves {
                "이기는 수예요.".to_string()
            } else {
                format!("최선의 수예요. 승률 {}%를 유지했어요.", pct(p_after))
            }
        }
        Grade::Good => format!(
            "좋은 수예요. 최선과 거의 차이가 없어요 (승률 {}% → {}%).",
            pct(p_before),
            pct(p_after)
        ),
        Grade::Inaccuracy => format!(
            "조금 아쉬운 수예요 (승률 {}% → {}%).",
            pct(p_before),
            pct(p_after)
        ),
        Grade::Mistake => format!(
            "실수예요. 승률이 {}%에서 {}%로 떨어졌어요.",
            pct(p_before),
            pct(p_after)
        ),
        Grade::Blunder => format!(
            "큰 실수예요. 승률이 {}%에서 {}%로 크게 떨어졌어요.",
            pct(p_before),
            pct(p_after)
        ),
    };

    // Weak moves lead with what went wrong, good moves with what went right.
    points.sort_by_key(|p| (p.positive == weak, -p.weight));
    points.truncate(3);
    if points.is_empty() && !weak {
        points.push(Point {
            weight: 0,
            positive: true,
            text: "큰 변화 없이 영역과 아마존 활동성의 균형을 유지했어요.".into(),
        });
    }
    if weak && points.iter().all(|p| p.positive) && reply_points.is_empty() {
        let text = if !best_points.is_empty() {
            "이 수 자체의 손해보다는 더 좋은 기회를 놓친 수예요. 아래 추천 수와 비교해 보세요."
                .to_string()
        } else if let Some(r) = reply {
            format!(
                "당장 보이는 손해는 작지만, 상대가 {r}{} 응수하면 형세가 기울어요. 몇 수 앞을 내다보면 더 나은 수가 있었어요.",
                move_ro(r)
            )
        } else {
            "당장 보이는 손해는 작지만, 몇 수 앞을 내다보면 더 나은 수가 있었어요.".into()
        };
        points.insert(
            0,
            Point {
                weight: 0,
                positive: false,
                text,
            },
        );
        points.truncate(3);
    }

    let t = territory(&after);
    let balance = format!(
        "영역 현황 (먼저 닿는 칸): 백 {}칸 · 흑 {}칸 · 경합 {}칸",
        t.owned[0].count_ones(),
        t.owned[1].count_ones(),
        t.contested.count_ones()
    );

    Explanation {
        grade,
        p_before,
        p_after,
        summary,
        points,
        best,
        best_points,
        reply,
        reply_points,
        balance,
        gained: facts.gained,
        lost: facts.lost,
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_points(ps: &[Point]) -> String {
    let items: Vec<String> = ps
        .iter()
        .map(|p| {
            format!(
                "{{\"positive\":{},\"text\":{}}}",
                p.positive,
                json_str(&p.text)
            )
        })
        .collect();
    format!("[{}]", items.join(","))
}

fn json_squares(b: Bitboard) -> String {
    let v: Vec<String> = Bits(b).map(|s| s.to_string()).collect();
    format!("[{}]", v.join(","))
}

impl Explanation {
    pub fn to_json(&self) -> String {
        let best = match self.best {
            Some(m) => format!(
                "{{\"move\":\"{}\",\"sq\":[{},{},{}]}}",
                m, m.from, m.to, m.arrow
            ),
            None => "null".into(),
        };
        let reply = match self.reply {
            Some(m) => json_str(&m.to_string()),
            None => "null".into(),
        };
        format!(
            "{{\"grade\":\"{}\",\"label\":\"{}\",\"pBefore\":{},\"pAfter\":{},\"summary\":{},\"points\":{},\"best\":{},\"bestPoints\":{},\"reply\":{},\"replyPoints\":{},\"balance\":{},\"gained\":{},\"lost\":{}}}",
            self.grade.key(),
            self.grade.label(),
            self.p_before,
            self.p_after,
            json_str(&self.summary),
            json_points(&self.points),
            best,
            json_points(&self.best_points),
            reply,
            json_points(&self.reply_points),
            json_str(&self.balance),
            json_squares(self.gained),
            json_squares(self.lost)
        )
    }
}

/// Color name ("백"/"흑") for callers composing their own text.
pub fn side_name(c: Color) -> &'static str {
    color_name(c)
}

/// Searches `board` and the position after `played` with `iterations` each
/// and explains `played`. Native counterpart of what the web UI does.
pub fn analyze_move(board: &Board, played: Move, cfg: MctsConfig, iterations: u32) -> Explanation {
    let mut pre = Mcts::new(*board, cfg);
    pre.run(iterations);
    let best = pre.best_move();
    let mut after = *board;
    after.make_move(played);
    let mut post = Mcts::new(after, cfg);
    post.run(iterations);
    let p = |v: f32| (v + 1.0) / 2.0;
    // Prefer comparing best and played inside the same tree (separate
    // searches are each optimistic for their side to move).
    let cands = pre.candidates();
    let same_tree = cands
        .iter()
        .find(|c| c.mv == played && c.visits >= (iterations / 30).max(30));
    let (p_before, p_after) = match (same_tree, cands.first()) {
        (Some(c), Some(top)) if pre.root_proven() == 0 && !c.q.is_nan() => (p(top.q), p(c.q)),
        _ => {
            let p_after = if after.has_moves() {
                1.0 - p(post.root_value())
            } else {
                1.0
            };
            // Measure the best move the same way (search after it).
            let p_best = match best {
                Some(b) if b != played && pre.root_proven() == 0 => {
                    let mut ab = *board;
                    ab.make_move(b);
                    if ab.has_moves() {
                        let mut t = Mcts::new(ab, cfg);
                        t.run(iterations);
                        1.0 - p(t.root_value())
                    } else {
                        1.0
                    }
                }
                _ => p(pre.root_value()),
            };
            (p_best, p_after)
        }
    };
    explain(
        board,
        played,
        best,
        post.best_move(),
        p_before.max(p_after),
        p_after,
    )
}
