//! Monte Carlo tree search (PUCT) with static-evaluation leaves.
//!
//! - Expanding a node generates every legal move, evaluates each resulting
//!   position with `eval::evaluate`, keeps the best `top_k` as edges and turns
//!   their scores into priors with a softmax. The node's value is the best
//!   child score squashed with `tanh` (a 1-ply minimax estimate), so no random
//!   playouts are used.
//! - Selection uses PUCT (as in AlphaZero), so a learned policy/value can later
//!   replace the evaluation-based priors and values without touching the tree.
//! - Values are in [-1, 1]; an edge's `w` is from the point of view of the
//!   player who makes that move.

use crate::board::{bit, neighbors, queen_reach, Bits, Board, Move, BOARD_MASK};
use crate::eval::{evaluate, EvalWeights};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MctsConfig {
    pub c_puct: f32,
    /// Moves kept per node after static ordering.
    pub top_k: usize,
    /// Two-stage ordering: first score each amazon step (from, to) without
    /// an arrow, then score arrows only for the best `half_k` steps.
    /// 0 = score every full move.
    pub half_k: usize,
    /// Softmax temperature for priors, in evaluation units.
    pub prior_temp: f32,
    /// `value = tanh(score / value_scale)`.
    pub value_scale: f32,
    /// First-play urgency: unvisited edges get `parent value - fpu_reduction`.
    pub fpu_reduction: f32,
    pub weights: EvalWeights,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            c_puct: 1.5,
            top_k: 40,
            half_k: 16,
            prior_temp: 2.0,
            value_scale: 8.0,
            fpu_reduction: 0.2,
            weights: EvalWeights::default(),
        }
    }
}

/// Score given to a move that leaves the opponent without moves.
const WIN_SCORE: f32 = 1000.0;

#[derive(Clone, Copy, Debug)]
struct Edge {
    mv: Move,
    prior: f32,
    n: u32,
    w: f32,
    /// Index of the child node, 0 = not created yet (node 0 is the root).
    child: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct Node {
    first_edge: u32,
    num_edges: u32,
    expanded: bool,
    n: u32,
    /// Static value for the side to move at this node.
    value: f32,
}

/// One candidate at the root, for display.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Candidate {
    pub mv: Move,
    pub visits: u32,
    /// Mean value for the side to move at the root, in [-1, 1].
    pub q: f32,
    pub prior: f32,
}

pub struct Mcts {
    pub config: MctsConfig,
    root: Board,
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    scratch: Vec<Move>,
    scored: Vec<(f32, Move)>,
    path: Vec<u32>,
}

impl Mcts {
    pub fn new(root: Board, config: MctsConfig) -> Mcts {
        let mut m = Mcts {
            config,
            root,
            nodes: Vec::new(),
            edges: Vec::new(),
            scratch: Vec::with_capacity(2200),
            scored: Vec::with_capacity(2200),
            path: Vec::with_capacity(64),
        };
        m.reset(root);
        m
    }

    pub fn reset(&mut self, root: Board) {
        self.root = root;
        self.nodes.clear();
        self.edges.clear();
        self.nodes.push(Node::default());
    }

    pub fn root(&self) -> &Board {
        &self.root
    }

    pub fn iterations(&self) -> u32 {
        self.nodes[0].n
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Expands `node` at position `board`; returns its value for the side to
    /// move.
    fn expand(&mut self, node: usize, board: &mut Board) -> f32 {
        let cfg = self.config;
        score_moves(board, &cfg, &mut self.scratch, &mut self.scored);
        self.nodes[node].expanded = true;
        if self.scored.is_empty() {
            self.nodes[node].value = -1.0;
            return -1.0;
        }
        let k = cfg.top_k.min(self.scored.len()).max(1);
        if self.scored.len() > k {
            self.scored
                .select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            self.scored.truncate(k);
        }
        self.scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let best = self.scored[0].0;
        let mut total = 0.0;
        for e in self.scored.iter_mut() {
            // Reuse the score slot for the unnormalised prior.
            e.0 = ((e.0 - best) / cfg.prior_temp).exp();
            total += e.0;
        }
        let first = self.edges.len() as u32;
        for &(p, mv) in &self.scored {
            self.edges.push(Edge {
                mv,
                prior: p / total,
                n: 0,
                w: 0.0,
                child: 0,
            });
        }
        let value = if best >= WIN_SCORE {
            1.0
        } else {
            (best / cfg.value_scale).tanh()
        };
        let n = &mut self.nodes[node];
        n.first_edge = first;
        n.num_edges = k as u32;
        n.value = value;
        value
    }

    fn select(&self, node: usize) -> usize {
        let nd = &self.nodes[node];
        let sqrt_n = ((nd.n + 1) as f32).sqrt();
        let fpu = nd.value - self.config.fpu_reduction;
        let mut best = f32::NEG_INFINITY;
        let mut best_i = nd.first_edge as usize;
        for i in nd.first_edge as usize..(nd.first_edge + nd.num_edges) as usize {
            let e = &self.edges[i];
            let q = if e.n > 0 { e.w / e.n as f32 } else { fpu };
            let u = self.config.c_puct * e.prior * sqrt_n / (1 + e.n) as f32;
            if q + u > best {
                best = q + u;
                best_i = i;
            }
        }
        best_i
    }

    /// Runs one simulation.
    pub fn iterate(&mut self) {
        let mut board = self.root;
        let mut node = 0usize;
        self.path.clear();
        let mut v;
        loop {
            let nd = self.nodes[node];
            if !nd.expanded {
                v = self.expand(node, &mut board);
                break;
            }
            if nd.num_edges == 0 {
                v = -1.0; // no moves: side to move has lost
                break;
            }
            let ei = self.select(node);
            self.path.push(ei as u32);
            board.make_move(self.edges[ei].mv);
            if self.edges[ei].child == 0 {
                self.nodes.push(Node::default());
                self.edges[ei].child = (self.nodes.len() - 1) as u32;
            }
            node = self.edges[ei].child as usize;
        }
        // Backup. `v` is for the side to move at the leaf; each edge's value
        // is for the player who made it, i.e. the parent's side to move.
        self.nodes[node].n += 1;
        self.nodes[0].n += (node != 0) as u32;
        for i in (0..self.path.len()).rev() {
            v = -v;
            let e = &mut self.edges[self.path[i] as usize];
            e.n += 1;
            e.w += v;
            if i > 0 {
                let parent = self.edges[self.path[i - 1] as usize].child as usize;
                self.nodes[parent].n += 1;
            }
        }
    }

    pub fn run(&mut self, iterations: u32) {
        for _ in 0..iterations {
            self.iterate();
        }
    }

    /// Most-visited edge of an expanded node (ties: higher prior, then the
    /// earlier edge).
    fn most_visited(&self, node: usize) -> usize {
        let nd = &self.nodes[node];
        let mut best = nd.first_edge as usize;
        for i in best + 1..(nd.first_edge + nd.num_edges) as usize {
            let (e, b) = (&self.edges[i], &self.edges[best]);
            if e.n > b.n || (e.n == b.n && e.prior > b.prior) {
                best = i;
            }
        }
        best
    }

    /// Root candidates sorted by visits (then prior).
    pub fn candidates(&self) -> Vec<Candidate> {
        let root = &self.nodes[0];
        if !root.expanded {
            return Vec::new();
        }
        let mut v: Vec<Candidate> = (root.first_edge..root.first_edge + root.num_edges)
            .map(|i| {
                let e = &self.edges[i as usize];
                Candidate {
                    mv: e.mv,
                    visits: e.n,
                    q: if e.n > 0 { e.w / e.n as f32 } else { f32::NAN },
                    prior: e.prior,
                }
            })
            .collect();
        // Stable sort keeps edge order on full ties, matching `most_visited`.
        v.sort_by(|a, b| b.visits.cmp(&a.visits).then(b.prior.total_cmp(&a.prior)));
        v
    }

    pub fn best_move(&self) -> Option<Move> {
        let root = &self.nodes[0];
        if !root.expanded || root.num_edges == 0 {
            return None;
        }
        Some(self.edges[self.most_visited(0)].mv)
    }

    /// Value of the root for its side to move, in [-1, 1].
    pub fn root_value(&self) -> f32 {
        match self.candidates().first() {
            Some(c) if c.visits > 0 => c.q,
            _ => self.nodes[0].value,
        }
    }

    /// Principal variation: most-visited edges from the root.
    pub fn pv(&self, max_len: usize) -> Vec<Move> {
        let mut out = Vec::new();
        let mut node = 0usize;
        while out.len() < max_len {
            let nd = &self.nodes[node];
            if !nd.expanded || nd.num_edges == 0 {
                break;
            }
            let e = &self.edges[self.most_visited(node)];
            if e.n == 0 {
                break;
            }
            out.push(e.mv);
            if e.child == 0 {
                break;
            }
            node = e.child as usize;
        }
        out
    }
}

/// Score of `m` for the player making it.
#[inline]
fn score_after(board: &mut Board, m: Move, w: &EvalWeights) -> f32 {
    board.make_move(m);
    let s = if board.has_moves() {
        -evaluate(board, w)
    } else {
        WIN_SCORE
    };
    board.unmake_move(m);
    s
}

/// Fills `out` with `(score, move)` for the candidate moves of `board`
/// (all moves, or the two-stage subset when `cfg.half_k > 0`).
fn score_moves(
    board: &mut Board,
    cfg: &MctsConfig,
    scratch: &mut Vec<Move>,
    out: &mut Vec<(f32, Move)>,
) {
    out.clear();
    let w = &cfg.weights;
    if cfg.half_k == 0 {
        scratch.clear();
        board.generate_moves(scratch);
        for &m in scratch.iter() {
            out.push((score_after(board, m, w), m));
        }
        return;
    }
    // Stage 1: amazon steps, evaluated with the arrow not yet shot.
    let occ = board.occupied();
    let me = board.side.index();
    let opp = board.amazons[1 - me];
    let opp_nbrs = Bits(opp).fold(0, |acc, s| acc | neighbors(s));
    let mut steps: Vec<(f32, u8, u8)> = Vec::with_capacity(128);
    for from in Bits(board.amazons[me]) {
        let occ_wo = occ ^ bit(from);
        for to in Bits(queen_reach(from, occ_wo)) {
            // The opponent can be sealed by one arrow iff after the step it
            // has at most one free square next to its amazons and the arrow
            // reaches it. Such steps always survive the cut.
            let libs = opp_nbrs & !(occ_wo | bit(to)) & BOARD_MASK;
            let wins = libs == 0 || (libs.count_ones() == 1 && queen_reach(to, occ_wo) & libs != 0);
            let score = if wins {
                WIN_SCORE
            } else {
                let mut b = *board;
                b.amazons[me] ^= bit(from) | bit(to);
                b.side = b.side.opposite();
                -evaluate(&b, w)
            };
            steps.push((score, from, to));
        }
    }
    if steps.len() > cfg.half_k {
        steps.select_nth_unstable_by(cfg.half_k - 1, |a, b| b.0.total_cmp(&a.0));
        steps.truncate(cfg.half_k);
    }
    // Stage 2: every arrow for the surviving steps.
    for &(_, from, to) in &steps {
        let occ_wo = occ ^ bit(from);
        for arrow in Bits(queen_reach(to, occ_wo)) {
            let m = Move::new(from, to, arrow);
            out.push((score_after(board, m, w), m));
        }
    }
}

/// Best move by one-ply static evaluation (ties broken by move order).
pub fn greedy_move(board: &Board, weights: &EvalWeights) -> Option<Move> {
    let mut b = *board;
    let mut best: Option<(f32, Move)> = None;
    for m in board.legal_moves() {
        b.make_move(m);
        let s = if b.has_moves() {
            -evaluate(&b, weights)
        } else {
            WIN_SCORE
        };
        b.unmake_move(m);
        if best.is_none_or(|(bs, _)| s > bs) {
            best = Some((s, m));
        }
    }
    best.map(|(_, m)| m)
}
