//! Named player configurations, used by the arena (self-play strength
//! measurement) and by the web UI.
//!
//! Spec syntax: `random`, `greedy`, `mcts:<iterations>` or
//! `mcts-ms:<milliseconds>`, optionally followed by `,key=value` overrides of
//! `MctsConfig` fields (`c`, `top_k`, `half_k`, `temp`, `scale`, `fpu`) and
//! evaluation weights (`w_tie`, `w_phase`, `w_t1e`, `w_t1o`, `w_t2`, `w_c1`,
//! `w_c2`, `w_mob`).

use std::time::{Duration, Instant};

use crate::board::{Board, Move};
use crate::eval::EvalWeights;
use crate::rng::Rng;
use crate::search::{greedy_move, merge_candidates, Mcts, MctsConfig};

fn run_budget(t: &mut Mcts, budget: Budget) {
    match budget {
        Budget::Iterations(n) => t.run(n),
        Budget::Millis(ms) => {
            let end = Instant::now() + Duration::from_millis(ms);
            // Always at least one batch so a move exists.
            loop {
                t.run(16);
                if Instant::now() >= end {
                    break;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    Iterations(u32),
    Millis(u64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlayerSpec {
    Random,
    Greedy(EvalWeights),
    /// Config, budget per tree, number of independent trees (root
    /// parallelism, run one after another so results don't depend on cores;
    /// trees after the first use `jitter` 0.5 unless set).
    Mcts(MctsConfig, Budget, u32),
}

impl PlayerSpec {
    pub fn parse(s: &str) -> Result<PlayerSpec, String> {
        let mut parts = s.split(',');
        let head = parts.next().unwrap_or("").trim();
        let mut cfg = MctsConfig::default();
        let mut par = 1u32;
        for kv in parts {
            let (k, v) = kv.split_once('=').ok_or(format!("bad option {kv:?}"))?;
            let num = |v: &str| v.parse::<f32>().map_err(|e| format!("{k}: {e}"));
            match k.trim() {
                "c" => cfg.c_puct = num(v)?,
                "top_k" => cfg.top_k = num(v)? as usize,
                "half_k" => cfg.half_k = num(v)? as usize,
                "temp" => cfg.prior_temp = num(v)?,
                "scale" => cfg.value_scale = num(v)?,
                "escale" => cfg.endgame_scale = num(v)?,
                "fpu" => cfg.fpu_reduction = num(v)?,
                "jitter" => cfg.jitter = num(v)?,
                "par" => par = num(v)? as u32,
                "w_tie" => cfg.weights.tie = num(v)?,
                "w_phase" => cfg.weights.phase_w = num(v)?,
                "w_t1e" => cfg.weights.t1_end = num(v)?,
                "w_t1o" => cfg.weights.t1_open = num(v)?,
                "w_t2" => cfg.weights.t2 = num(v)?,
                "w_c1" => cfg.weights.c1 = num(v)?,
                "w_c2" => cfg.weights.c2 = num(v)?,
                "w_mob" => cfg.weights.mobility = num(v)?,
                _ => return Err(format!("unknown option {k:?}")),
            }
        }
        let (kind, arg) = head.split_once(':').unwrap_or((head, ""));
        let n = || arg.parse::<u64>().map_err(|e| format!("{head}: {e}"));
        match kind {
            "random" => Ok(PlayerSpec::Random),
            "greedy" => Ok(PlayerSpec::Greedy(cfg.weights)),
            "mcts" => Ok(PlayerSpec::Mcts(cfg, Budget::Iterations(n()? as u32), par)),
            "mcts-ms" => Ok(PlayerSpec::Mcts(cfg, Budget::Millis(n()?), par)),
            _ => Err(format!("unknown player {head:?}")),
        }
    }
}

pub struct Player {
    pub spec: PlayerSpec,
    rng: Rng,
}

impl Player {
    pub fn new(spec: PlayerSpec, seed: u64) -> Player {
        Player {
            spec,
            rng: Rng::new(seed),
        }
    }

    pub fn choose(&mut self, board: &Board) -> Option<Move> {
        match self.spec {
            PlayerSpec::Random => crate::random_move(board, &mut self.rng),
            PlayerSpec::Greedy(w) => greedy_move(board, &w),
            PlayerSpec::Mcts(cfg, budget, par) => {
                let mut lists = Vec::new();
                let mut proven_win = None;
                for i in 0..par.max(1) {
                    let mut c = cfg;
                    if i > 0 {
                        c.seed = self.rng.next_u64();
                        if c.jitter == 0.0 {
                            c.jitter = 0.5;
                        }
                    }
                    let mut t = Mcts::new(*board, c);
                    run_budget(&mut t, budget);
                    if t.root_proven() == 1 {
                        proven_win = t.best_move();
                    }
                    lists.push(t.candidates());
                }
                proven_win.or_else(|| merge_candidates(&lists).first().map(|c| c.mv))
            }
        }
    }
}
