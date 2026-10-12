//! Named player configurations, used by the arena (self-play strength
//! measurement) and by the web UI.
//!
//! Spec syntax: `random`, `greedy`, `mcts:<iterations>` or
//! `mcts-ms:<milliseconds>`, optionally followed by `,key=value` overrides of
//! `MctsConfig` fields (`c`, `top_k`, `half_k`, `temp`, `scale`, `fpu`).

use std::time::{Duration, Instant};

use crate::board::{Board, Move};
use crate::eval::EvalWeights;
use crate::rng::Rng;
use crate::search::{greedy_move, Mcts, MctsConfig};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Budget {
    Iterations(u32),
    Millis(u64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PlayerSpec {
    Random,
    Greedy(EvalWeights),
    Mcts(MctsConfig, Budget),
}

impl PlayerSpec {
    pub fn parse(s: &str) -> Result<PlayerSpec, String> {
        let mut parts = s.split(',');
        let head = parts.next().unwrap_or("").trim();
        let mut cfg = MctsConfig::default();
        for kv in parts {
            let (k, v) = kv.split_once('=').ok_or(format!("bad option {kv:?}"))?;
            let num = |v: &str| v.parse::<f32>().map_err(|e| format!("{k}: {e}"));
            match k.trim() {
                "c" => cfg.c_puct = num(v)?,
                "top_k" => cfg.top_k = num(v)? as usize,
                "half_k" => cfg.half_k = num(v)? as usize,
                "temp" => cfg.prior_temp = num(v)?,
                "scale" => cfg.value_scale = num(v)?,
                "fpu" => cfg.fpu_reduction = num(v)?,
                _ => return Err(format!("unknown option {k:?}")),
            }
        }
        let (kind, arg) = head.split_once(':').unwrap_or((head, ""));
        let n = || arg.parse::<u64>().map_err(|e| format!("{head}: {e}"));
        match kind {
            "random" => Ok(PlayerSpec::Random),
            "greedy" => Ok(PlayerSpec::Greedy(cfg.weights)),
            "mcts" => Ok(PlayerSpec::Mcts(cfg, Budget::Iterations(n()? as u32))),
            "mcts-ms" => Ok(PlayerSpec::Mcts(cfg, Budget::Millis(n()?))),
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
            PlayerSpec::Mcts(cfg, budget) => {
                let mut t = Mcts::new(*board, cfg);
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
                t.best_move()
            }
        }
    }
}
