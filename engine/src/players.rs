//! Named player configurations, used by the arena (self-play strength
//! measurement) and by the web UI.
//!
//! Spec syntax: `random`, `greedy`, `human:<1-3>` (human-like levels),
//! `level:<1-5>` (the web difficulty levels), `mcts:<iterations>` or
//! `mcts-ms:<milliseconds>`, optionally followed by `,key=value` overrides of
//! `MctsConfig` fields (`c`, `top_k`, `half_k`, `temp`, `scale`, `fpu`) and
//! evaluation weights (`w_tie`, `w_phase`, `w_t1e`, `w_t1o`, `w_t2`, `w_c1`,
//! `w_c2`, `w_mob`).

use std::time::{Duration, Instant};

use crate::board::{Board, Move};
use crate::eval::EvalWeights;
use crate::human::{game_style, human_level, human_move, HumanLevel};
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
    /// Human-like player (`human.rs`).
    Human(HumanLevel),
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
        let mut hopts: Vec<(String, f32)> = Vec::new();
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
                "nn" => cfg.net = crate::nn::load_net_file(v.trim())?,
                "ntemp" => cfg.net_temp = num(v)?,
                "nval" => cfg.net_value = num(v)? as u8,
                "npol" => cfg.net_policy = num(v)? != 0.0,
                "par" => par = num(v)? as u32,
                k if k.starts_with('h') => hopts.push((k.to_string(), num(v)?)),
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
            "human" => {
                let mut h = human_level(n()?.clamp(1, 3) as u8);
                for (k, v) in hopts {
                    match k.as_str() {
                        "hv" => h.vision = v as usize,
                        "hs" => h.sharpness = v,
                        "ha" => h.arrows = v as usize,
                        "hc" => h.check = v as usize,
                        "hrv" => h.reply_vision = v as usize,
                        "hra" => h.reply_arrows = v as usize,
                        "ht" => h.choice_temp = v,
                        "het" => h.endgame_temp = v,
                        "hw" => h.spot_win = v,
                        "hst" => h.style = v,
                        "hi" => h.intuition = v,
                        _ => return Err(format!("unknown option {k:?}")),
                    }
                }
                Ok(PlayerSpec::Human(h))
            }
            "level" => Ok(level_spec(n()? as u8)),
            "mcts" => Ok(PlayerSpec::Mcts(cfg, Budget::Iterations(n()? as u32), par)),
            "mcts-ms" => Ok(PlayerSpec::Mcts(cfg, Budget::Millis(n()?), par)),
            _ => Err(format!("unknown player {head:?}")),
        }
    }
}

/// MCTS iterations of difficulty levels 4 and 5 (fixed iterations so the
/// strength does not depend on the device; the web's level 5 is the full
/// time-limited parallel search, roughly this many iterations or more).
pub const LEVEL4_ITERATIONS: u32 = 30;
pub const LEVEL5_ITERATIONS: u32 = 8000;

/// The player of a web difficulty level (1–5).
pub fn level_spec(level: u8) -> PlayerSpec {
    match level {
        0..=3 => PlayerSpec::Human(human_level(level)),
        4 => PlayerSpec::Mcts(
            MctsConfig::default(),
            Budget::Iterations(LEVEL4_ITERATIONS),
            1,
        ),
        _ => PlayerSpec::Mcts(
            MctsConfig::default(),
            Budget::Iterations(LEVEL5_ITERATIONS),
            1,
        ),
    }
}

pub struct Player {
    pub spec: PlayerSpec,
    rng: Rng,
    /// Per-game evaluation style of a human-like player.
    style: EvalWeights,
}

impl Player {
    pub fn new(spec: PlayerSpec, seed: u64) -> Player {
        let mut rng = Rng::new(seed);
        let style = match spec {
            PlayerSpec::Human(l) => game_style(&l, &mut rng),
            _ => EvalWeights::default(),
        };
        Player { spec, rng, style }
    }

    pub fn choose(&mut self, board: &Board) -> Option<Move> {
        self.choose_in_game(board, &[])
    }

    /// Like `choose`, with the moves played so far (human-like players react
    /// to the last moves).
    pub fn choose_in_game(&mut self, board: &Board, history: &[Move]) -> Option<Move> {
        match self.spec {
            PlayerSpec::Human(l) => {
                if !board.has_moves() {
                    return None;
                }
                human_move(board, history, &l, &self.style, &mut self.rng)
            }
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
