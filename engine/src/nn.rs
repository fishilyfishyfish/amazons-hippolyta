//! Neural-network input features and the small network used for evaluation.
//!
//! Positions are always seen from the side to move ("canonical" view): when
//! Black is to move the board is flipped vertically and the colors swapped,
//! so the network only ever learns one side. Moves are mapped the same way.
//! This module is the single source of truth for the features: the training
//! data generator and the inference code both call `planes`.

use crate::board::{Bitboard, Bits, Board, Color, Move, BOARD_MASK};
use crate::eval::{king_levels, queen_levels};

/// Number of input planes.
pub const PLANES: usize = 13;

/// Vertical flip of a square (row r -> 9 - r).
#[inline]
pub fn flip_sq(sq: u8) -> u8 {
    (9 - sq / 10) * 10 + sq % 10
}

/// Left-right mirror of a square (file a <-> j).
#[inline]
pub fn mirror_sq(sq: u8) -> u8 {
    sq / 10 * 10 + (9 - sq % 10)
}

fn map_bb(b: Bitboard, f: impl Fn(u8) -> u8) -> Bitboard {
    Bits(b).fold(0, |acc, s| acc | 1u128 << f(s))
}

/// The position seen from the side to move: "my" amazons are
/// `amazons[0]`, and White is always to move.
pub fn canonical(board: &Board) -> Board {
    match board.side {
        Color::White => *board,
        Color::Black => Board {
            amazons: [
                map_bb(board.amazons[1], flip_sq),
                map_bb(board.amazons[0], flip_sq),
            ],
            arrows: map_bb(board.arrows, flip_sq),
            side: Color::White,
        },
    }
}

/// A move in canonical coordinates (and back: the map is its own inverse).
pub fn canonical_move(board: &Board, m: Move) -> Move {
    match board.side {
        Color::White => m,
        Color::Black => Move::new(flip_sq(m.from), flip_sq(m.to), flip_sq(m.arrow)),
    }
}

/// Input planes as bitboards (canonical view):
/// 0 my amazons, 1 opponent amazons, 2 arrows,
/// 3/4/5 squares at my queen distance 1 / 2 / any,
/// 6/7/8 the same for the opponent,
/// 9/10 squares at king distance 1 for me / the opponent,
/// 11/12 squares I / the opponent reach strictly first by queen moves.
pub fn planes(board: &Board) -> [Bitboard; PLANES] {
    let c = canonical(board);
    let occ = c.occupied();
    let q = [
        queen_levels(c.amazons[0], occ),
        queen_levels(c.amazons[1], occ),
    ];
    let k = [
        king_levels(c.amazons[0], occ),
        king_levels(c.amazons[1], occ),
    ];
    let mut closer = [0u128; 2];
    let (mut cum0, mut cum1) = (0u128, 0u128);
    for d in 1..=q[0].depth.max(q[1].depth) {
        cum0 |= q[0].levels[d];
        cum1 |= q[1].levels[d];
        closer[0] |= q[0].levels[d] & !cum1;
        closer[1] |= q[1].levels[d] & !cum0;
    }
    [
        c.amazons[0],
        c.amazons[1],
        c.arrows,
        q[0].levels[1],
        q[0].levels[2],
        q[0].reached,
        q[1].levels[1],
        q[1].levels[2],
        q[1].reached,
        k[0].levels[1],
        k[1].levels[1],
        closer[0] & BOARD_MASK,
        closer[1] & BOARD_MASK,
    ]
}

/// Planes as a dense `[PLANES * 100]` 0/1 vector (plane-major).
pub fn dense_planes(board: &Board) -> Vec<f32> {
    let p = planes(board);
    let mut out = vec![0f32; PLANES * 100];
    for (i, bb) in p.iter().enumerate() {
        for s in Bits(*bb) {
            out[i * 100 + s as usize] = 1.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Network
//
// Input PLANES x 10 x 10 -> conv3x3(C) + ReLU -> B residual blocks
// (conv3x3, ReLU, conv3x3, + skip, ReLU) -> two heads:
//   value:  conv1x1(2) + ReLU -> fc(200 -> 64) + ReLU -> fc(64 -> 1) -> tanh
//   policy: conv1x1(3) -> logits for from / to / arrow squares (3 x 100)
// Batch norm is folded into the conv weights by the exporter.
//
// Weight file (little endian): b"AMNN", u32 version = 1, u32 C, u32 B, then
// f32 arrays in this order: stem w[C][PLANES][3][3], b[C]; per block:
// w1[C][C][3][3], b1[C], w2[C][C][3][3], b2[C]; value conv w[2][C], b[2];
// fc1 w[64][200], b[64]; fc2 w[1][64], b[1]; policy conv w[3][C], b[3].

struct Conv {
    cin: usize,
    cout: usize,
    k: usize,     // 1 or 3
    wt: Vec<f32>, // transposed: [cin * k * k][cout]
    b: Vec<f32>,
}

pub struct Net {
    pub channels: usize,
    pub blocks: usize,
    stem: Conv,
    res: Vec<(Conv, Conv)>,
    vconv: Conv,
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    fc2_w: Vec<f32>,
    fc2_b: f32,
    pconv: Conv,
}

/// Network output for one position, in canonical coordinates.
pub struct NetOutput {
    /// Expected result for the side to move, in [-1, 1].
    pub value: f32,
    /// Logits for the from / to / arrow square.
    pub from: [f32; 100],
    pub to: [f32; 100],
    pub arrow: [f32; 100],
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn u32(&mut self) -> Result<u32, String> {
        let b = self
            .data
            .get(self.pos..self.pos + 4)
            .ok_or("truncated weights")?;
        self.pos += 4;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let b = self
            .data
            .get(self.pos..self.pos + 4 * n)
            .ok_or("truncated weights")?;
        self.pos += 4 * n;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
    fn conv(&mut self, cin: usize, cout: usize, k: usize) -> Result<Conv, String> {
        let w = self.f32s(cout * cin * k * k)?;
        let b = self.f32s(cout)?;
        let kdim = cin * k * k;
        let mut wt = vec![0f32; w.len()];
        for o in 0..cout {
            for x in 0..kdim {
                wt[x * cout + o] = w[o * kdim + x];
            }
        }
        Ok(Conv {
            cin,
            cout,
            k,
            wt,
            b,
        })
    }
}

/// For each of the 9 offsets of a 3x3 kernel, the (dst, src) square pairs
/// that stay on the board.
fn neighbor_table() -> &'static [Vec<(u8, u8)>; 9] {
    use std::sync::OnceLock;
    static T: OnceLock<[Vec<(u8, u8)>; 9]> = OnceLock::new();
    T.get_or_init(|| {
        std::array::from_fn(|k| {
            let (dr, dc) = (k as i32 / 3 - 1, k as i32 % 3 - 1);
            let mut v = Vec::new();
            for sq in 0..100i32 {
                let (r, c) = (sq / 10 + dr, sq % 10 + dc);
                if (0..10).contains(&r) && (0..10).contains(&c) {
                    v.push((sq as u8, (r * 10 + c) as u8));
                }
            }
            v
        })
    })
}

impl Conv {
    /// `input`: [cin][100] -> returns [cout][100] (no activation).
    fn forward(&self, input: &[f32], cols: &mut Vec<f32>) -> Vec<f32> {
        let kk = self.k * self.k;
        let kdim = self.cin * kk;
        // im2col: cols[ci * kk + k][sq]
        if self.k == 3 {
            cols.clear();
            cols.resize(kdim * 100, 0.0);
            let table = neighbor_table();
            for ci in 0..self.cin {
                let src = &input[ci * 100..ci * 100 + 100];
                for (k, pairs) in table.iter().enumerate() {
                    let row = &mut cols[(ci * 9 + k) * 100..(ci * 9 + k) * 100 + 100];
                    for &(d, s) in pairs {
                        row[d as usize] = src[s as usize];
                    }
                }
            }
        }
        let cols: &[f32] = if self.k == 3 { cols } else { input };
        let mut out = vec![0f32; self.cout * 100];
        for o in 0..self.cout {
            out[o * 100..o * 100 + 100].fill(self.b[o]);
        }
        // Each input row is read once and added into every output channel;
        // the output block (cout x 100 floats) stays in L1.
        for kx in 0..kdim {
            let src: &[f32; 100] = cols[kx * 100..kx * 100 + 100].try_into().unwrap();
            let ws = &self.wt[kx * self.cout..(kx + 1) * self.cout];
            for (o, &w) in ws.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let dst: &mut [f32; 100] = (&mut out[o * 100..o * 100 + 100]).try_into().unwrap();
                for i in 0..100 {
                    dst[i] += w * src[i];
                }
            }
        }
        out
    }
}

fn relu(v: &mut [f32]) {
    for x in v {
        *x = x.max(0.0);
    }
}

impl Net {
    pub fn from_bytes(data: &[u8]) -> Result<Net, String> {
        if data.get(0..4) != Some(b"AMNN") {
            return Err("not an AMNN weight file".into());
        }
        let mut r = Reader { data, pos: 4 };
        if r.u32()? != 1 {
            return Err("unsupported weight file version".into());
        }
        let c = r.u32()? as usize;
        let blocks = r.u32()? as usize;
        let stem = r.conv(PLANES, c, 3)?;
        let mut res = Vec::new();
        for _ in 0..blocks {
            res.push((r.conv(c, c, 3)?, r.conv(c, c, 3)?));
        }
        let vconv = r.conv(c, 2, 1)?;
        let fc1_w = r.f32s(64 * 200)?;
        let fc1_b = r.f32s(64)?;
        let fc2_w = r.f32s(64)?;
        let fc2_b = r.f32s(1)?[0];
        let pconv = r.conv(c, 3, 1)?;
        if r.pos != data.len() {
            return Err(format!(
                "{} trailing bytes in weight file",
                data.len() - r.pos
            ));
        }
        Ok(Net {
            channels: c,
            blocks,
            stem,
            res,
            vconv,
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
            pconv,
        })
    }

    /// Evaluates a position (from the side to move's point of view; policy in
    /// canonical coordinates, see `canonical_move`).
    pub fn eval(&self, board: &Board) -> NetOutput {
        self.eval_planes(&dense_planes(board))
    }

    pub fn eval_planes(&self, input: &[f32]) -> NetOutput {
        let mut cols = Vec::new();
        let mut x = self.stem.forward(input, &mut cols);
        relu(&mut x);
        for (c1, c2) in &self.res {
            let mut h = c1.forward(&x, &mut cols);
            relu(&mut h);
            let mut h2 = c2.forward(&h, &mut cols);
            for (a, b) in h2.iter_mut().zip(&x) {
                *a += b;
            }
            relu(&mut h2);
            x = h2;
        }
        let mut v = self.vconv.forward(&x, &mut cols);
        relu(&mut v);
        let mut hidden = self.fc1_b.clone();
        for (j, hj) in hidden.iter_mut().enumerate() {
            let w = &self.fc1_w[j * 200..j * 200 + 200];
            *hj += w.iter().zip(&v).map(|(a, b)| a * b).sum::<f32>();
            *hj = hj.max(0.0);
        }
        let value = (self.fc2_b
            + self
                .fc2_w
                .iter()
                .zip(&hidden)
                .map(|(a, b)| a * b)
                .sum::<f32>())
        .tanh();
        let p = self.pconv.forward(&x, &mut cols);
        NetOutput {
            value,
            from: p[0..100].try_into().unwrap(),
            to: p[100..200].try_into().unwrap(),
            arrow: p[200..300].try_into().unwrap(),
        }
    }
}

/// Prior logit of `m` (real coordinates) under the factorised policy.
#[inline]
pub fn move_logit(out: &NetOutput, board: &Board, m: Move) -> f32 {
    let c = canonical_move(board, m);
    out.from[c.from as usize] + out.to[c.to as usize] + out.arrow[c.arrow as usize]
}

// ---------------------------------------------------------------------------
// Registry, so that `MctsConfig` (which is `Copy`) can refer to a network by
// a small id.

use std::sync::{Arc, Mutex, OnceLock};

fn registry() -> &'static Mutex<Vec<Arc<Net>>> {
    static R: OnceLock<Mutex<Vec<Arc<Net>>>> = OnceLock::new();
    R.get_or_init(|| Mutex::new(Vec::new()))
}

/// Registers a network and returns its id (1-based) for `MctsConfig::net`.
pub fn register_net(net: Net) -> u8 {
    let mut r = registry().lock().unwrap();
    r.push(Arc::new(net));
    assert!(r.len() < 256, "too many networks");
    r.len() as u8
}

/// The network with this id (`None` for 0 or an unknown id).
pub fn get_net(id: u8) -> Option<Arc<Net>> {
    if id == 0 {
        return None;
    }
    registry().lock().unwrap().get(id as usize - 1).cloned()
}

/// Loads (once per path) and registers a weight file; returns its id.
pub fn load_net_file(path: &str) -> Result<u8, String> {
    static LOADED: OnceLock<Mutex<Vec<(String, u8)>>> = OnceLock::new();
    let loaded = LOADED.get_or_init(|| Mutex::new(Vec::new()));
    let mut l = loaded.lock().unwrap();
    if let Some((_, id)) = l.iter().find(|(p, _)| p == path) {
        return Ok(*id);
    }
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    let id = register_net(Net::from_bytes(&data)?);
    l.push((path.to_string(), id));
    Ok(id)
}
