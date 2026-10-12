//! Checks the Rust network against the trainer's outputs and times it.
//! cargo run --release --example nncheck -- <net.bin> <dump.bin>

use std::time::Instant;

use amazons_engine::board::Board;
use amazons_engine::nn::{Net, PLANES};

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let net = Net::from_bytes(&std::fs::read(&args[0]).unwrap()).unwrap();
    let dump = std::fs::read(&args[1]).unwrap();
    let n = u32::from_le_bytes(dump[0..4].try_into().unwrap()) as usize;
    let all = f32s(&dump[4..]);
    let (inputs, rest) = all.split_at(n * PLANES * 100);
    let (values, logits) = rest.split_at(n);
    let mut worst = 0f32;
    for i in 0..n {
        let out = net.eval_planes(&inputs[i * PLANES * 100..(i + 1) * PLANES * 100]);
        worst = worst.max((out.value - values[i]).abs());
        let l = &logits[i * 300..(i + 1) * 300];
        for k in 0..100 {
            worst = worst.max((out.from[k] - l[k]).abs());
            worst = worst.max((out.to[k] - l[100 + k]).abs());
            worst = worst.max((out.arrow[k] - l[200 + k]).abs());
        }
    }
    println!(
        "C={} B={}: max |rust - numpy| over {n} positions = {worst:.2e}",
        net.channels, net.blocks
    );
    let b = Board::new();
    let iters = 2000;
    let t = Instant::now();
    let mut acc = 0f32;
    for _ in 0..iters {
        acc += net.eval(std::hint::black_box(&b)).value;
    }
    std::hint::black_box(acc);
    println!(
        "eval (with feature planes): {:.1} us",
        t.elapsed().as_secs_f64() / iters as f64 * 1e6
    );
}
