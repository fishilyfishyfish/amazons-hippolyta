#!/usr/bin/env bash
# Builds the engine to WebAssembly and writes the JS bindings to web/pkg/.
# Requires: rustup target wasm32-unknown-unknown, wasm-bindgen-cli matching
# the wasm-bindgen version pinned in engine/Cargo.toml.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/engine"
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --no-typescript --out-dir "$root/web/pkg" \
  target/wasm32-unknown-unknown/release/amazons_engine.wasm
ls -l "$root/web/pkg"
