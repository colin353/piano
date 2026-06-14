#!/bin/sh
# Build the WASM tuning app. Requires:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.100
set -e
cd "$(dirname "$0")/.."
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --no-typescript --out-dir web/pkg \
    target/wasm32-unknown-unknown/release/piano.wasm
echo "built web/pkg/. Serve with:  (cd web && python3 -m http.server 8080)"
echo "then open http://localhost:8080"
