# Live tuning app (WASM)

The synth core compiled to WebAssembly with a browser UI: live knobs, a
keyboard (on-screen / computer keys / Web MIDI), one-click test strikes,
real-Steinway reference playback for A/B, and a "copy settings" export.

## Run

```sh
cd web
python3 -m http.server 8080
# open http://localhost:8080  (Chrome recommended — Web MIDI + WASM)
```

Press **Start audio**, then **Strike C4 ff** (or play the keyboard). Move
the sliders and listen — changes apply live. Use the Reference buttons to
A/B against the real Steinway. When it sounds right, **Copy as env vars**
and paste them back to me to bake as defaults.

Audio runs on a ScriptProcessorNode (~46 ms latency) — fine for tuning;
not meant for virtuoso performance (use `piano-live` for that).

## Rebuild after Rust changes

```sh
./build-wasm.sh    # needs: rustup target add wasm32-unknown-unknown
                   #        cargo install wasm-bindgen-cli --version 0.2.100
```

`pkg/` (generated wasm + JS glue) and `refs/*.wav` are committed so the app
runs without a build step. The knob names match the `PIANO_*` env vars the
CLI uses, so exported settings drop straight into the source defaults.
