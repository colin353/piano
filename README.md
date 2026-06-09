# piano

A physics-based piano synthesizer in Rust, hill-climbed against real
Steinway samples. Goal: a piano that is genuinely enjoyable to play live,
with quality comparable to commercial physical-modeling instruments.

## Layout

- `src/` — the Rust synth core and offline renderer.
  - `lib.rs` defines the `Synth` trait: streaming, real-time-safe
    (no allocation in `process`), stereo. Offline rendering is the same
    core run unclocked.
  - `baseline.rs` — the deliberately dumb anchor synth (1-second sine at
    the fundamental). New synths register in `create_synth`.
  - `bin/render.rs` — CLI: render a single note or a MIDI file to WAV,
    plus a worst-case-load real-time-factor benchmark.
- `scorer/` — Python scoring harness (uv project).
- `assets/midi/` — human-eval pieces (Goldberg Variations, from Mutopia).
- `assets/reference/` — git-ignored; fetch with `./fetch-reference.sh`.
- `experiments/` — append-only ledger of scoring runs (JSON per run).

## Workflow

```sh
# Build and render
cargo build --release
./target/release/piano-render note 60 96 -o /tmp/c4.wav --synth baseline
./target/release/piano-render midi assets/midi/goldberg-aria.mid --synth baseline -o out/aria.wav
./target/release/piano-render bench --synth baseline   # real-time factor

# Score against the Steinway reference grid (writes experiments/runs/*.json)
cd scorer
uv run piano-score --synth baseline --quick   # 14 pairs, fast iteration
uv run piano-score --synth baseline           # full 226-pair grid

# Visual diagnostics: spectrogram / envelope / partials, synth vs reference
uv run piano-plot --synth baseline --note 60 --layer FF -o /tmp/c4.png
```

## The objective function

`scorer/piano_scorer/features.py`, versioned via `SCORER_VERSION`. After
onset alignment and early-RMS loudness normalization, each (note, velocity
layer) pair is scored on: multi-resolution log-mel spectrogram distance,
log-RMS envelope distance (decay shape), and partial-domain features —
inharmonicity, partial frequency deviation, partial amplitude profile,
per-partial decay rates. Lower is better; 0 is identical audio.

Calibration of the scale: comparing the *real* FF layer against the real
MF layer of the same note scores ≈ 0.7–2.3 — that is the practical noise
floor. The sine baseline scores ≈ 14. Score history is only comparable
within a `SCORER_VERSION`; bump it on any metric change and re-run the
baseline.

Human eval remains the final judge: render the Goldberg pieces and play
the instrument. The metric exists to make iteration fast, not to be
satisfied.

## Reference samples

[SplendidGrandPiano](https://github.com/sfzinstruments/SplendidGrandPiano)
(AKAI, public domain): 226 stereo 44.1 kHz FLAC samples across four real
dynamic layers — PP (vel 41–67), MP (68–84), MF (85–100), FF (101–127),
representative render velocities 54/76/92/114. Exact note numbers come
from `pitch_keycenter` in `Data/*.txt` (the library's filenames use a
shifted octave convention; never parse them).
