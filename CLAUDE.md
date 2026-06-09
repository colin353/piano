# piano — physics-based piano synth

Read README.md for layout and workflow. Rules that aren't obvious from it:

- **Real-time is a hard constraint.** The end goal is live playing. Synth
  `process()` must never allocate, lock, or do I/O. Run
  `piano-render bench --synth <name>` after synth changes; real-time
  factor must stay comfortably above 1 (target >5x for headroom).
- **Scorer discipline.** Any change to `scorer/piano_scorer/features.py`
  metrics invalidates all prior scores: bump `SCORER_VERSION`, re-run the
  baseline, and never compare losses across versions.
- **Experiment ledger.** `experiments/runs/` is append-only; every scoring
  run lands there automatically. Commit run records together with the
  synth change that produced them.
- **Reference samples**: run `./fetch-reference.sh` if
  `assets/reference/SplendidGrandPiano` is missing. Note numbers come from
  `pitch_keycenter` in `Data/*.txt`, never from sample filenames.
- **Score scale (v3)**: 0 = identical; ~0.7–1.8 = real adjacent dynamic
  layers of the same note (practical noise floor); ~13.8 = sine baseline;
  modal-v2 sits at ~3.1 full-grid mean.
- **Ear-features**: some realism features are score-invisible or carry a
  small documented score cost (pitch glide, reverb, releases, pedaling).
  Renders for humans use the room (`--legato`, `--auto-pedal` for typeset
  MIDI); scoring renders stay dry by construction.
- Use `piano-plot` to inspect synth-vs-reference spectrograms, envelopes,
  and partials when scores move in unexpected ways — read the PNG.
- Python runs via `uv` from `scorer/` (`uv run piano-score ...`). The
  system Python has no pip.
