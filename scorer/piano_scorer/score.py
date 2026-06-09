"""Score a synth against the Steinway reference grid and append the result
to the experiment ledger.

Usage (from repo root):
    cd scorer && uv run piano-score --synth baseline [--quick]

Renders every (note, layer) pair in the reference map with the Rust
piano-render binary, compares each against the real sample, and writes a
JSON run record to experiments/runs/.
"""

import argparse
import json
import multiprocessing
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import features
from .sfz import load_reference_map

# v2: longer analysis windows for bass, peak quality gate in the partial
# tracker. Not comparable with v1 numbers.
# v3: partial_freq only counts reference partials above -60 dB rel the
# fundamental (below that, treble-layer "partials" are mic noise and
# their absence in the synth is not a defect).
SCORER_VERSION = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_REPO = REPO_ROOT / "assets" / "reference" / "SplendidGrandPiano"
RENDER_BIN = REPO_ROOT / "target" / "release" / "piano-render"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

# --quick subset: a spread of notes across the keyboard, FF + PP layers.
QUICK_NOTES = [23, 36, 48, 60, 72, 84, 96]
QUICK_LAYERS = {"PP", "FF"}


def render_note(synth, note, velocity, out_path):
    subprocess.run(
        [
            str(RENDER_BIN), "note", str(note), str(velocity),
            "--synth", synth, "--dur", "6", "-o", str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def score_one(task):
    synth, sample, tmpdir = task
    wav = Path(tmpdir) / f"{sample.note}_{sample.layer}.wav"
    render_note(synth, sample.note, sample.velocity, wav)
    ref_audio = features.load_mono(sample.path)
    syn_audio = features.load_mono(wav)
    total, comps = features.compare_pair(
        ref_audio, syn_audio, features.midi_note_freq(sample.note)
    )
    return {
        "note": sample.note,
        "layer": sample.layer,
        "velocity": sample.velocity,
        "loss": total,
        "components": comps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", required=True)
    parser.add_argument("--quick", action="store_true",
                        help="score a small note/layer subset for fast iteration")
    parser.add_argument("--jobs", type=int, default=multiprocessing.cpu_count() // 2)
    parser.add_argument("--label", default="", help="free-form experiment note")
    args = parser.parse_args()

    samples = load_reference_map(REFERENCE_REPO)
    if args.quick:
        wanted = set()
        for note in QUICK_NOTES:
            for layer in QUICK_LAYERS:
                in_layer = [s for s in samples if s.layer == layer]
                wanted.add(min(in_layer, key=lambda s: abs(s.note - note)))
        samples = sorted(wanted, key=lambda s: (s.note, s.layer))

    start = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [(args.synth, s, tmpdir) for s in samples]
        with multiprocessing.Pool(args.jobs) as pool:
            results = pool.map(score_one, tasks)

    losses = np.array([r["loss"] for r in results])
    component_means = {
        key: float(np.mean([r["components"][key] for r in results]))
        for key in results[0]["components"]
    }
    by_register = {}
    for name, lo, hi in [("bass", 21, 47), ("mid", 48, 71), ("treble", 72, 108)]:
        sel = [r["loss"] for r in results if lo <= r["note"] <= hi]
        if sel:
            by_register[name] = float(np.mean(sel))

    git_rev = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout.strip() or "unknown"

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scorer_version": SCORER_VERSION,
        "synth": args.synth,
        "label": args.label,
        "git_rev": git_rev,
        "quick": args.quick,
        "n_samples": len(results),
        "loss_mean": float(losses.mean()),
        "loss_median": float(np.median(losses)),
        "loss_worst": float(losses.max()),
        "components_mean": component_means,
        "by_register": by_register,
        "elapsed_s": round(time.time() - start, 1),
        "pairs": results,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = run["timestamp"].replace(":", "").replace("-", "").replace("+0000", "Z")
    out = RUNS_DIR / f"{stamp}_{args.synth}{'_quick' if args.quick else ''}.json"
    out.write_text(json.dumps(run, indent=1))

    print(f"\nsynth={args.synth}  scorer=v{SCORER_VERSION}  "
          f"{len(results)} pairs in {run['elapsed_s']}s")
    print(f"loss  mean={run['loss_mean']:.3f}  median={run['loss_median']:.3f}  "
          f"worst={run['loss_worst']:.3f}")
    print("components: " + "  ".join(f"{k}={v:.3f}" for k, v in component_means.items()))
    print("registers:  " + "  ".join(f"{k}={v:.3f}" for k, v in by_register.items()))
    print(f"run record: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
