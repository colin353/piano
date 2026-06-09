"""Coordinate-descent hill-climb over the synth's scalar knobs.

The synth reads PIANO_* environment variables (see Knobs in modal2.rs);
renders are deterministic, so the quick score is a noiseless objective.

Usage (from repo root):
    cd scorer && uv run piano-optimize [--synth modal-v2] [--rounds 2]
"""

import argparse
import os
import re
import subprocess

KNOBS = {
    # name: (default, multiplicative step, hard bounds)
    "PIANO_DETUNE_SCALE": (1.35, 1.35, (0.3, 3.0)),
    "PIANO_ATTACK_SCALE": (1.0, 1.3, (0.4, 2.5)),
    "PIANO_BED_GAIN": (1.11, 1.35, (0.4, 4.0)),
    "PIANO_NOISE_GAIN": (1.0, 1.35, (0.2, 3.0)),
    "PIANO_SPLIT_MAX": (0.7, 1.2, (0.3, 0.95)),
    "PIANO_FAST_SCALE": (1.0, 1.25, (0.5, 2.0)),
}


def evaluate(synth, values):
    env = os.environ | {k: f"{v:.4f}" for k, v in values.items()}
    out = subprocess.run(
        ["uv", "run", "piano-score", "--synth", synth, "--quick", "--jobs", "12"],
        capture_output=True, text=True, env=env,
    ).stdout
    m = re.search(r"loss\s+mean=([\d.]+)", out)
    if not m:
        raise RuntimeError(f"no score in output:\n{out}")
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", default="modal-v2")
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    values = {k: spec[0] for k, spec in KNOBS.items()}
    best = evaluate(args.synth, values)
    print(f"start: {best:.4f}  {values}")

    for round_idx in range(args.rounds):
        improved = False
        for knob, (_, step, (lo, hi)) in KNOBS.items():
            for factor in (1.0 / step, step):
                trial = dict(values)
                trial[knob] = min(hi, max(lo, values[knob] * factor))
                if trial[knob] == values[knob]:
                    continue
                loss = evaluate(args.synth, trial)
                marker = ""
                if loss < best - 1e-4:
                    best, values, improved = loss, trial, True
                    marker = "  <-- accepted"
                print(f"  {knob}={trial[knob]:.3f}: {loss:.4f}{marker}")
        print(f"round {round_idx + 1}: best {best:.4f}  {values}")
        if not improved:
            break

    print("\nfinal knobs:")
    for k, v in values.items():
        print(f"  {k}={v:.3f}")
    print(f"final quick loss: {best:.4f}")


if __name__ == "__main__":
    main()
