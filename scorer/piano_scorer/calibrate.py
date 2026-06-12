"""Fit per-note synth parameters from the reference samples.

For every (note, dynamic layer) sample we measure: actual tuning (the
piano is stretch-tuned — Railsback curve), inharmonicity B, and per-partial
amplitude/decay profiles. The output table (data/calibration.json) is
embedded in the Rust synth, which interpolates across note and velocity.

Usage (from repo root):
    cd scorer && uv run piano-calibrate
"""

import json
import multiprocessing
from pathlib import Path

import numpy as np

from . import features
from .score import REFERENCE_REPO, REPO_ROOT
from .sfz import load_reference_map

# v2: two-stage (prompt/aftersound) decay fit per partial, steeper
# amplitude extrapolation beyond the last measured partial.
# v3: broadband resonant bed (soundboard/duplex/sympathetic ring) level
# and spectral centroid, measured from the late part of each sample.
# v4: per-note loudness relative to the layer median, from raw (un-
# normalized) sample RMS. The scorer normalizes loudness per pair, so
# nothing else constrains keyboard loudness balance — without this the
# mid-range (many comparable partials + loud bed) plays far too loud.
# v5: amp tail slope clamped to [-8, -2.5] dB/partial (treble spectra fall
# steeply for n=1-3 then plateau; extending the initial slope killed all
# high partials), and every layer uses the FF-measured bed (PP beds are
# inflated by mic noise — quiet source, fixed noise floor).
# v6: layer_gains_db — each layer's median raw loudness relative to FF,
# i.e. the measured dynamics curve of the real piano. Replaces the
# synth's ad-hoc vel^1.6 loudness map.
# v8: N_SLOTS 60 -> 150 (bass notes have audible partials to ~5 kHz;
# 60 slots capped A0 at 1.65 kHz).
# v10: defective-sample repair. Several library samples (G#3, C4, G4 in
# every layer) have abnormally fast-dying fundamentals — partials 2-4
# end up 10-22 dB ABOVE the fundamental where neighbors sit 15 dB below.
# The synth faithfully reproduced the defect (and the pair-metric cannot
# object, since synth matches reference). Notes whose spectral balance
# (mean of partials 2-4 rel fundamental) deviates > +10 dB from the
# neighborhood median at audible levels are rebuilt by interpolating
# per-partial data from the nearest clean neighbors.
# v9: slow_split median-filtered across neighboring notes per partial —
# the raw fits are bimodally noisy (0.05 vs 0.7 on adjacent notes), so
# interpolation swept through 0.5 where a two-string unison cancels
# completely (the mid-keyboard 'throbbing null' defect).
# v7: attack_ms — onset-to-envelope-peak rise time. Instant-on partials
# sound plucked; real hammered notes swell over 10-50 ms.
CALIBRATION_VERSION = 10

# The bed decays slowly; this assumed rate back-projects the late
# measurement to t=0 and is what the synth plays it back with.
BED_DECAY_DB_S = 4.0
N_SLOTS = 150  # fixed-length partial arrays; synth clips at Nyquist anyway

AMP_DB_FLOOR = -80.0
DECAY_MIN, DECAY_MAX = 0.3, 300.0  # dB/s


def _fill(values, valid, slope_window=6, clamp=None, max_tail_slope=None,
          min_tail_slope=None):
    """Densify a partial-indexed series: interpolate interior gaps and
    extrapolate the tail with the mean slope of the last valid points.
    `max_tail_slope` caps the extrapolation slope (e.g. forces amplitude to
    keep falling past the last measured partial — extrapolating a flat tail
    produced strong phantom partials up to Nyquist, audible as metallic
    treble)."""
    out = np.array(values, dtype=float)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return np.zeros(N_SLOTS)
    interior = np.interp(np.arange(len(out)), idx, out[idx])
    out = interior
    # Extrapolate beyond the last valid index out to N_SLOTS.
    tail = np.empty(N_SLOTS)
    tail[: len(out)] = out
    last = idx[-1]
    lo = max(0, last - slope_window)
    slope = (out[last] - out[lo]) / max(1, last - lo)
    if max_tail_slope is not None:
        slope = min(slope, max_tail_slope)
    if min_tail_slope is not None:
        slope = max(slope, min_tail_slope)
    for n in range(len(out), N_SLOTS):
        tail[n] = out[last] + slope * (n - last)
    tail[: idx[0]] = out[idx[0]]
    if clamp:
        tail = np.clip(tail, *clamp)
    return tail


def _bed(audio, sr, partial_freqs):
    """Level (dB rel. the early sound, projected back to t=0) and spectral
    centroid of the late broadband bed. Partial bands are notched out of
    the late spectrum first — for bass/mid notes the partials are still
    loud at the measurement time and would otherwise be counted as bed."""
    early_rms = np.sqrt(np.mean(audio[: int(0.2 * sr)] ** 2))
    t = min(2.0, len(audio) / sr - 0.45)
    if t < 0.8 or early_rms <= 0:
        return -45.0, 800.0
    late = audio[int(t * sr): int((t + 0.4) * sr)]
    spec = np.fft.rfft(late * np.hanning(len(late)))
    freqs = np.fft.rfftfreq(len(late), 1 / sr)
    keep = np.ones(len(freqs), bool)
    for f in partial_freqs:
        if np.isfinite(f):
            keep &= np.abs(freqs - f) > max(0.02 * f, 15.0)
    mag = np.abs(spec) * keep
    # Parseval with the Hann window's power correction (~sqrt(3/8)).
    residual_rms = np.sqrt(np.sum(mag**2) / len(late) ** 2 * 2 / 0.375)
    level = 20 * np.log10(residual_rms / early_rms + 1e-9)
    level_t0 = float(np.clip(level + BED_DECAY_DB_S * t, -55.0, -18.0))
    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))
    return level_t0, float(np.clip(centroid, 200.0, 4000.0))


def _raw_loudness_db(raw, sr):
    """Early RMS of the un-normalized sample, dB. Onset-trimmed the same
    way features.prepare does."""
    peak = np.max(np.abs(raw))
    if peak <= 0:
        return -60.0
    above = np.flatnonzero(np.abs(raw) > 0.02 * peak)
    start = max(0, above[0] - 64) if len(above) else 0
    rms = np.sqrt(np.mean(raw[start: start + int(0.4 * sr)] ** 2))
    return float(20 * np.log10(rms + 1e-9))


def _attack_ms(audio, sr):
    """Rise time from onset to the envelope peak, in ms."""
    head = audio[: int(0.25 * sr)]
    frame = int(0.002 * sr)
    n = len(head) // frame
    rms = np.sqrt(np.mean(head[: n * frame].reshape(n, frame) ** 2, axis=1))
    return float(np.clip(np.argmax(rms) * 2.0, 2.0, 80.0))


def calibrate_one(sample):
    raw = features.load_mono(sample.path)
    loudness_db = _raw_loudness_db(raw, features.SR)
    audio = features.prepare(raw)
    nominal = features.midi_note_freq(sample.note)
    n_partials = int(np.clip(18000 / nominal, 5, N_SLOTS))
    p = features.extract_partials(audio, nominal, n_partials=n_partials)

    valid = np.isfinite(p.freqs)
    amps = _fill(np.where(valid, p.amps_db, 0), valid,
                 clamp=(AMP_DB_FLOOR, 20), max_tail_slope=-2.5,
                 min_tail_slope=-8.0)

    def fill_decay(series):
        ok = valid & np.isfinite(series)
        return _fill(np.where(ok, series, 0), ok, clamp=(DECAY_MIN, DECAY_MAX))

    decays_fast = fill_decay(p.decays_fast)
    decays_slow = np.minimum(fill_decay(p.decays_slow), decays_fast)
    split_ok = valid & np.isfinite(p.slow_split)
    splits = _fill(np.where(split_ok, p.slow_split, 0), split_ok,
                   clamp=(0.02, 0.7))
    f0_cents = float(np.clip(1200 * np.log2(p.f0 / nominal), -60, 60))
    bed_db, bed_centroid = _bed(audio, features.SR, p.freqs)
    return {
        "note": sample.note,
        "layer": sample.layer,
        "velocity": sample.velocity,
        "f0_cents": round(f0_cents, 2),
        "b": float(np.clip(p.inharmonicity, 1e-6, 2e-2)),
        "bed_db": round(bed_db, 1),
        "bed_centroid_hz": round(bed_centroid, 0),
        "loudness_db": round(loudness_db, 2),  # made layer-relative below
        "attack_ms": round(_attack_ms(audio, features.SR), 1),
        "amps_db": [round(float(v), 2) for v in amps],
        "decays_fast_db_s": [round(float(v), 2) for v in decays_fast],
        "decays_slow_db_s": [round(float(v), 2) for v in decays_slow],
        "slow_split": [round(float(v), 3) for v in splits],
    }


def main():
    samples = load_reference_map(REFERENCE_REPO)
    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
        rows = pool.map(calibrate_one, samples)

    # Median-filter f0 and B across neighboring notes within each layer to
    # suppress single-sample tracking errors (B varies smoothly in reality).
    layers = {}
    for row in rows:
        layers.setdefault(row["layer"], []).append(row)
    for layer_rows in layers.values():
        layer_rows.sort(key=lambda r: r["note"])
        # Collapse duplicate keycenters (some appear in two region groups).
        seen = {}
        for r in layer_rows:
            seen[r["note"]] = r
        layer_rows[:] = [seen[n] for n in sorted(seen)]
        for key in ("f0_cents", "b"):
            vals = np.array([r[key] for r in layer_rows])
            smooth = vals.copy()
            for i in range(len(vals)):
                lo, hi = max(0, i - 1), min(len(vals), i + 2)
                smooth[i] = np.median(vals[lo:hi])
            for r, v in zip(layer_rows, smooth):
                r[key] = round(float(v), 6 if key == "b" else 2)
        # Loudness: relative to the layer median (absolute recording gain
        # is arbitrary; the keyboard balance is what matters), lightly
        # smoothed across neighbors.
        loud = np.array([r["loudness_db"] for r in layer_rows])
        layer_rows_median = float(np.median(loud))
        for r in layer_rows:
            r["_layer_median"] = layer_rows_median
        loud -= layer_rows_median
        smooth = loud.copy()
        for i in range(len(loud)):
            lo, hi = max(0, i - 1), min(len(loud), i + 2)
            smooth[i] = np.median(loud[lo:hi])
        for r, v in zip(layer_rows, smooth):
            r["loudness_db"] = round(float(np.clip(v, -12, 12)), 2)
    # slow_split: median-filter per partial index across neighboring notes.
    for layer_rows in layers.values():
        for idx in range(N_SLOTS):
            vals = np.array([r["slow_split"][idx] for r in layer_rows])
            sm = vals.copy()
            for i in range(len(vals)):
                lo, hi = max(0, i - 1), min(len(vals), i + 2)
                sm[i] = np.median(vals[lo:hi])
            for r, v in zip(layer_rows, sm):
                r["slow_split"][idx] = round(float(v), 3)
    # Defective-sample repair (see v10 note above).
    ARRAY_KEYS = ("amps_db", "decays_fast_db_s", "decays_slow_db_s", "slow_split")
    for layer_name, layer_rows in layers.items():
        balances = np.array([np.mean(r["amps_db"][1:4]) for r in layer_rows])
        flagged = []
        for i in range(len(layer_rows)):
            lo, hi = max(0, i - 2), min(len(layer_rows), i + 3)
            nb = np.delete(balances[lo:hi], i - lo)
            dev = balances[i] - np.median(nb)
            if dev > 10.0 and balances[i] > -35.0:
                flagged.append(i)
        for i in flagged:
            prev = next((j for j in range(i - 1, -1, -1) if j not in flagged), None)
            nxt = next((j for j in range(i + 1, len(layer_rows)) if j not in flagged), None)
            if prev is None or nxt is None:
                continue
            a, b = layer_rows[prev], layer_rows[nxt]
            w = (layer_rows[i]["note"] - a["note"]) / (b["note"] - a["note"])
            for key in ARRAY_KEYS:
                layer_rows[i][key] = [
                    round(float(x + w * (y - x)), 3)
                    for x, y in zip(a[key], b[key])
                ]
            for key in ("attack_ms", "bed_db"):
                layer_rows[i][key] = round(
                    a[key] + w * (b[key] - a[key]), 1)
            print(f"  repaired {layer_name} note {layer_rows[i]['note']} "
                  f"(hollow fundamental, dev {balances[i] - np.median(np.delete(balances[max(0,i-2):i+3], i-max(0,i-2))):+.1f} dB)")
    # Bed level/centroid: trust only the FF measurements (loudest source,
    # best SNR); the bed scales with note loudness across layers anyway.
    ff_rows = sorted(layers["FF"], key=lambda r: r["note"])
    for layer_rows in layers.values():
        for r in layer_rows:
            nearest = min(ff_rows, key=lambda f: abs(f["note"] - r["note"]))
            r["bed_db"] = nearest["bed_db"]
            r["bed_centroid_hz"] = nearest["bed_centroid_hz"]

    ff_median = layers["FF"][0]["_layer_median"]
    layer_gains = {
        name: round(rows_[0]["_layer_median"] - ff_median, 2)
        for name, rows_ in layers.items()
    }
    out = {
        "version": CALIBRATION_VERSION,
        "n_slots": N_SLOTS,
        # Median raw loudness of each layer relative to FF: the measured
        # velocity->dynamics curve of the reference piano.
        "layer_gains_db": layer_gains,
        "layers": {
            name: [{k: v for k, v in r.items() if k not in ("layer", "_layer_median")}
                   for r in rows_]
            for name, rows_ in sorted(layers.items())
        },
    }
    dest = REPO_ROOT / "data" / "calibration.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out))
    size_kb = dest.stat().st_size // 1024
    print(f"wrote {dest.relative_to(REPO_ROOT)} ({size_kb} kB, "
          f"{sum(len(v) for v in layers.values())} fitted samples)")

    # Quick sanity printout: tuning stretch and B across the keyboard (FF).
    ff = out["layers"]["FF"]
    for r in ff[:: max(1, len(ff) // 10)]:
        print(f"  note {r['note']:3d}  f0 {r['f0_cents']:+6.1f}c  "
              f"B {r['b']:.2e}  decay1 {r['decays_fast_db_s'][0]:5.1f}"
              f"/{r['decays_slow_db_s'][0]:5.1f} dB/s  "
              f"split {r['slow_split'][0]:.2f}  amp2 {r['amps_db'][1]:+6.1f} dB")


if __name__ == "__main__":
    main()
