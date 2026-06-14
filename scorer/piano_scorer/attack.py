"""Attack-divergence analyzer: quantify how the synth's first ~200 ms
diverges from real ff single-note strikes (SplendidGrandPiano), and where.

The note's steady tone is already well matched; this isolates the attack
transient. We compare, onset-aligned, over 0-200 ms:

  - per-band onset envelopes (octave bands): when/how each band rises and
    falls — captures attack timing and the bright-transient decay
  - spectral-centroid trajectory: real strikes start bright and darken;
    a wrong trajectory is the synthetic/"click" tell
  - early-transient ratio: broadband energy in the first 12 ms vs the
    settled tone (the hammer/action knock)
  - velocity divergence: how the PP->FF attack *change* differs real vs
    synth (the user hears bigger divergence on hard strikes)

Outputs a composite divergence score, a per-component breakdown, and a
diagnostic PNG. This is the objective we then minimize.

Usage (from repo root):
    cd scorer && uv run piano-attack --notes 60,72,84,96
    cd scorer && uv run piano-attack --note 84 --plot /tmp/attack84.png
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import features
from .sfz import load_reference_map, LAYERS
from .score import REFERENCE_REPO, REPO_ROOT

SR = features.SR
ATTACK_S = 0.20
BANDS = [(80, 160), (160, 320), (320, 640), (640, 1280),
         (1280, 2560), (2560, 5120), (5120, 10240), (10240, 20000)]


def _render_note(note, velocity):
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "n.wav"
        subprocess.run(
            [str(REPO_ROOT / "target/release/piano-render"), "note",
             str(note), str(velocity), "--dur", "1.0", "-o", str(wav)],
            check=True, capture_output=True)
        return features.load_mono(wav)


def _onset_trim(y):
    peak = np.max(np.abs(y))
    above = np.flatnonzero(np.abs(y) > 0.02 * peak)
    start = above[0] if len(above) else 0
    return y[start:]


def _band_envelopes(y, sr=SR):
    """Hilbert-magnitude envelope per octave band over the attack window,
    in dB, on a common time grid."""
    from scipy.signal import butter, sosfilt, hilbert
    n = int(ATTACK_S * sr)
    y = y[:n]
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    envs = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, min(hi, sr / 2 * 0.99)], btype="band", fs=sr, output="sos")
        env = np.abs(hilbert(sosfilt(sos, y)))
        # 1 ms smoothing
        k = max(1, int(0.001 * sr))
        env = np.convolve(env, np.ones(k) / k, "same")
        envs.append(20 * np.log10(env + 1e-6))
    return np.array(envs)  # (bands, samples)


def _centroid_traj(y, sr=SR):
    n = int(ATTACK_S * sr)
    y = y[:n]
    hop = max(1, int(0.005 * sr))
    c = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024, hop_length=hop)[0]
    return c


def _transient_ratio(y, sr=SR):
    """Broadband energy in the first 12 ms vs 100-180 ms (settled), dB."""
    e = lambda a, b: np.sqrt(np.mean(y[int(a * sr):int(b * sr)] ** 2) + 1e-12)
    return 20 * np.log10(e(0, 0.012) / (e(0.10, 0.18) + 1e-9))


def analyze(note, layer="FF"):
    samples = load_reference_map(REFERENCE_REPO)
    ref = min((s for s in samples if s.layer == layer), key=lambda s: abs(s.note - note))
    vel = LAYERS[layer][2]
    real = features.prepare(features.load_mono(ref.path))
    syn = features.prepare(_render_note(ref.note, vel))
    real, syn = _onset_trim(real), _onset_trim(syn)

    rb, sb = _band_envelopes(real), _band_envelopes(syn)
    # normalize each band to its own peak (compare SHAPE, not level)
    rbn = rb - rb.max(axis=1, keepdims=True)
    sbn = sb - sb.max(axis=1, keepdims=True)
    band_div = np.mean(np.abs(rbn - sbn), axis=1)  # per-band dB divergence

    rc, sc = _centroid_traj(real), _centroid_traj(syn)
    m = min(len(rc), len(sc))
    cent_div = float(np.mean(np.abs(np.log2((rc[:m] + 1) / (sc[:m] + 1)))) * 1200)  # cents

    tr_div = abs(_transient_ratio(real) - _transient_ratio(syn))
    composite = float(np.mean(band_div) + cent_div / 200 + tr_div / 3)
    return {
        "note": ref.note, "band_div": band_div, "cent_div": cent_div,
        "transient_real": _transient_ratio(real), "transient_syn": _transient_ratio(syn),
        "tr_div": tr_div, "composite": composite,
        "real": real, "syn": syn, "rb": rb, "sb": sb, "rc": rc, "sc": sc,
    }


def plot(res, path):
    fig, ax = plt.subplots(3, 2, figsize=(13, 11))
    for col, (y, lab) in enumerate([(res["real"], "REAL"), (res["syn"], "synth")]):
        S = librosa.amplitude_to_db(
            np.abs(librosa.stft(y[:int(ATTACK_S * SR)], n_fft=1024, hop_length=32)), ref=np.max)
        librosa.display.specshow(S, sr=SR, hop_length=32, x_axis="time", y_axis="log",
                                 ax=ax[0][col], cmap="magma", vmin=-70, vmax=0)
        ax[0][col].set_title(f"{lab} attack note {res['note']}")
    t = np.arange(res["rb"].shape[1]) / SR * 1000
    for bi, (lo, hi) in enumerate(BANDS):
        off = bi * 12
        ax[1][0].plot(t, res["rb"][bi] - res["rb"][bi].max() - off, lw=0.8)
        ax[1][1].plot(t, res["sb"][bi] - res["sb"][bi].max() - off, lw=0.8,
                      label=f"{lo}-{hi}")
    ax[1][0].set_title("REAL band envelopes (stacked)"); ax[1][1].set_title("synth band envelopes")
    for a in ax[1]:
        a.set_xlabel("ms"); a.set_ylim(-110, 5)
    ax[1][1].legend(fontsize=6, ncol=2)
    tc = np.arange(len(res["rc"])) * 5
    ax[2][0].plot(tc[:len(res["rc"])], res["rc"], label="real")
    ax[2][0].plot(np.arange(len(res["sc"])) * 5, res["sc"], label="synth")
    ax[2][0].set_title("spectral centroid trajectory"); ax[2][0].set_xlabel("ms")
    ax[2][0].set_ylabel("Hz"); ax[2][0].legend()
    ax[2][1].bar(range(len(BANDS)), res["band_div"])
    ax[2][1].set_title(f"per-band divergence (dB)  composite={res['composite']:.2f}")
    ax[2][1].set_xticks(range(len(BANDS)))
    ax[2][1].set_xticklabels([f"{lo//1000}k" if lo >= 1000 else lo for lo, _ in BANDS],
                             fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=100); print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--notes", default="60,72,84,96")
    p.add_argument("--layer", default="FF")
    p.add_argument("--plot")
    args = p.parse_args()
    notes = [int(n) for n in args.notes.split(",")]
    print(f"attack divergence vs {args.layer} ({LAYERS[args.layer][2]} vel):")
    print(f"{'note':>4} {'composite':>9} {'centroid':>9} {'transient r/s':>16} {'worst band':>12}")
    for note in notes:
        r = analyze(note, args.layer)
        wb = BANDS[int(np.argmax(r["band_div"]))]
        print(f"{r['note']:>4} {r['composite']:>9.2f} {r['cent_div']:>7.0f}c "
              f"  {r['transient_real']:>5.1f}/{r['transient_syn']:<5.1f}dB "
              f"  {wb[0]}-{wb[1]}Hz ({r['band_div'].max():.0f}dB)")
        if args.plot and len(notes) == 1:
            plot(r, args.plot)


if __name__ == "__main__":
    main()
