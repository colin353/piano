"""Render side-by-side analysis plots of synth vs. reference for one note.

Usage (from repo root):
    cd scorer && uv run piano-plot --synth baseline --note 60 --layer FF -o /tmp/c4.png

Produces spectrograms, energy envelopes, and the early spectrum with
tracked partials — the visual diagnostics for assessing a synth by eye.
"""

import argparse
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import features
from .score import REFERENCE_REPO, render_note
from .sfz import load_reference_map


def spectrogram_db(y, sr):
    s = np.abs(librosa.stft(y, n_fft=4096, hop_length=1024))
    return librosa.amplitude_to_db(s, ref=np.max)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", required=True)
    parser.add_argument("--note", type=int, required=True)
    parser.add_argument("--layer", default="FF")
    parser.add_argument("-o", "--out", required=True)
    args = parser.parse_args()

    samples = load_reference_map(REFERENCE_REPO)
    in_layer = [s for s in samples if s.layer == args.layer]
    sample = min(in_layer, key=lambda s: abs(s.note - args.note))
    f0 = features.midi_note_freq(sample.note)

    with tempfile.TemporaryDirectory() as tmpdir:
        wav = Path(tmpdir) / "syn.wav"
        render_note(args.synth, sample.note, sample.velocity, wav)
        syn = features.prepare(features.load_mono(wav))
    ref = features.prepare(features.load_mono(sample.path))

    sr = features.SR
    n = int(features.SCORE_SECONDS * sr)
    ref, syn = ref[:n], syn[:n]
    total, comps = features.compare_pair(ref, syn, f0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fmax = min(f0 * 24, sr / 2)
    for col, (y, name) in enumerate([(ref, "reference"), (syn, args.synth)]):
        db = spectrogram_db(y, sr)
        librosa.display.specshow(
            db, sr=sr, hop_length=1024, x_axis="time", y_axis="hz",
            ax=axes[0][col], cmap="magma", vmin=-80, vmax=0,
        )
        axes[0][col].set_ylim(0, fmax)
        axes[0][col].set_title(f"{name} — note {sample.note} {sample.layer}")

        env = features._env_db(y, sr)
        t = librosa.frames_to_time(np.arange(len(env)), sr=sr, hop_length=512)
        axes[1][col].plot(t, env)
        axes[1][col].set_ylim(-70, 10)
        axes[1][col].set_xlabel("s")
        axes[1][col].set_ylabel("RMS dB")
        axes[1][col].set_title(f"{name} envelope")
        axes[1][col].grid(alpha=0.3)

        p = features.extract_partials(y, f0, sr)
        seg = y[int(0.05 * sr): int(1.25 * sr)]
        n_fft = int(2 ** np.ceil(np.log2(len(seg))))
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n_fft))
        fr = np.fft.rfftfreq(n_fft, 1 / sr)
        sel = fr < fmax
        axes[2][col].plot(fr[sel], 20 * np.log10(spec[sel] + 1e-9), lw=0.5)
        for f in p.freqs[np.isfinite(p.freqs)]:
            axes[2][col].axvline(f, color="r", alpha=0.3, lw=0.8)
        axes[2][col].set_title(f"{name} early spectrum (B={p.inharmonicity:.2e})")
        axes[2][col].set_xlabel("Hz")
        axes[2][col].set_ylabel("dB")

    comp_text = "  ".join(f"{k}={v:.2f}" for k, v in comps.items())
    fig.suptitle(f"loss={total:.3f}   {comp_text}", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}  (loss={total:.3f})")


if __name__ == "__main__":
    main()
