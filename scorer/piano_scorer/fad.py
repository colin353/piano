"""Piece-level realism metric: Fréchet Audio Distance against real piano
recordings.

The single-note scorer is precise about timbre but blind to everything
between notes (releases, transitions, pedal, room). FAD compares the
*distribution* of VGGish embeddings of our rendered music against real
recordings — no pairing or aligned MIDI required.

Interpretation needs calibration: the number includes content/production
mismatch, so we report a floor (one half of the real corpus vs the other)
and a ceiling (the sine baseline synth) alongside the synth's score.
The primary corpus is the user's Gould 1981 Goldberg recording, which is
content-matched to our Goldberg renders.

Usage (from repo root):
    cd scorer && uv run piano-fad --synth modal-v2
Corpus prep (once, needs /tmp/refs/ref1.wav from yt-dlp):
    uv run piano-fad --prepare-real /tmp/refs/ref1.wav
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from .score import REPO_ROOT, RUNS_DIR

FAD_SR = 16000
CHUNK_S = 30
CORPUS = REPO_ROOT / "out" / "fad"
GOLDBERG_MIDIS = ["goldberg-aria", "goldberg-v01", "goldberg-v05", "goldberg-v13"]
# Performed MIDI (ATEPP transcriptions of Gould's own Goldberg recordings):
# content- AND performance-matched to the real corpus.
PERFORMED_DIR = REPO_ROOT / "assets" / "midi" / "gould1981"
FAD_VERSION = 1


def _write_chunks(audio, out_dir, prefix, start_index=0):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    step = CHUNK_S * FAD_SR
    for i in range(0, len(audio) - step, step):
        chunk = audio[i: i + step]
        rms_db = 20 * np.log10(np.sqrt(np.mean(chunk**2)) + 1e-9)
        if rms_db < -45:
            continue  # silence / gaps
        sf.write(out_dir / f"{prefix}_{start_index + n:04d}.wav", chunk, FAD_SR)
        n += 1
    return n


def _load_16k_mono(path):
    import librosa
    y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != FAD_SR:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=FAD_SR)
    return y


def prepare_real(source_wav):
    """Chunk a real recording into the two real-corpus halves (the halves
    give the content-controlled floor measurement)."""
    y = _load_16k_mono(source_wav)
    # Drop the first/last minute (applause, announcements).
    y = y[60 * FAD_SR: -60 * FAD_SR]
    half = len(y) // 2
    a = _write_chunks(y[:half], CORPUS / "real_a", "real")
    b = _write_chunks(y[half:], CORPUS / "real_b", "real")
    print(f"real corpus: {a} + {b} chunks of {CHUNK_S}s at {FAD_SR} Hz")


def render_synth_corpus(synth, dry=False, performed=False):
    tag = ("_perf" if performed else "") + ("_dry" if dry else "")
    out_dir = CORPUS / f"synth_{synth}{tag}"
    if out_dir.exists():
        for f in out_dir.glob("*.wav"):
            f.unlink()
    if performed:
        midis = sorted(PERFORMED_DIR.glob("*.mid"))
        # Performed MIDI needs no --legato (real overlaps are in the data).
        extra = []
    else:
        midis = [REPO_ROOT / "assets/midi" / f"{n}.mid" for n in GOLDBERG_MIDIS]
        extra = ["--legato"]
    total = 0
    for path in midis:
        tmp = CORPUS / f"_{path.stem}.wav"
        subprocess.run(
            [str(REPO_ROOT / "target/release/piano-render"), "midi", str(path),
             "--synth", synth, "-o", str(tmp)]
            + extra + (["--dry"] if dry else []),
            check=True, capture_output=True,
        )
        total += _write_chunks(_load_16k_mono(tmp), out_dir, path.stem, total)
        tmp.unlink()
    return out_dir, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth")
    parser.add_argument("--dry", action="store_true", help="render without the room")
    parser.add_argument("--performed", action="store_true",
                        help="use the ATEPP Gould performance MIDI corpus")
    parser.add_argument("--prepare-real", metavar="WAV")
    args = parser.parse_args()

    if args.prepare_real:
        prepare_real(args.prepare_real)
        return
    if not args.synth:
        parser.error("--synth required (or --prepare-real)")

    real_a, real_b = CORPUS / "real_a", CORPUS / "real_b"
    if not real_a.exists():
        raise SystemExit("real corpus missing — run --prepare-real first")

    synth_dir, n = render_synth_corpus(args.synth, dry=args.dry,
                                       performed=args.performed)
    print(f"synth corpus: {n} chunks")

    # laion_clap (a transitive dep) runs an argparse at import time and
    # chokes on our CLI args; hide argv during the import.
    import sys
    argv, sys.argv = sys.argv, sys.argv[:1]
    from frechet_audio_distance import FrechetAudioDistance
    sys.argv = argv
    fad = FrechetAudioDistance(model_name="vggish", sample_rate=FAD_SR,
                               use_pca=False, use_activation=False, verbose=False)
    start = time.time()
    floor = fad.score(str(real_a), str(real_b))
    score = fad.score(str(real_a), str(synth_dir))
    print(f"\nFAD vs Gould Goldberg (VGGish):")
    print(f"  real-vs-real floor : {floor:.3f}")
    print(f"  {args.synth:18s} : {score:.3f}")

    run = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fad_version": FAD_VERSION,
        "synth": args.synth + ("_perf" if args.performed else "")
        + ("_dry" if args.dry else ""),
        "floor_real_vs_real": floor,
        "fad": score,
        "elapsed_s": round(time.time() - start, 1),
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = run["timestamp"].replace(":", "").replace("-", "")
    out = RUNS_DIR / f"{stamp}_fad_{args.synth}.json"
    out.write_text(json.dumps(run, indent=1))
    print(f"run record: {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
