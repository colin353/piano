"""Sweep master-EQ params to minimize content-matched VGGish FAD against
the located Gould spans. Renders the 9 movements with PIANO_EQ_* env,
windows at 16k, scores vs the matched real corpus."""
import sys, os, json, subprocess, tempfile, functools
from pathlib import Path
import numpy as np, soundfile as sf, librosa, torch

SR = 16000; WIN, HOP = 8.0, 4.0
OFFS = json.load(open("../data/gould_offsets.json"))
MDIR = Path("../assets/midi/gould1981")
WORK = Path(tempfile.mkdtemp())


def windows(sig, outdir, pfx):
    outdir.mkdir(parents=True, exist_ok=True)
    w, h = int(WIN*SR), int(HOP*SR); n = 0
    for i in range(0, max(1, len(sig)-w), h):
        c = sig[i:i+w]
        if len(c) == w and 20*np.log10(np.sqrt(np.mean(c**2))+1e-9) > -45:
            sf.write(outdir/f"{pfx}_{n:04d}.wav", c, SR); n += 1
    return n


def build_real():
    d = WORK/"real"
    full, fsr = sf.read("/tmp/refs/ref1.wav", always_2d=True); full = full.mean(axis=1)
    a = librosa.resample(full, orig_sr=fsr, target_sr=SR)
    n = 0
    for stem, info in OFFS.items():
        off, dur = int(info["off"]*SR), int(info["dur"]*SR)
        n += windows(a[off:off+dur], d, stem[:6]+str(n))
    return d, n


def render_synth(eq_env):
    d = WORK/"synth"; [f.unlink() for f in d.glob("*.wav")] if d.exists() else None
    env = os.environ | {k: f"{v}" for k, v in eq_env.items()}
    n = 0
    for stem in OFFS:
        w = WORK/"r.wav"
        subprocess.run(["../target/release/piano-render", "midi", str(MDIR/f"{stem}.mid"),
                        "--synth", "modal-v2", "-o", str(w)], check=True,
                       capture_output=True, env=env)
        y, sr = sf.read(w, always_2d=True)
        m = librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=SR)
        n += windows(m, d, stem[:6]+str(n))
    return d, n


def main():
    torch.load = functools.partial(torch.load, weights_only=False)
    argv = sys.argv; sys.argv = sys.argv[:1]
    from frechet_audio_distance import FrechetAudioDistance
    sys.argv = argv
    fad = FrechetAudioDistance(model_name="vggish", sample_rate=SR,
                               use_pca=False, use_activation=False, verbose=False)
    real, nr = build_real()
    print(f"real corpus {nr} windows", flush=True)

    def score(eq):
        synth, _ = render_synth(eq)
        return fad.score(str(real), str(synth))

    # 3 swept gains; fixed freqs/Q from the measured gap.
    base = {"PIANO_EQ_LOW_HZ": 300, "PIANO_EQ_PEAK_HZ": 5000, "PIANO_EQ_PEAK_Q": 0.8,
            "PIANO_EQ_HIGH_HZ": 2500}
    knobs = {"PIANO_EQ_LOW_DB": (-3.0, 2.0, (-9, 2)),
             "PIANO_EQ_PEAK_DB": (6.0, 2.0, (0, 16)),
             "PIANO_EQ_HIGH_DB": (7.0, 2.0, (0, 16))}
    vals = {k: v[0] for k, v in knobs.items()}
    best = score(base | vals)
    print(f"start {best:.4f}  {vals}", flush=True)
    for rnd in range(3):
        improved = False
        for k, (_, step, (lo, hi)) in knobs.items():
            for d in (-step, step):
                t = dict(vals); t[k] = float(np.clip(t[k]+d, lo, hi))
                if t[k] == vals[k]:
                    continue
                s = score(base | t)
                mark = ""
                if s < best - 1e-3:
                    best, vals, improved = s, t, True; mark = " <-- accept"
                print(f"  {k}={t[k]:+.1f}: {s:.4f}{mark}", flush=True)
        print(f"round {rnd+1}: best {best:.4f} {vals}", flush=True)
        if not improved:
            break
    print("FINAL", best, vals, flush=True)


if __name__ == "__main__":
    main()
