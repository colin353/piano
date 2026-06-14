"""Fit master-EQ gains to minimize the spectral (LTAS) gap to the real
Gould corpus over 300 Hz-12 kHz. This is the correct objective for tone
matching (VGGish FAD is insensitive/perverse on this axis). Renders the 9
movements with PIANO_EQ_* env, measures resulting LTAS, coordinate-descends."""
import os, json, subprocess, tempfile
from pathlib import Path
import numpy as np, soundfile as sf, librosa
from scipy.signal import welch

SR = 16000
OFFS = json.load(open("../data/gould_offsets.json"))
MDIR = Path("../assets/midi/gould1981")
WORK = Path(tempfile.mkdtemp())
BANDS = [(300, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 6000), (6000, 8000), (8000, 12000)]


def band_ltas(sig):
    fr, P = welch(sig, fs=SR, nperseg=4096)
    P = P / P.sum()
    return np.array([P[(fr >= lo) & (fr < hi)].sum() for lo, hi in BANDS])


def real_target():
    full, fsr = sf.read("/tmp/refs/ref1.wav", always_2d=True); full = full.mean(axis=1)
    a = librosa.resample(full, orig_sr=fsr, target_sr=SR)
    segs = [a[int(i["off"]*SR):int((i["off"]+i["dur"])*SR)] for i in OFFS.values()]
    return band_ltas(np.concatenate(segs))


def synth_ltas(eq):
    env = os.environ | {k: f"{v}" for k, v in eq.items()}
    segs = []
    for stem in OFFS:
        w = WORK/"r.wav"
        subprocess.run(["../target/release/piano-render", "midi", str(MDIR/f"{stem}.mid"),
                        "--synth", "modal-v2", "-o", str(w)], check=True,
                       capture_output=True, env=env)
        y, sr = sf.read(w, always_2d=True)
        segs.append(librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=SR))
    return band_ltas(np.concatenate(segs))


def main():
    target_db = 10*np.log10(real_target()+1e-20)
    base = {"PIANO_EQ_LOW_HZ": 300, "PIANO_EQ_PEAK_HZ": 5000, "PIANO_EQ_PEAK_Q": 0.8,
            "PIANO_EQ_HIGH_HZ": 2500}

    def cost(eq):
        s_db = 10*np.log10(synth_ltas(base | eq)+1e-20)
        # equal-power renormalize (compare shape), weight presence region
        d = (s_db - s_db.mean()) - (target_db - target_db.mean())
        w = np.array([1, 1, 1.2, 1.5, 1.5, 1.3, 0.8])
        return float(np.sqrt(np.mean((w*d)**2)))

    knobs = {"PIANO_EQ_LOW_DB": (-3.0, 1.5, (-9, 2)),
             "PIANO_EQ_PEAK_DB": (6.0, 1.5, (0, 14)),
             "PIANO_EQ_HIGH_DB": (7.0, 1.5, (0, 14))}
    vals = {k: v[0] for k, v in knobs.items()}
    best = cost(vals)
    print(f"start LTAS-rmse {best:.3f} dB  {vals}", flush=True)
    print(f"  (flat baseline: {cost({k:0 for k in knobs}):.3f} dB)", flush=True)
    for rnd in range(4):
        improved = False
        for k, (_, step, (lo, hi)) in knobs.items():
            for d in (-step, step):
                t = dict(vals); t[k] = float(np.clip(t[k]+d, lo, hi))
                if t[k] == vals[k]:
                    continue
                c = cost(t); mark = ""
                if c < best - 1e-3:
                    best, vals, improved = c, t, True; mark = " <-- accept"
                print(f"  {k}={t[k]:+.1f}: {c:.3f}{mark}", flush=True)
        print(f"round {rnd+1}: best {best:.3f} {vals}", flush=True)
        if not improved:
            break
    print("FINAL", round(best, 3), vals, flush=True)


if __name__ == "__main__":
    main()
