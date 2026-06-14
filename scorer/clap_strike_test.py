"""Experiment: does CLAP (48kHz) detect the snare-like attack strike that
VGGish (16kHz) is too low-bandwidth to see? Compares content-matched real
Gould audio against our synth rendering the same movements, with the
attack noise ON (PIANO_STRIKE_NOISE=1) vs OFF (=0)."""
import sys, functools, json, subprocess, tempfile, os
from pathlib import Path
import numpy as np, soundfile as sf, librosa, torch

REF = "/tmp/refs/ref1.wav"
OFFS = json.load(open("../data/gould_offsets.json"))
MDIR = Path("../assets/midi/gould1981")
WIN, HOP = 8.0, 4.0


def render(stem, sr_out, strike_noise):
    env = os.environ | {"PIANO_STRIKE_NOISE": str(strike_noise)}
    wav = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(["../target/release/piano-render", "midi", str(MDIR / f"{stem}.mid"),
                    "--synth", "modal-v2", "-o", str(wav)], check=True,
                   capture_output=True, env=env)
    y, sr = sf.read(wav, always_2d=True); wav.unlink()
    m = y.mean(axis=1)
    return librosa.resample(m, orig_sr=sr, target_sr=sr_out) if sr != sr_out else m


def windows(sig, sr, outdir, prefix):
    outdir.mkdir(parents=True, exist_ok=True)
    w, h = int(WIN * sr), int(HOP * sr)
    n = 0
    for i in range(0, max(1, len(sig) - w), h):
        c = sig[i:i + w]
        if len(c) < w or 20 * np.log10(np.sqrt(np.mean(c**2)) + 1e-9) < -45:
            continue
        sf.write(outdir / f"{prefix}_{n:04d}.wav", c, sr); n += 1
    return n


def build_real(sr, root):
    full, fsr = sf.read(REF, always_2d=True); full = full.mean(axis=1)
    a = librosa.resample(full, orig_sr=fsr, target_sr=sr) if fsr != sr else full
    d = root / "real"; [f.unlink() for f in d.glob("*.wav")] if d.exists() else None
    n = 0
    for stem, info in OFFS.items():
        off, dur = int(info["off"] * sr), int(info["dur"] * sr)
        n += windows(a[off:off + dur], sr, d, stem[:8] + f"_{n}")
    return d, n


def build_synth(sr, root, strike, tag):
    d = root / f"synth_{tag}"; [f.unlink() for f in d.glob("*.wav")] if d.exists() else None
    n = 0
    for stem in OFFS:
        n += windows(render(stem, sr, strike), sr, d, stem[:8] + f"_{n}")
    return d, n


def fad_model(name):
    torch.load = functools.partial(torch.load, weights_only=False)
    _lsd = torch.nn.Module.load_state_dict
    torch.nn.Module.load_state_dict = lambda self, sd, strict=True, assign=False: _lsd(
        self, sd, strict=False, assign=assign)
    argv = sys.argv; sys.argv = sys.argv[:1]
    from frechet_audio_distance import FrechetAudioDistance
    sys.argv = argv
    if name == "clap":
        return FrechetAudioDistance(model_name="clap", sample_rate=48000,
                                    submodel_name="music_audioset", verbose=False)
    return FrechetAudioDistance(model_name="vggish", sample_rate=16000,
                                use_pca=False, use_activation=False, verbose=False)


def run(model, sr):
    root = Path(f"../out/fad/{model}")
    real, nr = build_real(sr, root)
    son, n1 = build_synth(sr, root, 1.0, "strike_on")
    soff, n0 = build_synth(sr, root, 0.0, "strike_off")
    print(f"[{model}] corpus: real {nr}, strike_on {n1}, strike_off {n0} windows", flush=True)
    fad = fad_model(model)
    on = fad.score(str(real), str(son))
    off = fad.score(str(real), str(soff))
    print(f"\n=== {model.upper()} (sr {sr}) ===")
    print(f"  real vs synth STRIKE ON  : {on:.4f}")
    print(f"  real vs synth STRIKE OFF : {off:.4f}")
    print(f"  delta (on-off)           : {on-off:+.4f}  "
          f"({'OFF better' if off < on else 'ON better'})")
    return on, off


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("vggish", "both"):
        run("vggish", 16000)
    if which in ("clap", "both"):
        run("clap", 48000)
