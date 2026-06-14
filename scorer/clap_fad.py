"""Direct CLAP (music_audioset, 48kHz) Frechet Audio Distance — bypasses
the frechet-audio-distance package's CLAP path (incompatible with the
installed laion_clap 1.1.5). Hand-rolled embedding + Frechet distance."""
import sys, functools, os, contextlib
from pathlib import Path
import numpy as np, soundfile as sf, torch
from scipy import linalg

CKPT = "/home/colin/.cache/torch/hub/music_audioset_epoch_15_esc_90.14.pt"


def load_clap():
    torch.load = functools.partial(torch.load, weights_only=False)
    _l = torch.nn.Module.load_state_dict
    torch.nn.Module.load_state_dict = lambda s, sd, strict=True, assign=False: _l(
        s, sd, strict=False, assign=assign)
    import laion_clap
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        m.load_ckpt(CKPT)
    m.eval()
    return m


def embed_dir(model, d, batch=24):
    files = sorted(Path(d).glob("*.wav"))
    xs = [sf.read(f, always_2d=True)[0].mean(axis=1).astype("float32") for f in files]
    embs = []
    with torch.no_grad():
        for i in range(0, len(xs), batch):
            chunk = np.stack(xs[i:i + batch])  # equal-length windows
            e = model.get_audio_embedding_from_data(chunk)
            embs.append(np.asarray(e if not torch.is_tensor(e) else e.cpu().numpy()))
    return np.concatenate(embs, 0)


def frechet(a, b):
    mu1, mu2 = a.mean(0), b.mean(0)
    c1 = np.cov(a, rowvar=False) + 1e-6 * np.eye(a.shape[1])
    c2 = np.cov(b, rowvar=False) + 1e-6 * np.eye(b.shape[1])
    covmean = linalg.sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(c1 + c2 - 2 * covmean))


if __name__ == "__main__":
    root = Path("../out/fad/clap")
    m = load_clap()
    real = embed_dir(m, root / "real")
    on = embed_dir(m, root / "synth_strike_on")
    off = embed_dir(m, root / "synth_strike_off")
    # real-vs-real floor: split real in half
    h = len(real) // 2
    print(f"\n=== CLAP music_audioset (48kHz) ===", flush=True)
    print(f"  floor real-vs-real       : {frechet(real[:h], real[h:]):.4f}")
    d_on = frechet(real, on); d_off = frechet(real, off)
    print(f"  real vs synth STRIKE ON  : {d_on:.4f}")
    print(f"  real vs synth STRIKE OFF : {d_off:.4f}")
    print(f"  delta (on-off)           : {d_on-d_off:+.4f}  "
          f"({'OFF better' if d_off < d_on else 'ON better'})")
