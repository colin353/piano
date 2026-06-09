"""Perceptual feature extraction and distance metrics for piano samples.

All comparisons happen after onset alignment and early-RMS loudness
normalization, so the score measures *timbre and time evolution*, not
absolute gain. Components (each ~0 for identical audio, O(1) for badly
wrong audio):

- mel:      multi-resolution log-mel spectrogram L1 distance
- envelope: log-RMS energy envelope L1 distance (decay shape)
- partials: inharmonicity B, partial frequency deviation, partial
            amplitude profile, and per-partial decay rates
"""

from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import get_window

SR = 44100
SCORE_SECONDS = 5.0
N_PARTIALS = 12


def load_mono(path, sr=SR):
    y, file_sr = sf.read(path, dtype="float64", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def prepare(y, sr=SR):
    """Trim to onset and normalize loudness over the first second."""
    peak = np.max(np.abs(y))
    if peak <= 0:
        return y
    above = np.flatnonzero(np.abs(y) > 0.02 * peak)
    start = max(0, above[0] - 64) if len(above) else 0
    y = y[start:]
    rms = np.sqrt(np.mean(y[: sr] ** 2))
    return y / max(rms, 1e-9)


def _common_trim(a, b, sr=SR):
    n = min(len(a), len(b), int(SCORE_SECONDS * sr))
    return a[:n], b[:n]


# ---------------------------------------------------------------- mel


def mel_distance(a, b, sr=SR):
    a, b = _common_trim(a, b, sr)
    total = 0.0
    resolutions = ((512, 64), (2048, 128), (8192, 128))
    for n_fft, n_mels in resolutions:
        dists = []
        for y in (a, b):
            m = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=n_fft, hop_length=n_fft // 4, n_mels=n_mels
            )
            dists.append(np.log10(m + 1e-7))
        frames = min(dists[0].shape[1], dists[1].shape[1])
        total += np.mean(np.abs(dists[0][:, :frames] - dists[1][:, :frames]))
    return total / len(resolutions)


# ---------------------------------------------------------------- envelope


def _env_db(y, sr=SR, hop=512, frame=2048):
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    return 20 * np.log10(rms + 1e-6)


def envelope_distance(a, b, sr=SR):
    a, b = _common_trim(a, b, sr)
    ea, eb = _env_db(a, sr), _env_db(b, sr)
    n = min(len(ea), len(eb))
    # 1.0 ~= 20 dB mean deviation in decay shape.
    return float(np.mean(np.abs(ea[:n] - eb[:n]))) / 20.0


# ---------------------------------------------------------------- partials


@dataclass
class PartialSet:
    f0: float                 # measured fundamental, Hz
    inharmonicity: float      # B coefficient
    freqs: np.ndarray         # measured partial frequencies, Hz (nan = absent)
    amps_db: np.ndarray       # partial amplitudes rel. partial 1, dB
    decays_db_s: np.ndarray   # per-partial decay rates, dB/s (nan = absent)


def extract_partials(y, nominal_f0, sr=SR, n_partials=N_PARTIALS):
    """Measure partial frequencies/amplitudes from an early-sustain window,
    then per-partial decay rates from a band-limited STFT energy slope."""
    seg = y[int(0.05 * sr): int(1.25 * sr)]
    if len(seg) < sr // 2:
        seg = y[: sr]
    n_fft = int(2 ** np.ceil(np.log2(len(seg))))
    spec = np.abs(np.fft.rfft(seg * get_window("hann", len(seg)), n_fft))
    freq_step = sr / n_fft

    def peak_near(f_lo, f_hi):
        lo, hi = int(f_lo / freq_step), int(f_hi / freq_step)
        if hi <= lo + 2 or hi >= len(spec):
            return np.nan, 0.0
        k = lo + int(np.argmax(spec[lo:hi]))
        if k == 0 or k + 1 >= len(spec):
            return k * freq_step, spec[k]
        # Quadratic interpolation around the bin peak.
        a_, b_, c_ = np.log(spec[k - 1: k + 2] + 1e-12)
        delta = 0.5 * (a_ - c_) / (a_ - 2 * b_ + c_ + 1e-12)
        return (k + np.clip(delta, -0.5, 0.5)) * freq_step, spec[k]

    # Fundamental first, searched widely around nominal.
    f0, a0 = peak_near(nominal_f0 * 0.94, nominal_f0 * 1.06)
    if not np.isfinite(f0) or a0 <= 0:
        f0 = nominal_f0

    # Iteratively track partials with a growing stretch estimate.
    freqs = np.full(n_partials, np.nan)
    amps = np.zeros(n_partials)
    freqs[0], amps[0] = f0, a0
    b_est = 0.0
    for n in range(2, n_partials + 1):
        expect = n * f0 * np.sqrt(1 + b_est * n * n)
        if expect > sr / 2 * 0.95:
            break
        f, a = peak_near(expect * 0.985, expect * 1.03)
        freqs[n - 1], amps[n - 1] = f, a
        good = np.isfinite(freqs[:n])
        ns = np.arange(1, n + 1)[good]
        if len(ns) >= 3:
            # Least squares on (f_n / (n f0))^2 = 1 + B n^2
            ratio2 = (freqs[:n][good] / (ns * f0)) ** 2
            b_est = max(0.0, float(np.polyfit(ns**2, ratio2, 1)[0]))

    amps_db = 20 * np.log10(amps / (amps[0] + 1e-12) + 1e-7)

    # Decay rates: energy slope per partial band over time.
    hop, win = 2048, 8192
    stft = np.abs(librosa.stft(y[: int(SCORE_SECONDS * sr)], n_fft=win, hop_length=hop))
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop)
    decays = np.full(n_partials, np.nan)
    for i, f in enumerate(freqs):
        if not np.isfinite(f):
            continue
        band = (f * 0.985, f * 1.015)
        lo = max(0, int(band[0] / (sr / win)))
        hi = min(stft.shape[0], max(lo + 1, int(band[1] / (sr / win)) + 1))
        e_db = 20 * np.log10(stft[lo:hi].max(axis=0) + 1e-9)
        peak_i = int(np.argmax(e_db))
        floor = e_db[peak_i] - 50
        end_i = peak_i + 1
        while end_i < len(e_db) and e_db[end_i] > floor:
            end_i += 1
        if end_i - peak_i >= 4:
            t = times[peak_i:end_i]
            decays[i] = -float(np.polyfit(t, e_db[peak_i:end_i], 1)[0])

    return PartialSet(f0, b_est, freqs, amps_db, decays)


def partial_distance(ref: PartialSet, syn: PartialSet):
    """Compare partial structure. Partials present in the reference but
    absent in the synth are charged at a fixed penalty per component."""
    comps = {}

    cents = 1200 * np.abs(np.log2((syn.f0 + 1e-9) / (ref.f0 + 1e-9)))
    comps["f0_cents"] = min(cents, 100) / 25.0

    both = np.isfinite(ref.freqs) & np.isfinite(syn.freqs)
    ref_only = np.isfinite(ref.freqs) & ~np.isfinite(syn.freqs)
    if both.sum() >= 2:
        dev = 1200 * np.abs(np.log2(syn.freqs[both] / ref.freqs[both]))
        freq_term = np.mean(np.minimum(dev, 100)) / 25.0
    else:
        freq_term = 2.0
    comps["partial_freq"] = freq_term + 2.0 * ref_only.sum() / len(ref.freqs)

    # Amplitude profile: 1.0 ~= 12 dB mean deviation. Missing partials in
    # the synth show up as huge negative dB and are penalized naturally.
    n = min(len(ref.amps_db), len(syn.amps_db))
    amp_dev = np.abs(ref.amps_db[:n] - syn.amps_db[:n])
    comps["partial_amps"] = float(np.mean(np.minimum(amp_dev, 40))) / 12.0

    both_decay = both & np.isfinite(ref.decays_db_s) & np.isfinite(syn.decays_db_s)
    if both_decay.sum() >= 1:
        d = np.abs(ref.decays_db_s[both_decay] - syn.decays_db_s[both_decay])
        comps["partial_decay"] = float(np.mean(np.minimum(d, 30))) / 10.0
    else:
        comps["partial_decay"] = 2.0

    return comps


# ---------------------------------------------------------------- combined

WEIGHTS = {
    "mel": 1.0,
    "envelope": 1.0,
    "f0_cents": 0.5,
    "partial_freq": 1.0,
    "partial_amps": 1.0,
    "partial_decay": 1.0,
}


def compare_pair(ref_audio, syn_audio, nominal_f0, sr=SR):
    """Full comparison of one (reference, synth) sample pair.
    Returns (total_loss, per-component dict)."""
    ref = prepare(ref_audio, sr)
    syn = prepare(syn_audio, sr)
    comps = {
        "mel": mel_distance(ref, syn, sr),
        "envelope": envelope_distance(ref, syn, sr),
    }
    comps.update(
        partial_distance(
            extract_partials(ref, nominal_f0, sr),
            extract_partials(syn, nominal_f0, sr),
        )
    )
    total = sum(WEIGHTS[k] * v for k, v in comps.items())
    return total, comps


def midi_note_freq(note):
    return 440.0 * 2 ** ((note - 69) / 12)
