//! Modal synthesis v2: per-note parameters fitted from the reference
//! samples ([`crate::calibration`]), plus two phenomena that v1 lacked:
//!
//! - **Unison detuning / double decay.** Most piano notes have 2-3 strings
//!   tuned a hair apart. Lower partials are rendered as two resonators with
//!   a small relative detune and different decay rates: their interference
//!   produces the beats and the prompt-sound/aftersound double decay of a
//!   real note, and kills the static "chime" character of a lone sinusoid.
//! - **Hammer noise.** A short filtered noise burst at the attack
//!   (velocity-dependent level and brightness).
//!
//! The resonator bank is laid out structure-of-arrays and stepped in
//! fixed 8-wide lanes so LLVM autovectorizes the hot loop; resonators that
//! decay below audibility are compacted out between blocks.

use crate::calibration::{Calibration, N_SLOTS};
use crate::{Synth, midi_note_freq};

const MAX_VOICES: usize = 40;
/// Only the lower partials get the unison pair; beating above this is
/// inaudible and the second resonator just costs CPU.
const UNISON_PARTIALS: usize = 16;
const LANES: usize = 8;
/// Resonators per voice: every partial, plus up to two extra unison
/// strings for the low partials, rounded up to a whole number of lanes.
const MAX_RES: usize = (N_SLOTS + 2 * UNISON_PARTIALS).div_ceil(LANES) * LANES;

struct Voice {
    note: u8,
    n: usize, // active resonators, always a multiple of LANES
    re: [f32; MAX_RES],
    im: [f32; MAX_RES],
    rot_re: [f32; MAX_RES],
    rot_im: [f32; MAX_RES],
    decay: [f32; MAX_RES],
    /// Base angular step per resonator, for the attack pitch glide.
    base_w: [f32; MAX_RES],
    /// Per-resonator damper factor (1.0 = open). Frequency-dependent: the
    /// damper felt kills high partials almost instantly while the
    /// fundamental of low notes audibly rings through it — a uniform cut
    /// reads as an unnatural abrupt stop.
    damping: [f32; MAX_RES],
    /// Damper landing blend 0..1: real dampers take ~35 ms to settle onto
    /// the string; instant grip is a derivative discontinuity in the
    /// envelope that reads as a clipped release.
    damper_blend: f32,
    held: bool,
    pan_l: f32,
    pan_r: f32,
    // Two-group attack swell from the measured rise time: the fundamental
    // region (first 8 resonators = partials 1-4) swells at the fitted
    // rate, upper partials arrive ~3x faster. Instant-on partials read
    // as a plucked string, not a hammered one.
    attack_slow: f32,
    attack_slow_coeff: f32,
    attack_fast: f32,
    attack_fast_coeff: f32,
    /// Attack pitch glide: a hard-struck string is momentarily sharp
    /// (tension modulation) and settles over ~100 ms. Current sharpness
    /// in cents; rotations are refreshed per block while it rings down.
    glide_cents: f32,
    // Hammer noise burst: filtered white noise with an exponential decay,
    // through a 2-pole lowpass (one pole leaves audible hiss — snare-like).
    noise_amp: f32,
    noise_decay: f32,
    noise_b: (f32, f32, f32),
    noise_a: (f32, f32),
    noise_z: (f32, f32),
    // Resonant bed (soundboard/sympathetic ring): noise through a 2-pole
    // resonant lowpass, decaying slowly until the damper falls.
    bed_amp: f32,
    bed_decay: f32,
    bed_b: (f32, f32, f32),
    bed_a: (f32, f32),
    bed_z: (f32, f32),
    bed_z2: (f32, f32),
    // Second cascaded stage (4-pole total): the real between-partial
    // residual is sparse tonal shimmer, not broadband noise — a 2-pole
    // bed leaves an audible hiss halo, worst in the partial-dense mid
    // range. Steeper filtering keeps the low body, kills the hiss.
    bed_y: (f32, f32),
    bed_y2: (f32, f32),
}

impl Voice {
    /// `phase` is randomized by the caller: starting every partial at
    /// phase 0 makes the first few ms of all partials add constructively —
    /// an N-times amplitude spike, worst for mid notes with many strong
    /// partials. The soundboard scrambles phase in reality.
    fn push_resonator(&mut self, freq: f32, amp: f32, phase: f32, decay_db_s: f32, sr: f32) {
        if self.n >= MAX_RES {
            return;
        }
        let w = std::f32::consts::TAU * freq / sr;
        let k = self.n;
        self.re[k] = amp * phase.cos();
        self.im[k] = amp * phase.sin();
        self.rot_re[k] = w.cos();
        self.rot_im[k] = w.sin();
        self.base_w[k] = w;
        self.decay[k] = decay_factor(decay_db_s, sr);
        self.n += 1;
    }

    /// Drop resonators below audibility and re-pad to a lane multiple.
    fn compact(&mut self) {
        let mut keep = 0;
        for k in 0..self.n {
            if self.re[k] * self.re[k] + self.im[k] * self.im[k] > 1e-12 {
                if keep != k {
                    self.re[keep] = self.re[k];
                    self.im[keep] = self.im[k];
                    self.rot_re[keep] = self.rot_re[k];
                    self.rot_im[keep] = self.rot_im[k];
                    self.base_w[keep] = self.base_w[k];
                    self.decay[keep] = self.decay[k];
                    self.damping[keep] = self.damping[k];
                }
                keep += 1;
            }
        }
        self.n = keep;
        self.pad();
    }

    fn pad(&mut self) {
        while self.n % LANES != 0 {
            let k = self.n;
            self.re[k] = 0.0;
            self.im[k] = 0.0;
            self.rot_re[k] = 1.0;
            self.rot_im[k] = 0.0;
            self.base_w[k] = 0.0;
            self.decay[k] = 0.0;
            self.damping[k] = 1.0;
            self.n += 1;
        }
    }

    /// Apply (or lift) the damper. Felt absorption rises with frequency:
    /// stop time = base(register) * 600 / (600 + f). `pressure` 0..1
    /// scales the damping rate for half-pedaling (0 = damper lifted).
    fn set_damper(&mut self, on: bool, note: u8, sr: f32, pressure: f32) {
        if !on || pressure <= 0.02 {
            self.damping[..self.n].fill(1.0);
            self.damper_blend = 1.0;
            return;
        }
        // Start the landing ramp only when coming from fully open.
        if self.damping[..self.n].iter().all(|&d| d >= 1.0) {
            self.damper_blend = 0.0;
        }
        let pos = (note as f32 - 21.0) / 87.0;
        let base = 0.20 + 0.45 * (1.0 - pos) * (1.0 - pos);
        for k in 0..self.n {
            if self.decay[k] == 0.0 {
                continue;
            }
            let freq = self.base_w[k] * sr / std::f32::consts::TAU;
            let stop = base * 600.0 / (600.0 + freq.max(0.0));
            let rate = 60.0 / stop.max(0.015) * pressure * pressure;
            self.damping[k] = decay_factor(rate, sr);
        }
    }

    fn energy(&self) -> f32 {
        let mut e = self.noise_amp * self.noise_amp + self.bed_amp * self.bed_amp;
        for k in 0..self.n {
            e += self.re[k] * self.re[k] + self.im[k] * self.im[k];
        }
        e
    }
}

/// Sympathetic resonance: every piano string shares the bridge, so any
/// sounding note rings every *undamped* string — keys held down, pedal
/// down, and everything above F#6 (no dampers). Modeled as a bank of
/// 88 strings x 3 partials of resonators tuned from the calibration,
/// driven by the voice mix, with per-string damper gating.
const BANK_PARTIALS: usize = 3;
const BANK_NOTES: usize = 88;
const BANK_RES: usize = BANK_NOTES * BANK_PARTIALS; // multiple of LANES

struct StringBank {
    re: [f32; BANK_RES],
    im: [f32; BANK_RES],
    rot_re: [f32; BANK_RES],
    rot_im: [f32; BANK_RES],
    decay: [f32; BANK_RES],   // current per-resonator decay (open or damped)
    open_decay: [f32; BANK_RES],
    damped_decay: [f32; BANK_RES],
    gain_in: [f32; BANK_RES],
    pan_l: [f32; BANK_RES],
    pan_r: [f32; BANK_RES],
    open: [bool; BANK_NOTES],
    /// ping[played_note_index][resonator]: how strongly a note-on of
    /// `played_note` rings each string partial — partial amplitudes of the
    /// played note times a frequency-proximity kernel. Gives the instant
    /// sympathetic ping that slow driven buildup cannot.
    ping: Vec<f32>,
}

impl StringBank {
    fn new(sample_rate: f32) -> Box<StringBank> {
        let cal = Calibration::embedded();
        let mut bank = Box::new(StringBank {
            re: [0.0; BANK_RES],
            im: [0.0; BANK_RES],
            rot_re: [1.0; BANK_RES],
            rot_im: [0.0; BANK_RES],
            decay: [0.0; BANK_RES],
            open_decay: [0.0; BANK_RES],
            damped_decay: [0.0; BANK_RES],
            gain_in: [0.0; BANK_RES],
            pan_l: [0.0; BANK_RES],
            pan_r: [0.0; BANK_RES],
            open: [false; BANK_NOTES],
            ping: vec![0.0; BANK_NOTES * BANK_RES],
        });
        let nyquist = sample_rate * 0.5 * 0.95;
        for string in 0..BANK_NOTES {
            let note = 21 + string as u8;
            let params = cal.lookup(note, 100);
            let f0 = midi_note_freq(note) * 2f32.powf(params.f0_cents / 1200.0);
            let pos = string as f32 / 87.0;
            let angle = (0.25 + 0.5 * pos) * std::f32::consts::FRAC_PI_2;
            for p in 0..BANK_PARTIALS {
                let k = string * BANK_PARTIALS + p;
                let nf = (p + 1) as f32;
                let freq = nf * f0 * (1.0 + params.b * nf * nf).sqrt();
                if freq >= nyquist {
                    continue; // stays a dead resonator (gain 0, decay 0)
                }
                let w = std::f32::consts::TAU * freq / sample_rate;
                bank.rot_re[k] = w.cos();
                bank.rot_im[k] = w.sin();
                // Sympathetic ring decays like the string's aftersound.
                let ring = params.decays_slow[p].clamp(1.0, 30.0);
                bank.open_decay[k] = decay_factor(ring, sample_rate);
                // Resting dampers LEAK: partials with nodes at the damper
                // position keep ringing, and duplex segments have no
                // dampers at all. ~35 dB/s instead of a felt-stop kill —
                // this leakage is most of a real piano's release halo.
                bank.damped_decay[k] =
                    decay_factor(35.0, sample_rate).min(bank.open_decay[k]);
                bank.decay[k] = bank.damped_decay[k];
                // Lower partials couple more strongly through the bridge.
                // The (1 - decay) factor normalizes resonant buildup: a
                // driven resonator accumulates ~ gain/(1-decay) at its own
                // frequency, which is ~1e5 for slow-ringing strings. The
                // leading constant is the bridge coupling strength, set so
                // a staccato strike leaves an audible ring in matched
                // open strings (one-way drive can't get both transient and
                // steady-state coupling from first principles).
                bank.gain_in[k] = 8.0 * (1.0 - bank.open_decay[k]) / nf;
                bank.pan_l[k] = angle.cos();
                bank.pan_r[k] = angle.sin();
            }
            if note >= 90 {
                // No dampers up here: always open.
                bank.open[string] = true;
                for p in 0..BANK_PARTIALS {
                    let k = string * BANK_PARTIALS + p;
                    bank.decay[k] = bank.open_decay[k];
                    bank.damped_decay[k] = bank.open_decay[k];
                }
            }
        }

        // Coupling table: for every (played note, string partial) pair.
        let mut bank_freqs = [0f32; BANK_RES];
        for k in 0..BANK_RES {
            // Recover the resonator frequency from its rotation.
            bank_freqs[k] =
                bank.rot_im[k].atan2(bank.rot_re[k]) * sample_rate / std::f32::consts::TAU;
        }
        for played in 0..BANK_NOTES {
            let note = 21 + played as u8;
            let params = cal.lookup(note, 100);
            let f0 = midi_note_freq(note) * 2f32.powf(params.f0_cents / 1200.0);
            for m in 1..=16usize {
                let mf = m as f32;
                let freq = mf * f0 * (1.0 + params.b * mf * mf).sqrt();
                if freq >= nyquist {
                    break;
                }
                let amp = 10f32.powf(params.amps_db[m - 1].clamp(-40.0, 6.0) / 20.0);
                let width = 1.5 + 0.004 * freq; // Hz, ~string bandwidth
                for k in 0..BANK_RES {
                    if bank.gain_in[k] == 0.0 {
                        continue;
                    }
                    let df = (bank_freqs[k] - freq).abs();
                    if df < width * 4.0 {
                        let kernel = 1.0 / (1.0 + (df / width) * (df / width));
                        bank.ping[played * BANK_RES + k] += amp * kernel / mf.sqrt();
                    }
                }
            }
        }
        bank
    }

    /// Ring all open strings (except the played one) at note-on.
    fn ping(&mut self, note: u8, strength: f32, rng: &mut u32) {
        if !(21..21 + BANK_NOTES as u8).contains(&note) {
            return;
        }
        let played = (note - 21) as usize;
        for k in 0..BANK_RES {
            if !self.open[k / BANK_PARTIALS] || k / BANK_PARTIALS == played {
                continue;
            }
            let c = self.ping[played * BANK_RES + k];
            if c <= 1e-4 {
                continue;
            }
            *rng ^= *rng << 13;
            *rng ^= *rng >> 17;
            *rng ^= *rng << 5;
            let phase = (*rng as f32 / u32::MAX as f32) * std::f32::consts::TAU;
            self.re[k] += strength * c * phase.cos();
            self.im[k] += strength * c * phase.sin();
        }
    }

    fn set_open(&mut self, note: u8, open: bool) {
        if !(21..21 + BANK_NOTES as u8).contains(&note) {
            return;
        }
        let string = (note - 21) as usize;
        self.open[string] = open || note >= 90;
        for p in 0..BANK_PARTIALS {
            let k = string * BANK_PARTIALS + p;
            self.decay[k] = if open { self.open_decay[k] } else { self.damped_decay[k] };
        }
    }

    /// Advance one sample: excite with the bridge signal, return (l, r).
    #[inline]
    fn step(&mut self, drive: f32) -> (f32, f32) {
        let mut acc_l = [0f32; LANES];
        let mut acc_r = [0f32; LANES];
        for chunk in (0..BANK_RES).step_by(LANES) {
            for lane in 0..LANES {
                let k = chunk + lane;
                let out = self.im[k];
                let re = (self.re[k] * self.rot_re[k] - self.im[k] * self.rot_im[k])
                    * self.decay[k]
                    + drive * self.gain_in[k];
                self.im[k] =
                    (self.re[k] * self.rot_im[k] + self.im[k] * self.rot_re[k]) * self.decay[k];
                self.re[k] = re;
                acc_l[lane] += out * self.pan_l[k];
                acc_r[lane] += out * self.pan_r[k];
            }
        }
        (acc_l.iter().sum(), acc_r.iter().sum())
    }
}

/// Scalar tuning knobs. Defaults are the shipped values; each can be
/// overridden via environment variable (PIANO_DETUNE_SCALE etc.) so the
/// scoring harness can hill-climb them without rebuilding.
#[derive(Clone, Copy)]
struct Knobs {
    detune_scale: f32,
    attack_scale: f32,
    glide_cents: f32,
    bed_gain: f32,
    noise_gain: f32,
    split_max: f32,
    fast_scale: f32,
    bank_drive: f32,
    bank_ping: f32,
    body_gain: f32,
}

impl Knobs {
    fn from_env() -> Knobs {
        let get = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        Knobs {
            detune_scale: get("PIANO_DETUNE_SCALE", 1.82),
            attack_scale: get("PIANO_ATTACK_SCALE", 0.77),
            glide_cents: get("PIANO_GLIDE_CENTS", 3.0),
            bed_gain: get("PIANO_BED_GAIN", 0.82),
            noise_gain: get("PIANO_NOISE_GAIN", 1.0),
            split_max: get("PIANO_SPLIT_MAX", 0.7),
            fast_scale: get("PIANO_FAST_SCALE", 0.8),
            bank_drive: get("PIANO_BANK_DRIVE", 0.031),
            bank_ping: get("PIANO_BANK_PING", 0.08),
            body_gain: get("PIANO_BODY_GAIN", 0.25),
        }
    }
}

/// The soundboard body: a small dark feedback-delay network that the
/// whole instrument plays into. Unlike the per-voice bed (which dies
/// with its voice) and the room (presentation-only), the body is part of
/// the instrument: it keeps ringing ~0.4 s after dampers fall — the
/// 'dry reverberance' a real piano has even in a dead room — and gives
/// the top octave its knock-into-the-box character.
struct Body {
    lines: [Vec<f32>; 4],
    pos: [usize; 4],
    fb: [f32; 4],
    damp: [f32; 4],
    damp_coeff: f32,
    gain: f32,
}

impl Body {
    fn new(sample_rate: f32, gain: f32) -> Body {
        let ms = |m: f32| ((m / 1000.0 * sample_rate) as usize).max(1);
        // Short, mutually-detuned delays: dense early soundboard modes.
        let lens = [ms(7.9), ms(11.3), ms(14.7), ms(18.9)];
        let rt = 0.55; // seconds to -60 dB
        let mut fb = [0f32; 4];
        for (i, &len) in lens.iter().enumerate() {
            fb[i] = 10f32.powf(-3.0 * len as f32 / (rt * sample_rate));
        }
        Body {
            lines: lens.map(|l| vec![0.0; l]),
            pos: [0; 4],
            fb,
            damp: [0.0; 4],
            damp_coeff: (-std::f32::consts::TAU * 2800.0 / sample_rate).exp(),
            gain,
        }
    }

    #[inline]
    fn step(&mut self, input: f32) -> f32 {
        let mut taps = [0f32; 4];
        let mut sum = 0.0;
        for k in 0..4 {
            taps[k] = self.lines[k][self.pos[k]];
            sum += taps[k];
        }
        let h = 0.5 * sum; // Householder (2/N, N=4)
        let mut out = 0.0;
        for k in 0..4 {
            let mut v = (taps[k] - h) * self.fb[k] + input * 0.5;
            self.damp[k] = self.damp[k] * self.damp_coeff + v * (1.0 - self.damp_coeff);
            v = self.damp[k];
            self.lines[k][self.pos[k]] = v;
            self.pos[k] += 1;
            if self.pos[k] == self.lines[k].len() {
                self.pos[k] = 0;
            }
            out += if k % 2 == 0 { taps[k] } else { -taps[k] };
        }
        out * self.gain
    }
}

pub struct ModalV2 {
    sample_rate: f32,
    voices: Vec<Voice>,
    sustain: bool,
    /// Continuous CC64 position for half-pedaling.
    sustain_pos: f32,
    /// CC67 una corda: hammers shifted, striking fewer strings softly.
    una_corda: bool,
    rng: u32,
    held: [bool; 128],
    bank: Box<StringBank>,
    /// Bridge coupling into the bank and bank level back into the mix.
    bank_drive: f32,
    bank_level: f32,
    body: Body,
    knobs: Knobs,
}

impl ModalV2 {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            voices: Vec::with_capacity(MAX_VOICES),
            sustain: false,
            sustain_pos: 0.0,
            una_corda: false,
            rng: 0x12345678,
            held: [false; 128],
            bank: StringBank::new(sample_rate),
            bank_drive: Knobs::from_env().bank_drive,
            bank_level: 1.0,
            body: Body::new(sample_rate, Knobs::from_env().body_gain),
            knobs: Knobs::from_env(),
        }
    }
}

/// Matches BED_DECAY_DB_S in the calibration fitter: bed levels were
/// back-projected to t=0 assuming this playback decay rate.
const BED_DECAY_DB_S: f32 = 4.0;

/// RBJ biquad lowpass coefficients (normalized so a0 = 1).
fn biquad_lowpass(freq: f32, q: f32, sr: f32) -> ((f32, f32, f32), (f32, f32)) {
    let w0 = std::f32::consts::TAU * (freq / sr).min(0.45);
    let alpha = w0.sin() / (2.0 * q);
    let cos = w0.cos();
    let a0 = 1.0 + alpha;
    (
        (
            (1.0 - cos) / 2.0 / a0,
            (1.0 - cos) / a0,
            (1.0 - cos) / 2.0 / a0,
        ),
        (-2.0 * cos / a0, (1.0 - alpha) / a0),
    )
}

/// Unison detune in cents, wider toward the bass (real pianos beat more
/// audibly low down; trebles are tuned very tight).
fn unison_detune_cents(note: u8) -> f32 {
    let pos = (note as f32 - 21.0) / 87.0;
    1.6 - 0.9 * pos
}

fn decay_factor(db_per_s: f32, sample_rate: f32) -> f32 {
    10f32.powf(-db_per_s / 20.0 / sample_rate)
}

/// Damper felt stops the string in roughly 50-150 ms depending on register
/// (notes above ~F#6 have no dampers at all).
fn damper_factor(note: u8, sample_rate: f32) -> f32 {
    if note >= 90 {
        return 1.0;
    }
    let stop_time = 0.05 + 0.10 * (1.0 - (note as f32 - 21.0) / 87.0);
    decay_factor(60.0 / stop_time, sample_rate)
}

impl Synth for ModalV2 {
    fn note_on(&mut self, note: u8, velocity: u8) {
        // Re-strike: the hammer re-contact largely replaces the previous
        // vibration of this string; letting the old voice ring unattenuated
        // alongside the new one piles up energy in fast repeated notes.
        let sr = self.sample_rate;
        for v in &mut self.voices {
            if v.note == note {
                v.held = false;
                v.set_damper(true, note, sr, 1.0);
            }
        }
        if self.voices.len() == self.voices.capacity() {
            if let Some(idx) = quietest(&self.voices) {
                self.voices.swap_remove(idx);
            }
        }
        // Una corda: the shifted hammer strikes fewer strings on softer
        // felt — quieter, darker, and more energy in the freely-resonating
        // aftersound.
        let una = self.una_corda;
        let params = Calibration::embedded().lookup(note, velocity);
        let vel = velocity as f32 / 127.0;
        let f0 = midi_note_freq(note) * 2f32.powf(params.f0_cents / 1200.0);
        let nyquist = self.sample_rate * 0.5 * 0.95;
        // Target early RMS: global gain x velocity curve x the measured
        // keyboard balance of the reference piano. Partial amplitudes are
        // normalized to hit this exactly after the voice is built.
        let mut target_rms = 0.06 * vel.powf(1.6) * 10f32.powf(params.loudness_db / 20.0);
        if una {
            target_rms *= 0.65;
        }
        let level = 1.0; // provisional partial scale, normalized below
        let detune_ratio = (unison_detune_cents(note) * self.knobs.detune_scale / 1200.0
            * std::f32::consts::LN_2)
            .exp_m1();

        let pos = (note as f32 - 21.0) / 87.0;
        // Hard strikes start sharp: tension-modulation glide, strongest
        // low on the keyboard, negligible for soft playing.
        let glide0 = self.knobs.glide_cents * vel * vel * vel * (1.2 - pos);
        let angle = (0.15 + 0.7 * pos) * std::f32::consts::FRAC_PI_2;
        // Hammer/action noise: dark (action thump + soundboard knock live
        // mostly below ~1-3 kHz), brighter when hit hard and toward the
        // treble. It also grows faster with velocity than the tone does —
        // pp notes have almost none.
        let pos4 = pos * pos * pos * pos;
        let cutoff = (300.0 + 2500.0 * vel * vel) * (0.5 + 0.8 * pos + 0.9 * pos4);
        let noise_seconds = 0.010 + 0.015 * (1.0 - pos);
        let noise_filter = biquad_lowpass(cutoff, 0.9, self.sample_rate);

        let mut voice = Voice {
            note,
            n: 0,
            re: [0.0; MAX_RES],
            im: [0.0; MAX_RES],
            rot_re: [1.0; MAX_RES],
            rot_im: [0.0; MAX_RES],
            base_w: [0.0; MAX_RES],
            decay: [0.0; MAX_RES],
            damping: [1.0; MAX_RES],
            damper_blend: 1.0,
            glide_cents: glide0,
            held: true,
            pan_l: angle.cos(),
            pan_r: angle.sin(),
            attack_slow: 0.0,
            attack_slow_coeff: {
                // Linear ramp hitting 1.0 exactly at the fitted rise time.
                let t = (params.attack_ms * self.knobs.attack_scale / 1000.0).clamp(0.002, 0.09);
                1.0 / (t * self.sample_rate)
            },
            attack_fast: 0.0,
            attack_fast_coeff: {
                let t = (params.attack_ms * self.knobs.attack_scale / 1000.0).clamp(0.002, 0.09)
                    / 3.0;
                1.0 / (t * self.sample_rate)
            },
            noise_amp: target_rms * (0.12 + 0.30 * pos + 0.5 * pos4) * vel
                * self.knobs.noise_gain,
            noise_decay: decay_factor(60.0 / noise_seconds, self.sample_rate),
            noise_b: noise_filter.0,
            noise_a: noise_filter.1,
            noise_z: (0.0, 0.0),
            // bed_db was measured relative to the note's early RMS, which
            // is exactly target_rms after normalization.
            bed_amp: target_rms * 10f32.powf(params.bed_db / 20.0) * self.knobs.bed_gain,
            bed_decay: decay_factor(BED_DECAY_DB_S, self.sample_rate),
            bed_b: biquad_lowpass(params.bed_centroid_hz * 0.6, 1.2, self.sample_rate).0,
            bed_a: biquad_lowpass(params.bed_centroid_hz * 0.6, 1.2, self.sample_rate).1,
            bed_z: (0.0, 0.0),
            bed_z2: (0.0, 0.0),
            bed_y: (0.0, 0.0),
            bed_y2: (0.0, 0.0),
        };

        let mut rng = self.rng;
        let mut rand_phase = move || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            (rng as f32 / u32::MAX as f32) * std::f32::consts::TAU
        };
        for n in 1..=N_SLOTS {
            let nf = n as f32;
            let freq = nf * f0 * (1.0 + params.b * nf * nf).sqrt();
            if freq >= nyquist {
                break;
            }
            // Calibration covers velocities 54..114; outside that range
            // extrapolate brightness with a spectral tilt (softer hammer =
            // darker, harder = brighter) since the layers carry no data.
            let tilt_db = if velocity < 54 {
                -0.010 * (54 - velocity) as f32 * (nf - 1.0)
            } else if velocity > 114 {
                0.008 * (velocity - 114) as f32 * (nf - 1.0)
            } else {
                0.0
            };
            let una_tilt = if una { -0.35 * (nf - 1.0) } else { 0.0 };
            let amp = level * 10f32.powf((params.amps_db[n - 1] + tilt_db + una_tilt) / 20.0);
            let fast = (params.decays_fast[n - 1] * self.knobs.fast_scale).clamp(0.3, 300.0);
            let slow = params.decays_slow[n - 1].clamp(0.3, fast);
            // Cap the slow share of the dominant low partials: a 50/50
            // split lets the unison components cancel completely, which
            // is deeper than real instruments ever beat.
            let split_cap = (0.30 + 0.06 * (nf - 1.0)).min(self.knobs.split_max);
            let mut split = params.slow_split[n - 1].clamp(0.02, split_cap);
            if una {
                split = (split * 1.6).min(0.9);
            }
            if n <= UNISON_PARTIALS {
                // Split the partial across the unison strings using the
                // fitted prompt-sound / aftersound rates: their sum
                // reproduces the measured double decay, their detunes the
                // beats. Notes with three strings get three resonators —
                // a two-resonator unison cancels COMPLETELY whenever the
                // fitted split passes near 0.5 (audible as a throbbing
                // null on specific notes); three phasors with scrambled
                // phases essentially never null together.
                // The slow aftersound is mostly the string's perpendicular
                // polarization — phase-quadrature to the prompt sound and
                // nearly co-tuned. Quadrature components cannot cancel, so
                // the fast/slow amplitude crossing is a smooth crossover
                // instead of the deep destructive notch a free-phase pair
                // produces (the 'distorted mid note' defect). A separate,
                // genuinely detuned small component carries the audible
                // unison beat shimmer.
                let phase_c = rand_phase();
                voice.push_resonator(
                    freq, amp * (1.0 - split), phase_c, fast, self.sample_rate,
                );
                let quad_share = if note >= 44 { 0.7 } else { 1.0 };
                voice.push_resonator(
                    freq + 0.05,
                    amp * split * quad_share,
                    phase_c + std::f32::consts::FRAC_PI_2,
                    slow,
                    self.sample_rate,
                );
                if note >= 44 {
                    let jit = 0.75 + 0.5 * (rand_phase() / std::f32::consts::TAU);
                    let df = (freq * detune_ratio * jit).max(0.7);
                    voice.push_resonator(
                        freq + df,
                        amp * split * 0.3,
                        rand_phase(),
                        slow * 1.3,
                        self.sample_rate,
                    );
                }
            } else {
                voice.push_resonator(freq, amp, rand_phase(), fast, self.sample_rate);
            }
        }
        // Normalize so the voice's RMS over the same 0.4 s window the
        // calibration measured hits target_rms. Each resonator's mean
        // square over W samples is (a^2/2) * (1 - d^2W) / (W (1 - d^2)) —
        // for fast-decaying treble notes this is far below the initial
        // RMS, and ignoring it left the treble too quiet.
        let window = (0.4 * self.sample_rate) as i32;
        let mean_square: f32 = (0..voice.n)
            .map(|k| {
                let a2 = voice.re[k] * voice.re[k] + voice.im[k] * voice.im[k];
                let d2 = voice.decay[k] * voice.decay[k];
                let g = if d2 > 0.999_999 {
                    1.0
                } else {
                    (1.0 - d2.powi(window)) / (window as f32 * (1.0 - d2))
                };
                a2 / 2.0 * g
            })
            .sum();
        let scale = target_rms / mean_square.sqrt().max(1e-9);
        for k in 0..voice.n {
            voice.re[k] *= scale;
            voice.im[k] *= scale;
        }
        voice.pad();
        self.voices.push(voice);
        // Advance the synth RNG so consecutive notes get fresh phases.
        self.rng = self.rng.wrapping_mul(0x9E3779B9).wrapping_add(1);
        self.held[note as usize] = true;
        let mut rng = self.rng;
        self.bank.ping(note, target_rms * self.knobs.bank_ping, &mut rng);
        self.rng = rng;
        self.bank.set_open(note, true);
    }

    fn note_off(&mut self, note: u8) {
        self.held[note as usize] = false;
        let sr = self.sample_rate;
        for v in &mut self.voices {
            if v.note == note && v.held {
                v.held = false;
                // Damper pressure follows the (possibly partial) pedal.
                let pressure = 1.0 - self.sustain_pos;
                v.set_damper(true, note, sr, pressure);
                if pressure > 0.4 {
                    // The soundboard keeps ringing briefly after the string
                    // is damped; release the bed on its own ~0.3 s slope
                    // instead of cutting it with the string.
                    v.bed_decay = decay_factor(60.0 / 0.6, sr);
                    // Damper thud: a dark noise burst scaled by how much
                    // the string was still vibrating, with a faint
                    // mechanical floor (the key/damper always clunks a
                    // little, even on a silent string). Dampers don't
                    // exist above F#6.
                    if v.note < 90 {
                        // The 420 Hz lowpass discards most broadband noise
                        // energy; the leading constant compensates.
                        let ringing =
                            (v.energy() / 2.0).sqrt() * 1.5 + 0.002;
                        v.noise_amp = v.noise_amp.max(ringing);
                        v.noise_decay = decay_factor(60.0 / 0.045, sr);
                        let f = biquad_lowpass(420.0, 0.8, sr);
                        v.noise_b = f.0;
                        v.noise_a = f.1;
                    }
                }
            }
        }
        if !self.sustain {
            self.bank.set_open(note, false);
        }
    }

    fn set_sustain(&mut self, position: f32) {
        self.sustain_pos = position;
        let down = position >= 0.5;
        let sr = self.sample_rate;
        // Half-pedal: continuously rescale damping on released voices.
        for v in &mut self.voices {
            if !v.held {
                let note = v.note;
                v.set_damper(true, note, sr, 1.0 - position);
                v.bed_decay = if down {
                    decay_factor(BED_DECAY_DB_S, sr)
                } else {
                    decay_factor(60.0 / 0.6, sr)
                };
            }
        }
        if down != self.sustain {
            self.sustain = down;
            for note in 21..21 + BANK_NOTES as u8 {
                if down {
                    self.bank.set_open(note, true);
                } else if !self.held[note as usize] {
                    self.bank.set_open(note, false);
                }
            }
        }
    }

    fn set_control(&mut self, controller: u8, value: f32) {
        if controller == 67 {
            self.una_corda = value >= 0.5;
        }
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        left.fill(0.0);
        right.fill(0.0);
        for voice in &mut self.voices {
            let blend = voice.damper_blend;
            voice.damper_blend =
                (voice.damper_blend + left.len() as f32 / (0.035 * self.sample_rate)).min(1.0);
            if voice.glide_cents > 0.05 {
                let ratio = 2f32.powf(voice.glide_cents / 1200.0);
                for k in 0..voice.n {
                    if voice.decay[k] > 0.0 {
                        let w = voice.base_w[k] * ratio;
                        voice.rot_re[k] = w.cos();
                        voice.rot_im[k] = w.sin();
                    }
                }
                // ~60 ms settling time, advanced once per block.
                voice.glide_cents *=
                    (-(left.len() as f32) / (0.06 * self.sample_rate)).exp();
                if voice.glide_cents <= 0.05 {
                    voice.glide_cents = 0.0;
                    for k in 0..voice.n {
                        if voice.decay[k] > 0.0 {
                            voice.rot_re[k] = voice.base_w[k].cos();
                            voice.rot_im[k] = voice.base_w[k].sin();
                        }
                    }
                }
            }
            for i in 0..left.len() {
                let mut sum_slow = 0.0f32;
                let mut sum_fast = 0.0f32;
                for chunk in (0..voice.n).step_by(LANES) {
                    let mut acc = [0f32; LANES];
                    // Fixed-width branchless lane loop: autovectorizes.
                    for lane in 0..LANES {
                        let k = chunk + lane;
                        let out = voice.im[k];
                        let scale = voice.decay[k]
                            * (1.0 + (voice.damping[k] - 1.0) * blend);
                        let re = (voice.re[k] * voice.rot_re[k]
                            - voice.im[k] * voice.rot_im[k])
                            * scale;
                        voice.im[k] = (voice.re[k] * voice.rot_im[k]
                            + voice.im[k] * voice.rot_re[k])
                            * scale;
                        voice.re[k] = re;
                        acc[lane] += out;
                    }
                    let chunk_sum: f32 = acc.iter().sum();
                    if chunk == 0 {
                        sum_slow += chunk_sum;
                    } else {
                        sum_fast += chunk_sum;
                    }
                }
                voice.attack_slow = (voice.attack_slow + voice.attack_slow_coeff).min(1.0);
                voice.attack_fast = (voice.attack_fast + voice.attack_fast_coeff).min(1.0);
                let sum = sum_slow * voice.attack_slow + sum_fast * voice.attack_fast;
                let mut l = sum * voice.pan_l;
                let mut r = sum * voice.pan_r;
                if voice.noise_amp > 1e-7 || voice.bed_amp > 1e-7 {
                    // xorshift32 white noise; two draws so the bed is
                    // genuinely stereo (independent L/R streams).
                    self.rng ^= self.rng << 13;
                    self.rng ^= self.rng >> 17;
                    self.rng ^= self.rng << 5;
                    let white = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    self.rng ^= self.rng << 13;
                    self.rng ^= self.rng >> 17;
                    self.rng ^= self.rng << 5;
                    let white2 = (self.rng as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    // Hammer burst: resonant 2-pole lowpass.
                    {
                        let (b0, b1, b2) = voice.noise_b;
                        let (a1, a2) = voice.noise_a;
                        let y = b0 * white + voice.noise_z.0;
                        voice.noise_z.0 = b1 * white - a1 * y + voice.noise_z.1;
                        voice.noise_z.1 = b2 * white - a2 * y;
                        l += y * voice.noise_amp * voice.pan_l;
                        r += y * voice.noise_amp * voice.pan_r;
                        voice.noise_amp *= voice.noise_decay;
                    }
                    // Bed: two independent filtered noise streams.
                    let (b0, b1, b2) = voice.bed_b;
                    let (a1, a2) = voice.bed_a;
                    let mut y = b0 * white + voice.bed_z.0;
                    voice.bed_z.0 = b1 * white - a1 * y + voice.bed_z.1;
                    voice.bed_z.1 = b2 * white - a2 * y;
                    let mut y2 = b0 * white2 + voice.bed_z2.0;
                    voice.bed_z2.0 = b1 * white2 - a1 * y2 + voice.bed_z2.1;
                    voice.bed_z2.1 = b2 * white2 - a2 * y2;
                    // Second cascaded stage.
                    let yc = b0 * y + voice.bed_y.0;
                    voice.bed_y.0 = b1 * y - a1 * yc + voice.bed_y.1;
                    voice.bed_y.1 = b2 * y - a2 * yc;
                    y = yc;
                    let yc2 = b0 * y2 + voice.bed_y2.0;
                    voice.bed_y2.0 = b1 * y2 - a1 * yc2 + voice.bed_y2.1;
                    voice.bed_y2.1 = b2 * y2 - a2 * yc2;
                    y2 = yc2;
                    // ~+3 dB makes up the passband loss of the cascade.
                    l += y * voice.bed_amp * 1.4 * voice.pan_l;
                    r += y2 * voice.bed_amp * 1.4 * voice.pan_r;
                    voice.bed_amp *= voice.bed_decay;
                }
                left[i] += l;
                right[i] += r;
            }
            voice.compact();
        }
        // Sympathetic bank: driven by the voice mix (no feedback — the
        // bank does not hear itself). The soundboard body rings on top of
        // everything and survives note releases.
        for i in 0..left.len() {
            let drive = (left[i] + right[i]) * self.bank_drive;
            let (bl, br) = self.bank.step(drive);
            left[i] += bl * self.bank_level;
            right[i] += br * self.bank_level;
            let body = self.body.step(0.5 * (left[i] + right[i]));
            left[i] += body;
            right[i] += body;
        }
        self.voices.retain(|v| v.energy() > 1e-12);
    }
}

fn quietest(voices: &[Voice]) -> Option<usize> {
    voices
        .iter()
        .enumerate()
        .map(|(i, v)| (i, v.energy()))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}
