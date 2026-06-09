//! Modal synthesis v1: each note is a bank of exponentially decaying
//! inharmonic partials, the physically-correct skeleton of a piano tone.
//!
//! Per-note parameters come from smooth heuristic curves anchored to
//! published piano data (and our own measurement of the reference C4
//! inharmonicity). Per-note *fitting* against the reference samples is the
//! next step; this version establishes the structure.
//!
//! Each partial is a complex phasor rotated by `e^{iω}` and scaled by a
//! per-sample decay factor — numerically stable and cheap (4 mul + 2 add
//! per partial per sample).

use crate::{Synth, midi_note_freq};

const MAX_VOICES: usize = 40;
const MAX_PARTIALS: usize = 28;
/// Strike point as a fraction of string length (typical piano ~1/8).
const STRIKE_POS: f32 = 0.12;

/// Log-linear interpolation of inharmonicity B through three anchors:
/// A0 ≈ 1e-4, C4 ≈ 2.8e-4 (measured from the reference), C8 ≈ 1e-2.
fn inharmonicity(note: u8) -> f32 {
    let pts = [(21.0f32, 1.0e-4f32), (60.0, 2.8e-4), (108.0, 1.0e-2)];
    let n = note as f32;
    let (a, b) = if n < pts[1].0 { (pts[0], pts[1]) } else { (pts[1], pts[2]) };
    let t = ((n - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
    (a.1.ln() + t * (b.1.ln() - a.1.ln())).exp()
}

/// Base decay rate of the fundamental in dB/s. Bass notes ring for tens of
/// seconds, top notes die in under a second.
fn base_decay_db_s(note: u8) -> f32 {
    let t = (note as f32 - 21.0) / 87.0;
    // ~1.5 dB/s at A0 → ~60 dB/s at C8, exponential in between.
    1.5 * (40.0f32).powf(t)
}

struct Partial {
    re: f32,
    im: f32,
    rot_re: f32,
    rot_im: f32,
    decay: f32, // per-sample amplitude factor
}

struct Voice {
    note: u8,
    partials: [Partial; MAX_PARTIALS],
    n_partials: usize,
    held: bool,
    /// Extra per-sample decay applied when the damper is on the string.
    damping: f32,
    pan_l: f32,
    pan_r: f32,
}

pub struct ModalSynth {
    sample_rate: f32,
    voices: Vec<Voice>,
    sustain: bool,
}

impl ModalSynth {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            voices: Vec::with_capacity(MAX_VOICES),
            sustain: false,
        }
    }
}

impl Synth for ModalSynth {
    fn note_on(&mut self, note: u8, velocity: u8) {
        if self.voices.len() == self.voices.capacity() {
            // Steal the quietest voice.
            if let Some(idx) = quietest(&self.voices) {
                self.voices.swap_remove(idx);
            }
        }
        let f0 = midi_note_freq(note);
        let b = inharmonicity(note);
        let vel = velocity as f32 / 127.0;

        // Brightness: spectral rolloff exponent from ~2.8 (pp) to ~1.6 (ff).
        let rolloff = 2.8 - 1.2 * vel;
        let base_decay = base_decay_db_s(note);
        let nyquist = self.sample_rate * 0.5 * 0.95;

        let mut partials = [(); MAX_PARTIALS].map(|_| Partial {
            re: 0.0, im: 0.0, rot_re: 1.0, rot_im: 0.0, decay: 1.0,
        });
        let mut n_partials = 0;
        for n in 1..=MAX_PARTIALS {
            let nf = n as f32;
            let freq = nf * f0 * (1.0 + b * nf * nf).sqrt();
            if freq >= nyquist {
                break;
            }
            // Amplitude: 1/n^rolloff shaped by the strike-point comb.
            let comb = (std::f32::consts::PI * nf * STRIKE_POS).sin().abs();
            let amp = 0.35 * vel * comb / nf.powf(rolloff);
            // Higher partials decay faster (frequency-dependent losses).
            let decay_db_s = base_decay * (1.0 + 0.12 * (nf - 1.0) + 0.7 * (freq / 4000.0));
            let decay = decay_factor(decay_db_s, self.sample_rate);
            let w = std::f32::consts::TAU * freq / self.sample_rate;
            partials[n - 1] = Partial {
                // Start at amplitude `amp`, phase 0 (phase coherence at the
                // attack is roughly what a hammer strike produces).
                re: amp,
                im: 0.0,
                rot_re: w.cos(),
                rot_im: w.sin(),
                decay,
            };
            n_partials += 1;
        }

        // Equal-power pan by keyboard position.
        let pos = (note as f32 - 21.0) / 87.0;
        let angle = (0.25 + 0.5 * pos) * std::f32::consts::FRAC_PI_2;

        self.voices.push(Voice {
            note,
            partials,
            n_partials,
            held: true,
            damping: 1.0,
            pan_l: angle.cos(),
            pan_r: angle.sin(),
        });
    }

    fn note_off(&mut self, note: u8) {
        for v in &mut self.voices {
            if v.note == note && v.held {
                v.held = false;
                if !self.sustain {
                    v.damping = damper_factor(v.note, self.sample_rate);
                }
            }
        }
    }

    fn set_sustain(&mut self, position: f32) {
        let down = position >= 0.5;
        if down != self.sustain {
            self.sustain = down;
            for v in &mut self.voices {
                if !v.held {
                    v.damping = if down {
                        1.0
                    } else {
                        damper_factor(v.note, self.sample_rate)
                    };
                }
            }
        }
    }

    fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        left.fill(0.0);
        right.fill(0.0);
        for voice in &mut self.voices {
            for i in 0..left.len() {
                let mut sum = 0.0;
                for p in &mut voice.partials[..voice.n_partials] {
                    sum += p.im;
                    let scale = p.decay * voice.damping;
                    let re = (p.re * p.rot_re - p.im * p.rot_im) * scale;
                    p.im = (p.re * p.rot_im + p.im * p.rot_re) * scale;
                    p.re = re;
                }
                left[i] += sum * voice.pan_l;
                right[i] += sum * voice.pan_r;
            }
        }
        // Drop voices that have decayed below audibility.
        self.voices.retain(|v| {
            v.partials[..v.n_partials]
                .iter()
                .map(|p| p.re * p.re + p.im * p.im)
                .sum::<f32>()
                > 1e-12
        });
    }
}

fn decay_factor(db_per_s: f32, sample_rate: f32) -> f32 {
    10f32.powf(-db_per_s / 20.0 / sample_rate)
}

/// Damper felt stops the string in roughly 50-300 ms depending on register
/// (high notes have no dampers at all above ~F#6, note 90).
fn damper_factor(note: u8, sample_rate: f32) -> f32 {
    if note >= 90 {
        return 1.0;
    }
    let stop_time = 0.05 + 0.10 * (1.0 - (note as f32 - 21.0) / 87.0);
    decay_factor(60.0 / stop_time, sample_rate)
}

fn quietest(voices: &[Voice]) -> Option<usize> {
    voices
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let e: f32 = v.partials[..v.n_partials]
                .iter()
                .map(|p| p.re * p.re + p.im * p.im)
                .sum();
            (i, e)
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
}
