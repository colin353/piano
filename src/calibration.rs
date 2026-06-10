//! Fitted per-note parameters, measured from the reference samples by
//! `piano-calibrate` and embedded at compile time. The synth interpolates
//! across note (within a dynamic layer) and across velocity (between
//! layers); fitted amplitude profiles are relative to partial 1, fitted
//! tuning is in cents from equal temperament (the Railsback stretch).

use serde::Deserialize;

pub const N_SLOTS: usize = 150;

#[derive(Deserialize)]
struct RawCalibration {
    version: u32,
    layers: std::collections::BTreeMap<String, Vec<RawNote>>,
}

#[derive(Deserialize)]
struct RawNote {
    note: u8,
    velocity: u8,
    f0_cents: f32,
    b: f32,
    bed_db: f32,
    bed_centroid_hz: f32,
    loudness_db: f32,
    attack_ms: f32,
    amps_db: Vec<f32>,
    decays_fast_db_s: Vec<f32>,
    decays_slow_db_s: Vec<f32>,
    slow_split: Vec<f32>,
}

/// Per-partial decay is two-stage: the prompt sound decays at
/// `decays_fast`, the aftersound at `decays_slow`, with `slow_split` of
/// the initial amplitude in the slow component.
pub struct NoteParams {
    pub f0_cents: f32,
    pub b: f32,
    /// Broadband resonant bed (soundboard/sympathetic ring): level in dB
    /// relative to the note's early sound, and spectral centroid in Hz.
    pub bed_db: f32,
    pub bed_centroid_hz: f32,
    /// Early-RMS loudness of this note relative to its layer median, dB —
    /// the measured keyboard balance of the reference piano.
    pub loudness_db: f32,
    /// Onset-to-peak rise time, ms. Instant attacks read as plucked.
    pub attack_ms: f32,
    pub amps_db: [f32; N_SLOTS],
    pub decays_fast: [f32; N_SLOTS],
    pub decays_slow: [f32; N_SLOTS],
    pub slow_split: [f32; N_SLOTS],
}

struct Layer {
    velocity: f32,
    notes: Vec<(f32, NoteParams)>, // sorted by note
}

pub struct Calibration {
    layers: Vec<Layer>, // sorted by velocity
}

impl Calibration {
    pub fn embedded() -> &'static Calibration {
        use std::sync::OnceLock;
        static CAL: OnceLock<Calibration> = OnceLock::new();
        CAL.get_or_init(|| {
            Calibration::parse(include_str!("../data/calibration.json"))
                .expect("embedded calibration must parse")
        })
    }

    pub fn parse(json: &str) -> Result<Calibration, Box<dyn std::error::Error>> {
        let raw: RawCalibration = serde_json::from_str(json)?;
        assert_eq!(raw.version, 9, "unknown calibration version");
        let mut layers: Vec<Layer> = raw
            .layers
            .into_values()
            .map(|notes| {
                let velocity = notes[0].velocity as f32;
                let mut entries: Vec<(f32, NoteParams)> = notes
                    .into_iter()
                    .map(|n| {
                        let to_array = |v: &[f32]| {
                            let mut a = [0f32; N_SLOTS];
                            a.copy_from_slice(&v[..N_SLOTS]);
                            a
                        };
                        (
                            n.note as f32,
                            NoteParams {
                                f0_cents: n.f0_cents,
                                b: n.b,
                                bed_db: n.bed_db,
                                bed_centroid_hz: n.bed_centroid_hz,
                                loudness_db: n.loudness_db,
                                attack_ms: n.attack_ms,
                                amps_db: to_array(&n.amps_db),
                                decays_fast: to_array(&n.decays_fast_db_s),
                                decays_slow: to_array(&n.decays_slow_db_s),
                                slow_split: to_array(&n.slow_split),
                            },
                        )
                    })
                    .collect();
                entries.sort_by(|a, b| a.0.total_cmp(&b.0));
                Layer { velocity, notes: entries }
            })
            .collect();
        layers.sort_by(|a, b| a.velocity.total_cmp(&b.velocity));
        Ok(Calibration { layers })
    }

    /// Interpolated parameters for any (note, velocity).
    pub fn lookup(&self, note: u8, velocity: u8) -> NoteParams {
        let vel = velocity as f32;
        let (lo, hi, w) = bracket(&self.layers, vel, |l| l.velocity);
        let a = lookup_in_layer(&self.layers[lo], note as f32);
        if lo == hi {
            return a;
        }
        let b = lookup_in_layer(&self.layers[hi], note as f32);
        lerp_params(&a, &b, w)
    }
}

fn lookup_in_layer(layer: &Layer, note: f32) -> NoteParams {
    let (lo, hi, w) = bracket(&layer.notes, note, |n| n.0);
    let a = &layer.notes[lo].1;
    if lo == hi {
        return clone_params(a);
    }
    lerp_params(a, &layer.notes[hi].1, w)
}

/// Find the bracketing pair around `x` in a sorted slice, returning
/// (lo_index, hi_index, weight toward hi). Clamps at the ends.
fn bracket<T>(items: &[T], x: f32, key: impl Fn(&T) -> f32) -> (usize, usize, f32) {
    if x <= key(&items[0]) {
        return (0, 0, 0.0);
    }
    for i in 0..items.len() - 1 {
        let (a, b) = (key(&items[i]), key(&items[i + 1]));
        if x <= b {
            return (i, i + 1, (x - a) / (b - a));
        }
    }
    (items.len() - 1, items.len() - 1, 0.0)
}

fn clone_params(p: &NoteParams) -> NoteParams {
    NoteParams {
        f0_cents: p.f0_cents,
        b: p.b,
        bed_db: p.bed_db,
        bed_centroid_hz: p.bed_centroid_hz,
        loudness_db: p.loudness_db,
        attack_ms: p.attack_ms,
        amps_db: p.amps_db,
        decays_fast: p.decays_fast,
        decays_slow: p.decays_slow,
        slow_split: p.slow_split,
    }
}

fn lerp_params(a: &NoteParams, b: &NoteParams, w: f32) -> NoteParams {
    let lerp_array = |x: &[f32; N_SLOTS], y: &[f32; N_SLOTS]| {
        let mut out = [0f32; N_SLOTS];
        for i in 0..N_SLOTS {
            out[i] = x[i] + w * (y[i] - x[i]);
        }
        out
    };
    NoteParams {
        f0_cents: a.f0_cents + w * (b.f0_cents - a.f0_cents),
        // B spans decades; interpolate in log space.
        b: (a.b.ln() + w * (b.b.ln() - a.b.ln())).exp(),
        bed_db: a.bed_db + w * (b.bed_db - a.bed_db),
        bed_centroid_hz: a.bed_centroid_hz + w * (b.bed_centroid_hz - a.bed_centroid_hz),
        loudness_db: a.loudness_db + w * (b.loudness_db - a.loudness_db),
        attack_ms: a.attack_ms + w * (b.attack_ms - a.attack_ms),
        amps_db: lerp_array(&a.amps_db, &b.amps_db),
        decays_fast: lerp_array(&a.decays_fast, &b.decays_fast),
        decays_slow: lerp_array(&a.decays_slow, &b.decays_slow),
        slow_split: lerp_array(&a.slow_split, &b.slow_split),
    }
}
