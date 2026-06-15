//! WASM bindings for the in-browser tuning app (web/). Wraps the synth
//! plus the master EQ and room reverb behind a small, JS-friendly API:
//! note events, a `set_param(name, value)` knob setter keyed by the same
//! PIANO_* suffixes the CLI uses, and a block renderer.

use crate::comp::Compressor;
use crate::eq::MasterEq;
use crate::modal2::ModalV2;
use crate::reverb::Reverb;
use crate::Synth;
use wasm_bindgen::prelude::*;

/// EQ/reverb knob values the engine owns (the synth owns the rest).
struct StageParams {
    eq_low_db: f32,
    eq_low_hz: f32,
    eq_peak_db: f32,
    eq_peak_hz: f32,
    eq_peak_q: f32,
    eq_high_db: f32,
    eq_high_hz: f32,
    reverb_rt60: f32,
    reverb_wet: f32,
    comp_thresh_db: f32,
    comp_ratio: f32,
    comp_attack_ms: f32,
    comp_release_ms: f32,
    comp_makeup_db: f32,
    comp_drive: f32,
    eq_on: bool,
    reverb_on: bool,
    comp_on: bool,
}

impl Default for StageParams {
    fn default() -> StageParams {
        StageParams {
            eq_low_db: -2.0,
            eq_low_hz: 300.0,
            eq_peak_db: 4.0,
            eq_peak_hz: 5000.0,
            eq_peak_q: 0.8,
            eq_high_db: 6.0,
            eq_high_hz: 2500.0,
            reverb_rt60: 1.5,
            reverb_wet: 0.35,
            comp_thresh_db: -22.0,
            comp_ratio: 2.5,
            comp_attack_ms: 15.0,
            comp_release_ms: 160.0,
            comp_makeup_db: 3.0,
            comp_drive: 0.12,
            eq_on: true,
            reverb_on: true,
            comp_on: true,
        }
    }
}

#[wasm_bindgen]
pub struct Engine {
    synth: ModalV2,
    eq: MasterEq,
    reverb: Reverb,
    comp: Compressor,
    sr: f32,
    p: StageParams,
    left: Vec<f32>,
    right: Vec<f32>,
}

#[wasm_bindgen]
impl Engine {
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: f32) -> Engine {
        let p = StageParams::default();
        Engine {
            synth: ModalV2::new(sample_rate),
            eq: build_eq(sample_rate, &p),
            reverb: Reverb::new(sample_rate, p.reverb_rt60, p.reverb_wet),
            comp: build_comp(sample_rate, &p),
            sr: sample_rate,
            p,
            left: vec![0.0; 2048],
            right: vec![0.0; 2048],
        }
    }

    pub fn note_on(&mut self, note: u8, velocity: u8) {
        self.synth.note_on(note, velocity);
    }
    pub fn note_off(&mut self, note: u8) {
        self.synth.note_off(note);
    }
    pub fn set_sustain(&mut self, position: f32) {
        self.synth.set_sustain(position);
    }
    pub fn set_control(&mut self, controller: u8, value: f32) {
        self.synth.set_control(controller, value);
    }

    /// Set any knob by its PIANO_* suffix. EQ/reverb knobs rebuild their
    /// stage; everything else forwards to the synth.
    pub fn set_param(&mut self, name: &str, value: f32) {
        match name {
            "EQ_LOW_DB" => self.p.eq_low_db = value,
            "EQ_LOW_HZ" => self.p.eq_low_hz = value,
            "EQ_PEAK_DB" => self.p.eq_peak_db = value,
            "EQ_PEAK_HZ" => self.p.eq_peak_hz = value,
            "EQ_PEAK_Q" => self.p.eq_peak_q = value,
            "EQ_HIGH_DB" => self.p.eq_high_db = value,
            "EQ_HIGH_HZ" => self.p.eq_high_hz = value,
            "EQ_ON" => self.p.eq_on = value >= 0.5,
            "REVERB_RT60" => self.p.reverb_rt60 = value,
            "REVERB_WET" => self.p.reverb_wet = value,
            "REVERB_ON" => self.p.reverb_on = value >= 0.5,
            "COMP_THRESH_DB" => self.p.comp_thresh_db = value,
            "COMP_RATIO" => self.p.comp_ratio = value,
            "COMP_ATTACK_MS" => self.p.comp_attack_ms = value,
            "COMP_RELEASE_MS" => self.p.comp_release_ms = value,
            "COMP_MAKEUP_DB" => self.p.comp_makeup_db = value,
            "COMP_DRIVE" => self.p.comp_drive = value,
            "COMP_ON" => self.p.comp_on = value >= 0.5,
            other => {
                self.synth.set_param(other, value);
                return;
            }
        }
        // An EQ/reverb/comp knob changed — rebuild the affected stage.
        if name.starts_with("EQ_") {
            self.eq = build_eq(self.sr, &self.p);
        } else if name.starts_with("COMP_") {
            self.comp = build_comp(self.sr, &self.p);
        } else {
            self.reverb = Reverb::new(self.sr, self.p.reverb_rt60, self.p.reverb_wet);
        }
    }

    /// Render `frames` samples; the host then reads left_ptr()/right_ptr().
    pub fn render(&mut self, frames: usize) {
        if self.left.len() < frames {
            self.left.resize(frames, 0.0);
            self.right.resize(frames, 0.0);
        }
        let l = &mut self.left[..frames];
        let r = &mut self.right[..frames];
        self.synth.process(l, r);
        // Match the CLI chain: EQ the instrument, then room, then master comp.
        if self.p.eq_on {
            self.eq.process(l, r);
        }
        if self.p.reverb_on {
            self.reverb.process(l, r);
        }
        if self.p.comp_on {
            self.comp.process(l, r);
        }
    }

    pub fn left_ptr(&self) -> *const f32 {
        self.left.as_ptr()
    }
    pub fn right_ptr(&self) -> *const f32 {
        self.right.as_ptr()
    }
}

fn build_eq(sr: f32, p: &StageParams) -> MasterEq {
    MasterEq::with_params(
        sr, p.eq_low_db, p.eq_low_hz, p.eq_peak_db, p.eq_peak_hz, p.eq_peak_q, p.eq_high_db,
        p.eq_high_hz,
    )
}

fn build_comp(sr: f32, p: &StageParams) -> Compressor {
    Compressor::with_params(
        sr, p.comp_thresh_db, p.comp_ratio, p.comp_attack_ms, p.comp_release_ms, p.comp_makeup_db,
        p.comp_drive, p.comp_on,
    )
}
