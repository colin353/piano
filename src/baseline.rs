//! The deliberately dumb baseline: each note is a sine wave at the
//! fundamental frequency, held for one second with a short fade in/out to
//! avoid clicks. Velocity maps linearly to amplitude. This exists purely to
//! anchor the bottom of the score ladder.

use crate::{Synth, midi_note_freq};

const NOTE_SECONDS: f32 = 1.0;
const FADE_SECONDS: f32 = 0.005;

struct Voice {
    phase: f32,
    phase_inc: f32,
    amp: f32,
    age: u32,
    total: u32,
    fade: u32,
}

pub struct BaselineSynth {
    sample_rate: f32,
    voices: Vec<Voice>,
}

impl BaselineSynth {
    pub fn new(sample_rate: f32) -> Self {
        Self {
            sample_rate,
            // Preallocated so note_on never allocates on the audio thread.
            voices: Vec::with_capacity(128),
        }
    }
}

impl Synth for BaselineSynth {
    fn note_on(&mut self, note: u8, velocity: u8) {
        if self.voices.len() == self.voices.capacity() {
            return;
        }
        self.voices.push(Voice {
            phase: 0.0,
            phase_inc: midi_note_freq(note) / self.sample_rate,
            amp: 0.2 * velocity as f32 / 127.0,
            age: 0,
            total: (NOTE_SECONDS * self.sample_rate) as u32,
            fade: (FADE_SECONDS * self.sample_rate) as u32,
        });
    }

    fn note_off(&mut self, _note: u8) {}

    fn set_sustain(&mut self, _position: f32) {}

    fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        left.fill(0.0);
        for voice in &mut self.voices {
            for sample in left.iter_mut() {
                if voice.age >= voice.total {
                    break;
                }
                let fade_in = (voice.age as f32 / voice.fade as f32).min(1.0);
                let fade_out =
                    ((voice.total - voice.age) as f32 / voice.fade as f32).min(1.0);
                *sample += (voice.phase * std::f32::consts::TAU).sin()
                    * voice.amp
                    * fade_in.min(fade_out);
                voice.phase = (voice.phase + voice.phase_inc).fract();
                voice.age += 1;
            }
        }
        self.voices.retain(|v| v.age < v.total);
        right.copy_from_slice(left);
    }
}
