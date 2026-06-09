//! Physics-based piano synthesizer.
//!
//! The core abstraction is [`Synth`]: a streaming, real-time-capable audio
//! generator driven by note events. Offline rendering (WAV files, MIDI
//! playback) is the same core run faster than real time — see [`render`].

pub mod baseline;
pub mod calibration;
pub mod events;
pub mod modal;
pub mod modal2;
pub mod render;

/// A streaming piano synthesizer.
///
/// Implementations must be suitable for a real-time audio thread: no
/// allocation, locking, or I/O inside [`Synth::process`].
pub trait Synth: Send {
    /// Begin a note. `note` is a MIDI note number (21..=108 for a piano),
    /// `velocity` is MIDI velocity (1..=127).
    fn note_on(&mut self, note: u8, velocity: u8);

    /// Release a note (the key, not necessarily the sound — sustain pedal
    /// and string decay decide when sound actually stops).
    fn note_off(&mut self, note: u8);

    /// Sustain (damper) pedal position, 0.0 = up, 1.0 = fully down.
    fn set_sustain(&mut self, position: f32);

    /// Generate the next `left.len()` samples of stereo audio, overwriting
    /// the buffers. Both slices are the same length.
    fn process(&mut self, left: &mut [f32], right: &mut [f32]);
}

/// Construct a synth by name. This is the registry the render CLI and the
/// scoring harness use to select an implementation.
pub fn create_synth(name: &str, sample_rate: f32) -> Option<Box<dyn Synth>> {
    match name {
        "baseline" => Some(Box::new(baseline::BaselineSynth::new(sample_rate))),
        "modal-v1" => Some(Box::new(modal::ModalSynth::new(sample_rate))),
        "modal-v2" => Some(Box::new(modal2::ModalV2::new(sample_rate))),
        _ => None,
    }
}

/// Names of all registered synths, for CLI help and harness enumeration.
pub const SYNTH_NAMES: &[&str] = &["baseline", "modal-v1", "modal-v2"];

/// Equal-tempered frequency of a MIDI note, A4 (note 69) = 440 Hz.
pub fn midi_note_freq(note: u8) -> f32 {
    440.0 * 2f32.powf((note as f32 - 69.0) / 12.0)
}
