//! Offline rendering: drive a [`Synth`] through a timed event list and
//! capture the output, exactly as a real-time host would but unclocked.

use crate::Synth;
use crate::events::{Event, EventKind};
use std::path::Path;

const BLOCK: usize = 256;

/// Render an event timeline to interleaved stereo samples.
///
/// `tail` seconds of silence-driven processing are appended after the last
/// event so decays and pedal releases ring out naturally.
pub fn render_events(
    synth: &mut dyn Synth,
    events: &[Event],
    sample_rate: f32,
    tail: f64,
) -> Vec<f32> {
    let end_time = events.last().map(|e| e.time).unwrap_or(0.0) + tail;
    let total_samples = (end_time * sample_rate as f64).ceil() as usize;

    let mut out = Vec::with_capacity(total_samples * 2);
    let mut left = [0f32; BLOCK];
    let mut right = [0f32; BLOCK];

    let mut event_idx = 0;
    let mut cursor: usize = 0;
    while cursor < total_samples {
        // Dispatch all events that fall at or before the current sample.
        while event_idx < events.len() {
            let event_sample = (events[event_idx].time * sample_rate as f64) as usize;
            if event_sample > cursor {
                break;
            }
            match events[event_idx].kind {
                EventKind::NoteOn { note, velocity } => synth.note_on(note, velocity),
                EventKind::NoteOff { note } => synth.note_off(note),
                EventKind::Sustain { position } => synth.set_sustain(position),
                EventKind::Control { controller, value } => synth.set_control(controller, value),
            }
            event_idx += 1;
        }

        // Process up to the next event or the end, in BLOCK-sized chunks.
        let next_boundary = if event_idx < events.len() {
            ((events[event_idx].time * sample_rate as f64) as usize).min(total_samples)
        } else {
            total_samples
        };
        let n = (next_boundary - cursor).clamp(1, BLOCK);
        synth.process(&mut left[..n], &mut right[..n]);
        for i in 0..n {
            out.push(left[i]);
            out.push(right[i]);
        }
        cursor += n;
    }
    out
}

/// Write interleaved stereo f32 samples as a 16-bit PCM WAV file.
pub fn write_wav(
    path: &Path,
    interleaved: &[f32],
    sample_rate: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in interleaved {
        writer.write_sample((s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;
    Ok(())
}
