//! Timed note events for offline rendering, and MIDI file loading.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventKind {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    Sustain { position: f32 },
    Control { controller: u8, value: f32 },
}

#[derive(Debug, Clone, Copy)]
pub struct Event {
    /// Seconds from the start of the timeline.
    pub time: f64,
    pub kind: EventKind,
}

/// Load a Standard MIDI File into a flat, time-sorted event list.
/// All tracks are merged; tempo changes are honored.
pub fn load_midi(path: &Path) -> Result<Vec<Event>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    let smf = midly::Smf::parse(&bytes)?;

    let ticks_per_beat = match smf.header.timing {
        midly::Timing::Metrical(t) => t.as_int() as f64,
        midly::Timing::Timecode(fps, sub) => {
            // Timecode: absolute time, no tempo dependence.
            let ticks_per_sec = fps.as_f32() as f64 * sub as f64;
            return Ok(collect_timecode(&smf, ticks_per_sec));
        }
    };

    // Merge tracks into (abs_tick, event) pairs, collecting tempo changes.
    let mut tempo_changes: Vec<(u64, f64)> = vec![(0, 500_000.0)]; // default 120 bpm
    let mut raw: Vec<(u64, EventKind)> = Vec::new();
    for track in &smf.tracks {
        let mut tick: u64 = 0;
        for ev in track {
            tick += ev.delta.as_int() as u64;
            match ev.kind {
                midly::TrackEventKind::Meta(midly::MetaMessage::Tempo(us)) => {
                    tempo_changes.push((tick, us.as_int() as f64));
                }
                midly::TrackEventKind::Midi { message, .. } => {
                    if let Some(kind) = midi_message_to_event(message) {
                        raw.push((tick, kind));
                    }
                }
                _ => {}
            }
        }
    }
    tempo_changes.sort_by_key(|&(t, _)| t);
    raw.sort_by_key(|&(t, _)| t);

    // Convert ticks to seconds by walking the tempo map.
    let mut events = Vec::with_capacity(raw.len());
    let mut tempo_idx = 0;
    let mut seg_start_tick: u64 = 0;
    let mut seg_start_time: f64 = 0.0;
    let mut us_per_beat = tempo_changes[0].1;
    for (tick, kind) in raw {
        while tempo_idx + 1 < tempo_changes.len() && tempo_changes[tempo_idx + 1].0 <= tick {
            tempo_idx += 1;
            let (change_tick, new_tempo) = tempo_changes[tempo_idx];
            seg_start_time +=
                (change_tick - seg_start_tick) as f64 / ticks_per_beat * us_per_beat / 1e6;
            seg_start_tick = change_tick;
            us_per_beat = new_tempo;
        }
        let time = seg_start_time
            + (tick - seg_start_tick) as f64 / ticks_per_beat * us_per_beat / 1e6;
        events.push(Event { time, kind });
    }
    Ok(events)
}

fn collect_timecode(smf: &midly::Smf, ticks_per_sec: f64) -> Vec<Event> {
    let mut events = Vec::new();
    for track in &smf.tracks {
        let mut tick: u64 = 0;
        for ev in track {
            tick += ev.delta.as_int() as u64;
            if let midly::TrackEventKind::Midi { message, .. } = ev.kind {
                if let Some(kind) = midi_message_to_event(message) {
                    events.push(Event {
                        time: tick as f64 / ticks_per_sec,
                        kind,
                    });
                }
            }
        }
    }
    events.sort_by(|a, b| a.time.total_cmp(&b.time));
    events
}

fn midi_message_to_event(message: midly::MidiMessage) -> Option<EventKind> {
    match message {
        midly::MidiMessage::NoteOn { key, vel } => {
            if vel.as_int() == 0 {
                Some(EventKind::NoteOff { note: key.as_int() })
            } else {
                Some(EventKind::NoteOn {
                    note: key.as_int(),
                    velocity: vel.as_int(),
                })
            }
        }
        midly::MidiMessage::NoteOff { key, .. } => Some(EventKind::NoteOff { note: key.as_int() }),
        midly::MidiMessage::Controller { controller, value } if controller.as_int() == 64 => {
            Some(EventKind::Sustain {
                position: value.as_int() as f32 / 127.0,
            })
        }
        midly::MidiMessage::Controller { controller, value } if controller.as_int() == 67 => {
            Some(EventKind::Control {
                controller: 67,
                value: value.as_int() as f32 / 127.0,
            })
        }
        _ => None,
    }
}
