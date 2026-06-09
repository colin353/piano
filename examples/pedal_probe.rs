// quick manual probe via render_events
use piano::events::{Event, EventKind};
use piano::render::render_events;

fn rms_at(audio: &[f32], sr: f32, t: f64) -> f32 {
    let i = (t * sr as f64) as usize * 2;
    let seg = &audio[i..(i + (sr * 0.1) as usize * 2).min(audio.len())];
    (seg.iter().map(|s| s * s).sum::<f32>() / seg.len() as f32).sqrt()
}

fn main() {
    for (label, pedal) in [("no pedal", 0.0f32), ("half", 0.6), ("full", 1.0)] {
        let mut synth = piano::create_synth("modal-v2", 44100.0).unwrap();
        let events = vec![
            Event { time: 0.0, kind: EventKind::Sustain { position: pedal } },
            Event { time: 0.1, kind: EventKind::NoteOn { note: 48, velocity: 100 } },
            Event { time: 0.6, kind: EventKind::NoteOff { note: 48 } },
        ];
        let audio = render_events(synth.as_mut(), &events, 44100.0, 3.0);
        let pre = 20.0 * rms_at(&audio, 44100.0, 0.5).log10();
        let post1 = 20.0 * rms_at(&audio, 44100.0, 1.1).log10();
        let post2 = 20.0 * rms_at(&audio, 44100.0, 2.0).log10();
        println!("{label:9} pre {pre:6.1} dB | +0.5s {post1:6.1} | +1.4s {post2:6.1}");
    }
    // una corda A/B
    for (label, una) in [("normal", 0.0f32), ("una corda", 1.0)] {
        let mut synth = piano::create_synth("modal-v2", 44100.0).unwrap();
        let events = vec![
            Event { time: 0.0, kind: EventKind::Control { controller: 67, value: una } },
            Event { time: 0.1, kind: EventKind::NoteOn { note: 60, velocity: 100 } },
        ];
        let audio = render_events(synth.as_mut(), &events, 44100.0, 2.0);
        let level = 20.0 * rms_at(&audio, 44100.0, 0.3).log10();
        println!("{label:9} level at 0.3s: {level:6.1} dB");
    }
}
