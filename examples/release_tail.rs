use piano::events::{Event, EventKind};
use piano::render::render_events;

fn tail(label: &str) {
    let mut synth = piano::create_synth("modal-v2", 44100.0).unwrap();
    let events = vec![
        Event { time: 0.0, kind: EventKind::NoteOn { note: 48, velocity: 110 } },
        Event { time: 0.0, kind: EventKind::NoteOn { note: 55, velocity: 105 } },
        Event { time: 0.0, kind: EventKind::NoteOn { note: 64, velocity: 100 } },
        Event { time: 1.0, kind: EventKind::NoteOff { note: 48 } },
        Event { time: 1.0, kind: EventKind::NoteOff { note: 55 } },
        Event { time: 1.0, kind: EventKind::NoteOff { note: 64 } },
    ];
    let audio = render_events(synth.as_mut(), &events, 44100.0, 3.0);
    let rms = |t: f64| {
        let i = (t * 44100.0) as usize * 2;
        let seg = &audio[i..(i + 8820).min(audio.len())];
        20.0 * (seg.iter().map(|s| s * s).sum::<f32>() / seg.len() as f32)
            .sqrt()
            .log10()
    };
    println!(
        "{label}: held {:.1} | +0.2s {:.1} | +0.5s {:.1} | +1.0s {:.1} dB",
        rms(0.9), rms(1.2), rms(1.5), rms(2.0)
    );
}

fn main() {
    tail("chord release");
}
