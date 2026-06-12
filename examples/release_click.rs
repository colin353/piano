// Hunt release discontinuities: render loud-note-release and fast
// same-key repetition, then scan a high-passed version for spikes.
use piano::events::{Event, EventKind};
use piano::render::render_events;

fn scan(label: &str, events: Vec<Event>) {
    let mut synth = piano::create_synth("modal-v2", 44100.0).unwrap();
    let audio = render_events(synth.as_mut(), &events, 44100.0, 2.0);
    // mono + first-difference (crude highpass): clicks = outlier samples
    let mono: Vec<f32> = audio.chunks(2).map(|c| (c[0] + c[1]) * 0.5).collect();
    let diff: Vec<f32> = mono.windows(2).map(|w| w[1] - w[0]).collect();
    // robust scale from the body of the signal
    let mut sorted: Vec<f32> = diff.iter().map(|d| d.abs()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let p99 = sorted[(sorted.len() as f32 * 0.99) as usize].max(1e-9);
    let mut worst: Vec<(f32, f32)> = Vec::new();
    for (i, d) in diff.iter().enumerate() {
        if d.abs() > p99 * 8.0 {
            worst.push((i as f32 / 44100.0, d.abs() / p99));
        }
    }
    worst.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!("{label}: {} spike samples (>8x p99 slope)", worst.len());
    for (t, ratio) in worst.iter().take(5) {
        println!("   at {t:.3}s: {ratio:.0}x p99");
    }
}

fn main() {
    scan("loud release", vec![
        Event { time: 0.0, kind: EventKind::NoteOn { note: 60, velocity: 120 } },
        Event { time: 1.0, kind: EventKind::NoteOff { note: 60 } },
    ]);
    scan("fast repetition (quiet)", (0..8).flat_map(|i| {
        let t = i as f64 * 0.12;
        vec![
            Event { time: t, kind: EventKind::NoteOn { note: 64, velocity: 45 } },
            Event { time: t + 0.09, kind: EventKind::NoteOff { note: 64 } },
        ]
    }).collect());
}
// (extended by exp 26) — measure post-release tail with/without body
