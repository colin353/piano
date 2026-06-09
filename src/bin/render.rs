//! CLI for offline rendering and benchmarking.
//!
//! Usage:
//!   piano-render note <midi_note> <velocity> -o out.wav [--synth NAME] [--dur SECS] [--sr HZ]
//!   piano-render midi <file.mid> -o out.wav [--synth NAME] [--sr HZ]
//!   piano-render bench [--synth NAME] [--sr HZ]

use piano::events::{Event, EventKind};
use piano::render::{render_events, write_wav};
use std::path::PathBuf;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        usage();
    };

    let synth_name = flag_value(&args, "--synth").unwrap_or_else(|| "baseline".into());
    let sample_rate: f32 = flag_value(&args, "--sr")
        .map(|s| s.parse().expect("--sr must be a number"))
        .unwrap_or(44_100.0);
    let out = flag_value(&args, "-o").map(PathBuf::from);

    let Some(mut synth) = piano::create_synth(&synth_name, sample_rate) else {
        eprintln!(
            "unknown synth '{synth_name}'; available: {}",
            piano::SYNTH_NAMES.join(", ")
        );
        exit(1);
    };

    match command.as_str() {
        "note" => {
            let positional = positionals(&args);
            let [note, velocity] = positional[..] else {
                usage();
            };
            let note: u8 = note.parse().expect("note must be 0-127");
            let velocity: u8 = velocity.parse().expect("velocity must be 1-127");
            let dur: f64 = flag_value(&args, "--dur")
                .map(|s| s.parse().expect("--dur must be a number"))
                .unwrap_or(8.0);
            let events = vec![
                Event { time: 0.0, kind: EventKind::NoteOn { note, velocity } },
                Event { time: dur, kind: EventKind::NoteOff { note } },
            ];
            let audio = render_events(synth.as_mut(), &events, sample_rate, 4.0);
            let out = out.unwrap_or_else(|| usage());
            write_wav(&out, &audio, sample_rate as u32).expect("failed to write wav");
            println!("wrote {}", out.display());
        }
        "midi" => {
            let positional = positionals(&args);
            let [path] = positional[..] else {
                usage();
            };
            let events =
                piano::events::load_midi(path.as_ref()).expect("failed to load midi file");
            eprintln!("{} events over {:.1}s", events.len(),
                events.last().map(|e| e.time).unwrap_or(0.0));
            let audio = render_events(synth.as_mut(), &events, sample_rate, 6.0);
            let out = out.unwrap_or_else(|| usage());
            write_wav(&out, &audio, sample_rate as u32).expect("failed to write wav");
            println!("wrote {}", out.display());
        }
        "bench" => {
            // Dense load: 16 simultaneous notes retriggered every 250 ms for
            // 30 seconds with pedal down, approximating worst-case playing.
            let mut events = vec![Event { time: 0.0, kind: EventKind::Sustain { position: 1.0 } }];
            for step in 0..120 {
                let t = step as f64 * 0.25;
                for v in 0..16 {
                    let note = 21 + ((step * 16 + v) * 7) % 88;
                    events.push(Event {
                        time: t,
                        kind: EventKind::NoteOn { note: note as u8, velocity: 100 },
                    });
                    events.push(Event {
                        time: t + 0.2,
                        kind: EventKind::NoteOff { note: note as u8 },
                    });
                }
            }
            events.sort_by(|a, b| a.time.total_cmp(&b.time));
            let start = std::time::Instant::now();
            let audio = render_events(synth.as_mut(), &events, sample_rate, 2.0);
            let elapsed = start.elapsed().as_secs_f64();
            let rendered = audio.len() as f64 / 2.0 / sample_rate as f64;
            println!(
                "synth={synth_name} rendered {rendered:.1}s in {elapsed:.3}s — \
                 real-time factor {:.1}x",
                rendered / elapsed
            );
        }
        _ => usage(),
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

/// Positional arguments after the subcommand (skipping flag/value pairs).
fn positionals(args: &[String]) -> Vec<&String> {
    let mut out = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if args[i].starts_with('-') {
            i += 2;
        } else {
            out.push(&args[i]);
            i += 1;
        }
    }
    out
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  piano-render note <midi_note> <velocity> -o out.wav [--synth NAME] [--dur SECS] [--sr HZ]\n  piano-render midi <file.mid> -o out.wav [--synth NAME] [--sr HZ]\n  piano-render bench [--synth NAME] [--sr HZ]"
    );
    exit(2);
}
