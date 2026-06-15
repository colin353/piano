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
            let mut events =
                piano::events::load_midi(path.as_ref()).expect("failed to load midi file");
            if args.iter().any(|a| a == "--auto-pedal")
                && !events.iter().any(|e| matches!(e.kind, EventKind::Sustain { .. }))
            {
                // Typeset MIDI (Mutopia etc.) rarely carries CC64. Heuristic
                // pedaling: re-pedal at every bass note onset.
                let mut pedal = Vec::new();
                for e in &events {
                    if let EventKind::NoteOn { note, .. } = e.kind {
                        if note < 48 {
                            pedal.push(Event {
                                time: (e.time - 0.02).max(0.0),
                                kind: EventKind::Sustain { position: 0.0 },
                            });
                            pedal.push(Event {
                                time: e.time + 0.02,
                                kind: EventKind::Sustain { position: 1.0 },
                            });
                        }
                    }
                }
                eprintln!("auto-pedal: {} re-pedal points", pedal.len() / 2);
                events.extend(pedal);
                events.sort_by(|a, b| a.time.total_cmp(&b.time));
            }
            if args.iter().any(|a| a == "--legato") {
                // Typeset MIDI has exactly quantized note lengths with zero
                // overlap; human fingers overlap adjacent notes. Extend
                // each note-off ~70 ms into the gap, but never past a
                // re-strike of the same note.
                let note_ons: Vec<(f64, u8)> = events
                    .iter()
                    .filter_map(|e| match e.kind {
                        EventKind::NoteOn { note, .. } => Some((e.time, note)),
                        _ => None,
                    })
                    .collect();
                for e in &mut events {
                    if let EventKind::NoteOff { note } = e.kind {
                        let next_strike = note_ons
                            .iter()
                            .filter(|(t, n)| *n == note && *t > e.time)
                            .map(|(t, _)| *t)
                            .fold(f64::INFINITY, f64::min);
                        e.time = (e.time + 0.07).min(next_strike - 0.01);
                    }
                }
                events.sort_by(|a, b| a.time.total_cmp(&b.time));
            }
            eprintln!("{} events over {:.1}s", events.len(),
                events.last().map(|e| e.time).unwrap_or(0.0));
            let mut audio = render_events(synth.as_mut(), &events, sample_rate, 6.0);
            // Room reverb by default (presentation only — scoring uses the
            // dry `note` path); --dry disables.
            if !args.iter().any(|a| a == "--dry") {
                let mut room = piano::reverb::Reverb::default_room(sample_rate);
                let mut eq = piano::eq::MasterEq::from_env(sample_rate);
                let n = audio.len() / 2;
                let mut l: Vec<f32> = (0..n).map(|i| audio[2 * i]).collect();
                let mut r: Vec<f32> = (0..n).map(|i| audio[2 * i + 1]).collect();
                // EQ the instrument before the room, so the tail reverberates
                // the brightened tone rather than getting brightened itself;
                // master-bus compress/saturate last.
                eq.process(&mut l, &mut r);
                room.process(&mut l, &mut r);
                piano::comp::Compressor::from_env(sample_rate).process(&mut l, &mut r);
                for i in 0..n {
                    audio[2 * i] = l[i];
                    audio[2 * i + 1] = r[i];
                }
            }
            // Offline renders get peak normalization to -1 dBFS instead of
            // letting the WAV writer hard-clip loud passages.
            let peak = audio.iter().fold(0f32, |m, &s| m.max(s.abs()));
            if peak > 0.0 {
                let gain = 0.891 / peak;
                if gain < 1.0 {
                    eprintln!("peak {peak:.2} — applying {:.1} dB", 20.0 * gain.log10());
                }
                for s in &mut audio {
                    *s *= gain.min(1.0);
                }
            }
            let out = out.unwrap_or_else(|| usage());
            write_wav(&out, &audio, sample_rate as u32).expect("failed to write wav");
            println!("wrote {}", out.display());
        }
        "sympathetic-demo" => {
            // Hold C3 silently (string open, barely audible), strike C2
            // staccato: C3's string should ring on after C2 is damped.
            let events = vec![
                Event { time: 0.0, kind: EventKind::NoteOn { note: 48, velocity: 1 } },
                Event { time: 0.5, kind: EventKind::NoteOn { note: 36, velocity: 112 } },
                Event { time: 0.8, kind: EventKind::NoteOff { note: 36 } },
                Event { time: 6.0, kind: EventKind::NoteOff { note: 48 } },
            ];
            let audio = render_events(synth.as_mut(), &events, sample_rate, 1.0);
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

/// Positional arguments after the subcommand (skipping flags; boolean
/// flags take no value).
fn positionals(args: &[String]) -> Vec<&String> {
    const BOOLEAN_FLAGS: &[&str] = &["--auto-pedal", "--dry", "--legato"];
    let mut out = Vec::new();
    let mut i = 1;
    while i < args.len() {
        if BOOLEAN_FLAGS.contains(&args[i].as_str()) {
            i += 1;
        } else if args[i].starts_with('-') {
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
