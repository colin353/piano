//! Live playable piano: MIDI keyboard in, low-latency audio out.
//!
//! Usage:
//!   piano-live [--synth modal-v2] [--midi <name substring>] [--buffer 256]
//!              [--wet 0.93] [--rt60 2.25] [--dry]
//!   piano-live --list          # show MIDI inputs and audio devices
//!
//! MIDI events flow through a lock-free channel into the audio callback;
//! the synth never allocates on the audio thread.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc;

enum MidiEvent {
    NoteOn { note: u8, velocity: u8 },
    NoteOff { note: u8 },
    Sustain { position: f32 },
    Control { controller: u8, value: f32 },
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let host = cpal::default_host();

    if args.iter().any(|a| a == "--list") {
        println!("midi inputs:");
        let midi_in = midir::MidiInput::new("piano-list").expect("midi init failed");
        for port in midi_in.ports() {
            println!("  {}", midi_in.port_name(&port).unwrap_or_default());
        }
        println!("audio outputs:");
        if let Ok(devices) = host.output_devices() {
            for d in devices {
                println!("  {}", d.name().unwrap_or_default());
            }
        }
        return;
    }

    let synth_name = flag("--synth").unwrap_or_else(|| "modal-v2".into());
    let buffer: u32 = flag("--buffer")
        .map(|s| s.parse().expect("--buffer must be a number"))
        .unwrap_or(256);
    let wet: Option<f32> = flag("--wet").map(|s| s.parse().expect("--wet must be a number"));
    let rt60: Option<f32> = flag("--rt60").map(|s| s.parse().expect("--rt60 must be a number"));

    // Audio output.
    let device = host
        .default_output_device()
        .expect("no audio output device");
    let default_config = device.default_output_config().expect("no output config");
    let sample_rate = default_config.sample_rate().0;
    let channels = default_config.channels() as usize;
    let config = cpal::StreamConfig {
        channels: default_config.channels(),
        sample_rate: default_config.sample_rate(),
        buffer_size: cpal::BufferSize::Fixed(buffer),
    };

    let mut synth = piano::create_synth(&synth_name, sample_rate as f32)
        .unwrap_or_else(|| {
            eprintln!("unknown synth '{synth_name}'");
            std::process::exit(1);
        });

    let (tx, rx) = mpsc::channel::<MidiEvent>();

    // MIDI input (optional — without a device you just get silence).
    let midi_wanted = flag("--midi");
    let midi_in = midir::MidiInput::new("piano-live").expect("midi init failed");
    let port = {
        let ports = midi_in.ports();
        match &midi_wanted {
            Some(name) => ports.into_iter().find(|p| {
                midi_in
                    .port_name(p)
                    .map(|n| n.to_lowercase().contains(&name.to_lowercase()))
                    .unwrap_or(false)
            }),
            None => {
                let ports = midi_in.ports();
                // Skip the ALSA "Midi Through" loopback when auto-picking.
                ports.into_iter().find(|p| {
                    !midi_in
                        .port_name(p)
                        .unwrap_or_default()
                        .contains("Midi Through")
                })
            }
        }
    };
    let _midi_connection = match port {
        Some(port) => {
            let name = midi_in.port_name(&port).unwrap_or_default();
            println!("midi: {name}");
            Some(
                midi_in
                    .connect(
                        &port,
                        "piano-live-in",
                        move |_t, msg, _| {
                            let event = match msg {
                                [s, note, vel] if s & 0xF0 == 0x90 && *vel > 0 => {
                                    Some(MidiEvent::NoteOn { note: *note, velocity: *vel })
                                }
                                [s, note, _] if s & 0xF0 == 0x90 => {
                                    Some(MidiEvent::NoteOff { note: *note })
                                }
                                [s, note, _] if s & 0xF0 == 0x80 => {
                                    Some(MidiEvent::NoteOff { note: *note })
                                }
                                [s, 64, value] if s & 0xF0 == 0xB0 => {
                                    Some(MidiEvent::Sustain {
                                        position: *value as f32 / 127.0,
                                    })
                                }
                                [s, cc, value] if s & 0xF0 == 0xB0 => {
                                    Some(MidiEvent::Control {
                                        controller: *cc,
                                        value: *value as f32 / 127.0,
                                    })
                                }
                                _ => None,
                            };
                            if let Some(e) = event {
                                let _ = tx.send(e);
                            }
                        },
                        (),
                    )
                    .expect("midi connect failed"),
            )
        }
        None => {
            println!("midi: no input found (run with --list; synth will be silent)");
            None
        }
    };

    const BLOCK: usize = 128;
    let mut left = [0f32; BLOCK];
    let mut right = [0f32; BLOCK];
    let dry = args.iter().any(|a| a == "--dry");
    let mut room = (!dry).then(|| {
        if wet.is_some() || rt60.is_some() {
            piano::reverb::Reverb::new(sample_rate as f32, rt60.unwrap_or(1.5), wet.unwrap_or(0.35))
        } else {
            piano::reverb::Reverb::default_room(sample_rate as f32)
        }
    });
    let mut eq = (!dry).then(|| piano::eq::MasterEq::from_env(sample_rate as f32));
    let mut comp = (!dry).then(|| piano::comp::Compressor::from_env(sample_rate as f32));
    // Soft peak limiter: loud chords + the wet room can exceed full scale,
    // and raw clipping at the device is audible as 'tearing'. Instant
    // attack, ~80 ms release.
    let mut limiter_env = 1.0f32;
    let limiter_release = (-1.0 / (0.080 * sample_rate as f32)).exp();
    // Callback load telemetry, reported by the main thread.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    let load_max = Arc::new(AtomicU32::new(0));
    let load_max_cb = load_max.clone();

    let stream = device
        .build_output_stream(
            &config,
            move |out: &mut [f32], _| {
                let cb_start = std::time::Instant::now();
                while let Ok(e) = rx.try_recv() {
                    match e {
                        MidiEvent::NoteOn { note, velocity } => synth.note_on(note, velocity),
                        MidiEvent::NoteOff { note } => synth.note_off(note),
                        MidiEvent::Sustain { position } => synth.set_sustain(position),
                        MidiEvent::Control { controller, value } => {
                            synth.set_control(controller, value)
                        }
                    }
                }
                for frames in out.chunks_mut(BLOCK * channels) {
                    let n = frames.len() / channels;
                    synth.process(&mut left[..n], &mut right[..n]);
                    if let Some(eq) = eq.as_mut() {
                        eq.process(&mut left[..n], &mut right[..n]);
                    }
                    if let Some(room) = room.as_mut() {
                        room.process(&mut left[..n], &mut right[..n]);
                    }
                    if let Some(comp) = comp.as_mut() {
                        comp.process(&mut left[..n], &mut right[..n]);
                    }
                    for (i, frame) in frames.chunks_mut(channels).enumerate() {
                        let peak = left[i].abs().max(right[i].abs()) * limiter_env;
                        if peak > 0.95 {
                            limiter_env *= 0.95 / peak;
                        } else {
                            limiter_env = 1.0 - (1.0 - limiter_env) * limiter_release;
                        }
                        frame[0] = left[i] * limiter_env;
                        if channels > 1 {
                            frame[1] = right[i] * limiter_env;
                        }
                    }
                }
                // Load = callback time / buffer deadline, in 0.1% units.
                let frames_total = out.len() / channels;
                let deadline_us = frames_total as f32 / sample_rate as f32 * 1e6;
                let load =
                    (cb_start.elapsed().as_micros() as f32 / deadline_us * 1000.0) as u32;
                load_max_cb.fetch_max(load, Ordering::Relaxed);
            },
            |err| eprintln!("audio error: {err}"),
            None,
        )
        .expect("failed to build audio stream");

    stream.play().expect("failed to start audio stream");
    println!(
        "playing: synth={synth_name} {sample_rate} Hz, {buffer}-frame buffer \
         (~{:.1} ms). Ctrl-C to quit.",
        buffer as f32 / sample_rate as f32 * 1000.0
    );
    loop {
        std::thread::sleep(std::time::Duration::from_secs(5));
        let peak_load = load_max.swap(0, Ordering::Relaxed);
        if peak_load > 700 {
            eprintln!(
                "audio load peak {:.0}% of deadline — if you hear tearing, \
                 raise --buffer (e.g. 512)",
                peak_load as f32 / 10.0
            );
        }
    }
}
