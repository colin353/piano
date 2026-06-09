//! Live playable piano: MIDI keyboard in, low-latency audio out.
//!
//! Usage:
//!   piano-live [--synth modal-v2] [--midi <name substring>] [--buffer 128]
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
        .unwrap_or(128);

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
    let mut room = (!dry).then(|| piano::reverb::Reverb::new(sample_rate as f32, 1.8, 0.55));

    let stream = device
        .build_output_stream(
            &config,
            move |out: &mut [f32], _| {
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
                    if let Some(room) = room.as_mut() {
                        room.process(&mut left[..n], &mut right[..n]);
                    }
                    for (i, frame) in frames.chunks_mut(channels).enumerate() {
                        frame[0] = left[i];
                        if channels > 1 {
                            frame[1] = right[i];
                        }
                    }
                }
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
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
