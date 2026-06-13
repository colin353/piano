//! Room simulation: a small early-reflection stage into an 8-line FDN
//! (Householder feedback) with per-line high-frequency damping.
//!
//! This is a *presentation* stage: the reference samples are nearly dry,
//! so the scoring path renders without it (the `note` command), while
//! MIDI renders and live playing get the room by default. Allocation
//! happens only at construction; `process` is real-time safe.

const FDN_LINES: usize = 8;

struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    fn new(len: usize) -> DelayLine {
        DelayLine { buf: vec![0.0; len.max(1)], pos: 0 }
    }

    #[inline]
    fn read(&self) -> f32 {
        self.buf[self.pos]
    }

    #[inline]
    fn write(&mut self, v: f32) {
        self.buf[self.pos] = v;
        self.pos += 1;
        if self.pos == self.buf.len() {
            self.pos = 0;
        }
    }
}

pub struct Reverb {
    predelay_l: DelayLine,
    predelay_r: DelayLine,
    early: Vec<(DelayLine, f32, bool)>, // (delay, gain, goes_left)
    fdn: [DelayLine; FDN_LINES],
    fdn_gain: [f32; FDN_LINES],
    damp_state: [f32; FDN_LINES],
    damp_coeff: f32,
    wet: f32,
}

impl Reverb {
    /// Construct with the shipped room, honoring PIANO_REVERB_RT60 /
    /// PIANO_REVERB_WET env overrides (used by the FAD optimizer).
    pub fn default_room(sample_rate: f32) -> Reverb {
        let get = |name: &str, default: f32| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };
        Reverb::new(
            sample_rate,
            get("PIANO_REVERB_RT60", 1.5),
            get("PIANO_REVERB_WET", 0.5),
        )
    }

    /// `rt60` in seconds, `wet` 0..1 (dry stays at unity).
    pub fn new(sample_rate: f32, rt60: f32, wet: f32) -> Reverb {
        let ms = |m: f32| (m / 1000.0 * sample_rate) as usize;
        // Mutually prime-ish delay lengths, 29..83 ms.
        let lens = [
            ms(29.7), ms(37.1), ms(41.1), ms(43.7),
            ms(53.0), ms(61.3), ms(71.9), ms(83.3),
        ];
        let mut fdn_gain = [0f32; FDN_LINES];
        for (i, &len) in lens.iter().enumerate() {
            fdn_gain[i] = 10f32.powf(-3.0 * len as f32 / (rt60 * sample_rate));
        }
        Reverb {
            predelay_l: DelayLine::new(ms(14.0)),
            predelay_r: DelayLine::new(ms(16.5)),
            early: [
                (11.3, 0.50, true), (17.2, 0.42, false), (23.1, 0.34, true),
                (29.9, 0.28, false), (38.7, 0.22, true), (47.3, 0.17, false),
            ]
            .into_iter()
            .map(|(m, g, left)| (DelayLine::new(ms(m)), g, left))
            .collect(),
            fdn: lens.map(DelayLine::new),
            fdn_gain,
            damp_state: [0.0; FDN_LINES],
            // One-pole lowpass in the loop: rooms absorb HF faster.
            damp_coeff: (-std::f32::consts::TAU * 4800.0 / sample_rate).exp(),
            wet,
        }
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        for i in 0..left.len() {
            let dry_l = left[i];
            let dry_r = right[i];
            let mono = 0.5 * (dry_l + dry_r);

            self.predelay_l.write(mono);
            self.predelay_r.write(mono);
            let pre_l = self.predelay_l.read();
            let pre_r = self.predelay_r.read();

            // Early reflections.
            let mut er_l = 0.0;
            let mut er_r = 0.0;
            for (line, gain, goes_left) in &mut self.early {
                let out = line.read();
                line.write(mono);
                if *goes_left {
                    er_l += out * *gain;
                } else {
                    er_r += out * *gain;
                }
            }

            // FDN with Householder feedback: y_i = x_i - (2/N) * sum(x).
            let mut taps = [0f32; FDN_LINES];
            let mut sum = 0.0;
            for (k, line) in self.fdn.iter().enumerate() {
                taps[k] = line.read();
                sum += taps[k];
            }
            let h = 2.0 / FDN_LINES as f32 * sum;
            for k in 0..FDN_LINES {
                let inject = if k % 2 == 0 { pre_l + er_l } else { pre_r + er_r };
                let mut v = (taps[k] - h) * self.fdn_gain[k] + inject * 0.35;
                // HF damping inside the loop.
                self.damp_state[k] = self.damp_state[k] * self.damp_coeff
                    + v * (1.0 - self.damp_coeff);
                v = self.damp_state[k];
                self.fdn[k].write(v);
            }
            // Alternate-sign taps decorrelate the stereo tail.
            let tail_l = taps[0] - taps[2] + taps[4] - taps[6];
            let tail_r = taps[1] - taps[3] + taps[5] - taps[7];

            left[i] = dry_l + self.wet * (er_l * 0.55 + tail_l * 0.75);
            right[i] = dry_r + self.wet * (er_r * 0.55 + tail_r * 0.75);
        }
    }
}
