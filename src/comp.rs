//! Master bus compressor + soft saturation — a presentation stage for the
//! "glued, slightly VHS/tape" character of a mastered recording. Feed-
//! forward peak detector with attack/release-smoothed gain reduction, soft
//! knee, makeup gain, and a blendable tanh saturation for analog warmth.
//! Stereo-linked (one gain for both channels, so the image is preserved).
//! Allocation-free; off by default in the scoring path.

pub struct Compressor {
    gr_db: f32, // current gain reduction, <= 0
    thresh_db: f32,
    slope: f32, // 1/ratio - 1 (<= 0)
    knee_db: f32,
    attack: f32,
    release: f32,
    makeup: f32, // linear
    drive: f32,  // 0..1 saturation blend
    active: bool,
}

impl Compressor {
    pub fn from_env(sr: f32) -> Compressor {
        let g = |n: &str, d: f32| {
            std::env::var(n).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
        };
        Compressor::with_params(
            sr,
            g("PIANO_COMP_THRESH_DB", -22.0),
            g("PIANO_COMP_RATIO", 2.5),
            g("PIANO_COMP_ATTACK_MS", 15.0),
            g("PIANO_COMP_RELEASE_MS", 160.0),
            g("PIANO_COMP_MAKEUP_DB", 3.0),
            g("PIANO_COMP_DRIVE", 0.12),
            g("PIANO_COMP_ON", 1.0) >= 0.5,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        sr: f32,
        thresh_db: f32,
        ratio: f32,
        attack_ms: f32,
        release_ms: f32,
        makeup_db: f32,
        drive: f32,
        on: bool,
    ) -> Compressor {
        let coeff = |ms: f32| 1.0 - (-1.0 / (ms.max(0.1) / 1000.0 * sr)).exp();
        Compressor {
            gr_db: 0.0,
            thresh_db,
            slope: 1.0 / ratio.max(1.0) - 1.0,
            knee_db: 6.0,
            attack: coeff(attack_ms),
            release: coeff(release_ms),
            makeup: 10f32.powf(makeup_db / 20.0),
            drive: drive.clamp(0.0, 1.0),
            active: on && (ratio > 1.0 || drive > 0.0),
        }
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.active {
            return;
        }
        let k = 1.0 + self.drive * 4.0;
        let kt = k.tanh();
        for i in 0..left.len() {
            let peak = left[i].abs().max(right[i].abs());
            let level_db = 20.0 * (peak + 1e-6).log10();
            // Soft-knee static curve -> target gain reduction (<= 0).
            let over = level_db - self.thresh_db;
            let target = if over <= -self.knee_db * 0.5 {
                0.0
            } else if over >= self.knee_db * 0.5 {
                self.slope * over
            } else {
                let x = over + self.knee_db * 0.5;
                self.slope * x * x / (2.0 * self.knee_db)
            };
            // Smooth the gain reduction: faster when clamping down.
            let c = if target < self.gr_db { self.attack } else { self.release };
            self.gr_db += (target - self.gr_db) * c;
            let gain = 10f32.powf(self.gr_db / 20.0) * self.makeup;
            let mut l = left[i] * gain;
            let mut r = right[i] * gain;
            if self.drive > 0.0 {
                l = l * (1.0 - self.drive) + (l * k).tanh() / kt * self.drive;
                r = r * (1.0 - self.drive) + (r * k).tanh() / kt * self.drive;
            }
            // Brickwall safety against makeup overshoot.
            left[i] = l.clamp(-1.0, 1.0);
            right[i] = r.clamp(-1.0, 1.0);
        }
    }
}
