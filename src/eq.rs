//! Master tone EQ: a presentation post-stage (after the room) that shapes
//! the overall spectral balance. Our partial amplitudes were calibrated
//! from the darker SplendidGrandPiano samples; against a bright, present
//! recording (Gould 1981) the synth reads dull. This dials in the
//! presence/brilliance to taste. All gains come from env vars so the FAD
//! sweep can tune them; default is the FAD-optimized curve.
//!
//! Cascade: low-shelf (bass trim) -> presence peak -> high-shelf (air).
//! RBJ biquads, transposed direct form II, stereo. Allocation-free.

struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    z: [(f32, f32); 2], // per-channel state
}

impl Biquad {
    fn new(b0: f32, b1: f32, b2: f32, a0: f32, a1: f32, a2: f32) -> Biquad {
        Biquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0,
                 z: [(0.0, 0.0); 2] }
    }

    fn low_shelf(f0: f32, gain_db: f32, sr: f32) -> Biquad {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = std::f32::consts::TAU * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / 2.0 * (2.0f32).sqrt(); // S=1 shelf slope
        let sa = 2.0 * a.sqrt() * alpha;
        Biquad::new(
            a * ((a + 1.0) - (a - 1.0) * c + sa),
            2.0 * a * ((a - 1.0) - (a + 1.0) * c),
            a * ((a + 1.0) - (a - 1.0) * c - sa),
            (a + 1.0) + (a - 1.0) * c + sa,
            -2.0 * ((a - 1.0) + (a + 1.0) * c),
            (a + 1.0) + (a - 1.0) * c - sa,
        )
    }

    fn high_shelf(f0: f32, gain_db: f32, sr: f32) -> Biquad {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = std::f32::consts::TAU * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / 2.0 * (2.0f32).sqrt();
        let sa = 2.0 * a.sqrt() * alpha;
        Biquad::new(
            a * ((a + 1.0) + (a - 1.0) * c + sa),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * c),
            a * ((a + 1.0) + (a - 1.0) * c - sa),
            (a + 1.0) - (a - 1.0) * c + sa,
            2.0 * ((a - 1.0) - (a + 1.0) * c),
            (a + 1.0) - (a - 1.0) * c - sa,
        )
    }

    fn peak(f0: f32, q: f32, gain_db: f32, sr: f32) -> Biquad {
        let a = 10f32.powf(gain_db / 40.0);
        let w0 = std::f32::consts::TAU * f0 / sr;
        let (s, c) = w0.sin_cos();
        let alpha = s / (2.0 * q);
        Biquad::new(
            1.0 + alpha * a,
            -2.0 * c,
            1.0 - alpha * a,
            1.0 + alpha / a,
            -2.0 * c,
            1.0 - alpha / a,
        )
    }

    #[inline]
    fn run(&mut self, x: f32, ch: usize) -> f32 {
        let z = &mut self.z[ch];
        let y = self.b0 * x + z.0;
        z.0 = self.b1 * x - self.a1 * y + z.1;
        z.1 = self.b2 * x - self.a2 * y;
        y
    }
}

pub struct MasterEq {
    stages: Vec<Biquad>,
    active: bool,
}

impl MasterEq {
    /// Built from env vars (PIANO_EQ_*), honoring the FAD-tuned defaults.
    pub fn from_env(sr: f32) -> MasterEq {
        let g = |name: &str, default: f32| {
            std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
        };
        // Defaults: bass trim + presence/air lift toward the measured
        // Gould LTAS (the snare-removal had darkened 2-6 kHz; this refills
        // it tonally). Tune live via PIANO_EQ_* — e.g. PIANO_EQ_HIGH_DB.
        let low_db = g("PIANO_EQ_LOW_DB", -2.0);
        let low_hz = g("PIANO_EQ_LOW_HZ", 300.0);
        let peak_db = g("PIANO_EQ_PEAK_DB", 4.0);
        let peak_hz = g("PIANO_EQ_PEAK_HZ", 5000.0);
        let peak_q = g("PIANO_EQ_PEAK_Q", 0.8);
        let high_db = g("PIANO_EQ_HIGH_DB", 6.0);
        let high_hz = g("PIANO_EQ_HIGH_HZ", 2500.0);
        MasterEq::with_params(sr, low_db, low_hz, peak_db, peak_hz, peak_q, high_db, high_hz)
    }

    /// Explicit-parameter constructor (used by the WASM engine for live
    /// tuning, where there are no env vars).
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(
        sr: f32,
        low_db: f32,
        low_hz: f32,
        peak_db: f32,
        peak_hz: f32,
        peak_q: f32,
        high_db: f32,
        high_hz: f32,
    ) -> MasterEq {
        let stages = vec![
            Biquad::low_shelf(low_hz, low_db, sr),
            Biquad::peak(peak_hz, peak_q, peak_db, sr),
            Biquad::high_shelf(high_hz, high_db, sr),
        ];
        let active = low_db != 0.0 || peak_db != 0.0 || high_db != 0.0;
        MasterEq { stages, active }
    }

    pub fn process(&mut self, left: &mut [f32], right: &mut [f32]) {
        if !self.active {
            return;
        }
        for i in 0..left.len() {
            let mut l = left[i];
            let mut r = right[i];
            for s in &mut self.stages {
                l = s.run(l, 0);
                r = s.run(r, 1);
            }
            left[i] = l;
            right[i] = r;
        }
    }
}
