use std::sync::OnceLock;
use std::time::Instant;

fn phase_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BEE_PHASE_TIMING")
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false)
    })
}

pub(crate) fn phase_start() -> Option<Instant> {
    phase_timing_enabled().then(Instant::now)
}

pub(crate) fn log_phase(component: &'static str, phase: &'static str, start: Option<Instant>) {
    if let Some(start) = start {
        let elapsed = start.elapsed();
        tracing::info!(
            target: "bee_phase",
            component,
            phase,
            ms = elapsed.as_secs_f64() * 1000.0,
            "phase timing"
        );
    }
}

pub(crate) fn log_phase_chunk(
    component: &'static str,
    phase: &'static str,
    chunk: usize,
    start: Option<Instant>,
) {
    if let Some(start) = start {
        let elapsed = start.elapsed();
        tracing::info!(
            target: "bee_phase",
            component,
            phase,
            chunk,
            ms = elapsed.as_secs_f64() * 1000.0,
            "phase timing"
        );
    }
}
