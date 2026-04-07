//! FFI layer for bee — vox-ffi service + legacy C API.

use std::ffi::{c_char, c_float, c_uint, CStr, CString};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::time::{Duration, Instant};

use bee_rpc::{
    BeeDispatcher, CorrectionEdit, CorrectionOutput, EditResolution, EngineStats, FeedResult,
    RepoDownload, RepoFile,
};
use bee_transcribe::{Engine, EngineConfig, Language, Session, SessionOptions};
use tracing::info;
use vox::acceptor_on;
use vox_ffi::declare_link_endpoint;

mod correct;
mod engine;
mod session;
mod stats;

// ── Vox-FFI endpoint ───────────────────────────────────────────────────

declare_link_endpoint!(pub mod bee_ffi_endpoint { export = bee_ffi_v1_vtable; });

static VOX_BOOTSTRAPPED: AtomicBool = AtomicBool::new(false);

fn bootstrap_service_once() {
    if VOX_BOOTSTRAPPED.swap(true, Ordering::AcqRel) {
        return;
    }

    ffi_log("[bee-ffi] bootstrap_service_once: spawning runtime thread");
    std::thread::spawn(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        let Ok(runtime) = runtime else {
            ffi_log("[bee-ffi] failed to create tokio runtime");
            return;
        };

        runtime.block_on(async {
            ffi_log("[bee-ffi] runtime thread: waiting for accept");
            let Ok(link) = bee_ffi_endpoint::accept().await else {
                ffi_log("[bee-ffi] runtime thread: accept failed");
                return;
            };
            ffi_log("[bee-ffi] runtime thread: link accepted");

            let service = BeeService::new();
            let establish = acceptor_on(link)
                .on_connection(BeeDispatcher::new(service))
                .establish::<bee_rpc::BeeClient>()
                .await;

            match establish {
                Ok(client) => {
                    info!("bee-ffi: vox session established");
                    ffi_log("[bee-ffi] runtime thread: session established");
                    client.caller.closed().await;
                    ffi_log("[bee-ffi] runtime thread: caller closed");
                    if let Some(session) = client.session.as_ref() {
                        let _ = session.shutdown();
                    }
                }
                Err(error) => {
                    ffi_log(&format!(
                        "[bee-ffi] runtime thread: establish failed: {error}"
                    ));
                }
            }
        });
    });

    // Give the runtime thread a moment to start accepting
    std::thread::sleep(Duration::from_millis(10));
}

// ── BeeService impl ───────────────────────────────────────────────────

struct BeeServiceInner {
    engine: Mutex<Option<AsrEngine>>,
    sessions: Mutex<std::collections::HashMap<String, AsrSession>>,
    next_session_id: Mutex<u64>,
    correction: Mutex<Option<CorrectionEngine>>,
}

#[derive(Clone)]
struct BeeService {
    inner: Arc<BeeServiceInner>,
}

impl BeeService {
    fn new() -> Self {
        Self {
            inner: Arc::new(BeeServiceInner {
                engine: Mutex::new(None),
                sessions: Mutex::new(std::collections::HashMap::new()),
                next_session_id: Mutex::new(1),
                correction: Mutex::new(None),
            }),
        }
    }
}

const HF_BASE: &str = "https://huggingface.co";

fn hf_file_url(repo_id: &str, filename: &str) -> String {
    format!("{HF_BASE}/{repo_id}/resolve/main/{filename}")
}

impl bee_rpc::Bee for BeeService {
    async fn required_downloads(&self) -> Vec<RepoDownload> {
        engine::required_downloads()
    }

    async fn load_engine(&self, cache_dir: String) -> String {
        let cache_base = PathBuf::from(&cache_dir);
        let model_dir = cache_base.join("mlx-community--Qwen3-ASR-1.7B-4bit");

        match load_engine(&model_dir, &cache_base) {
            Ok(engine) => {
                *self.inner.engine.lock().unwrap() = Some(engine);
                String::new()
            }
            Err(e) => e,
        }
    }

    async fn create_session(&self, language: String) -> String {
        let guard = self.inner.engine.lock().unwrap();
        let Some(engine) = guard.as_ref() else {
            return String::new();
        };

        let lang = if language.is_empty() {
            Language::default()
        } else {
            Language(language)
        };

        let opts = SessionOptions {
            language: lang,
            ..SessionOptions::default()
        };

        let engine_ref: &Engine = &engine.inner;
        let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
        let mut session = engine_static.session(opts);

        if let Some(ref tensors) = engine.vad_tensors {
            match bee_vad::SileroVad::from_tensors(tensors) {
                Ok(vad) => session.set_vad(vad),
                Err(e) => ffi_log(&format!("Failed to create VAD: {e}")),
            }
        }

        let mut id_counter = self.inner.next_session_id.lock().unwrap();
        let id = format!("session-{}", *id_counter);
        *id_counter += 1;

        self.inner.sessions.lock().unwrap().insert(
            id.clone(),
            AsrSession {
                _engine: engine.inner.clone(),
                session,
                last_text: String::new(),
                finished: false,
            },
        );

        id
    }

    async fn feed(&self, session_id: String, samples: Vec<f32>) -> FeedResult {
        let mut sessions = self.inner.sessions.lock().unwrap();
        let Some(session) = sessions.get_mut(&session_id) else {
            return FeedResult {
                text: String::new(),
                committed_utf16_len: 0,
                alignments: vec![],
                is_final: false,
            };
        };

        if session.finished {
            return FeedResult {
                text: String::new(),
                committed_utf16_len: 0,
                alignments: vec![],
                is_final: false,
            };
        }

        match session.session.feed(&samples) {
            Ok(Some(update)) => {
                session.last_text = update.text.clone();
                let committed_utf16_len =
                    update.text[..update.committed_len].encode_utf16().count() as u32;
                let alignments = update.alignments.clone();
                FeedResult {
                    text: update.text,
                    committed_utf16_len,
                    alignments,
                    is_final: false,
                }
            }
            Ok(None) => FeedResult {
                text: session.last_text.clone(),
                committed_utf16_len: 0,
                alignments: vec![],
                is_final: false,
            },
            Err(e) => {
                ffi_log(&format!("FEED error: {e}"));
                FeedResult {
                    text: format!("error: {e}"),
                    committed_utf16_len: 0,
                    alignments: vec![],
                    is_final: false,
                }
            }
        }
    }

    async fn finish_session(&self, session_id: String) -> String {
        let mut sessions = self.inner.sessions.lock().unwrap();
        let Some(mut session) = sessions.remove(&session_id) else {
            return String::new();
        };

        if session.finished {
            return session.last_text;
        }
        session.finished = true;

        let arc = session._engine.clone();
        let engine_ref: &Engine = &arc;
        let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
        let real_session = std::mem::replace(
            &mut session.session,
            engine_static.session(SessionOptions::default()),
        );

        match real_session.finish() {
            Ok(update) => update.text,
            Err(e) => format!("error: {e}"),
        }
    }

    async fn set_language(&self, session_id: String, language: String) -> bool {
        // Language is set at session creation — changing mid-session isn't supported
        // by the engine, but we accept the call gracefully.
        let sessions = self.inner.sessions.lock().unwrap();
        sessions.contains_key(&session_id) && !language.is_empty()
    }

    async fn transcribe_samples(&self, samples: Vec<f32>) -> String {
        let guard = self.inner.engine.lock().unwrap();
        let Some(engine) = guard.as_ref() else {
            return "error: engine not loaded".into();
        };

        let engine_ref: &Engine = &engine.inner;
        let engine_static: &'static Engine = unsafe { std::mem::transmute(engine_ref) };
        let mut session = engine_static.session(SessionOptions::default());

        if let Some(ref tensors) = engine.vad_tensors {
            if let Ok(vad) = bee_vad::SileroVad::from_tensors(tensors) {
                session.set_vad(vad);
            }
        }

        if let Err(e) = session.feed(&samples) {
            return format!("error: {e}");
        }

        match session.finish() {
            Ok(update) => update.text,
            Err(e) => format!("error: {e}"),
        }
    }

    async fn get_stats(&self) -> EngineStats {
        let guard = self.inner.engine.lock().unwrap();
        match guard.as_ref() {
            Some(engine) => {
                let s = engine.stats.get();
                EngineStats {
                    cpu_percent: s.cpu_percent,
                    gpu_percent: s.gpu_percent,
                    vram_used_mb: s.vram_used_mb,
                    ram_used_mb: s.ram_used_mb,
                }
            }
            None => EngineStats {
                cpu_percent: 0.0,
                gpu_percent: 0.0,
                vram_used_mb: 0.0,
                ram_used_mb: 0.0,
            },
        }
    }

    async fn correct_load(
        &self,
        dataset_dir: String,
        events_path: String,
        gate_threshold: f32,
        ranker_threshold: f32,
    ) -> String {
        let events = if events_path.is_empty() {
            None
        } else {
            Some(PathBuf::from(events_path))
        };
        match load_correction_engine(
            Path::new(&dataset_dir),
            events,
            gate_threshold,
            ranker_threshold,
        ) {
            Ok(engine) => {
                *self.inner.correction.lock().unwrap() = Some(engine);
                String::new()
            }
            Err(e) => e,
        }
    }

    async fn correct_process(&self, text: String, app_id: String) -> CorrectionOutput {
        let mut guard = self.inner.correction.lock().unwrap();
        let Some(engine) = guard.as_mut() else {
            return CorrectionOutput {
                session_id: String::new(),
                best_text: text,
                edits: vec![],
            };
        };

        let app_id_opt = if app_id.is_empty() {
            None
        } else {
            Some(app_id)
        };

        let spans = enumerate_transcript_spans_with(
            &text,
            3,
            None::<&[bee_types::TranscriptAlignmentToken]>,
            |span_text| engine.g2p.ipa_tokens(span_text).ok().flatten(),
        );

        let mut edits = Vec::new();
        let session_id = format!(
            "{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        for span in &spans {
            let query = RetrievalQuery {
                text: span.text.clone(),
                ipa_tokens: span.ipa_tokens.clone(),
                reduced_ipa_tokens: span.reduced_ipa_tokens.clone(),
                feature_tokens: feature_tokens_for_ipa(&span.ipa_tokens),
                token_count: (span.token_end - span.token_start) as u8,
            };
            let shortlist = query_index(&engine.index, &query, 50);
            if shortlist.is_empty() {
                continue;
            }
            let scored = score_shortlist(span, &shortlist, &engine.index);
            if scored.is_empty() {
                continue;
            }

            let candidates_with_flags: Vec<_> = scored
                .iter()
                .map(|c| {
                    let flags = engine
                        .index
                        .aliases
                        .iter()
                        .find(|a| a.alias_id == c.alias_id)
                        .map(|a| a.identifier_flags.clone())
                        .unwrap_or_default();
                    (c.clone(), flags)
                })
                .collect();

            let ctx =
                bee_correct::judge::extract_span_context(&text, span.char_start, span.char_end);
            let ctx = SpanContext {
                app_id: app_id_opt.clone(),
                ..ctx
            };

            let decision = engine.judge.score_span(span, &candidates_with_flags, &ctx);

            if let Some(ref chosen) = decision.chosen {
                edits.push(CorrectionEdit {
                    edit_id: format!("e{}", edits.len()),
                    span_start: span.char_start as u32,
                    span_end: span.char_end as u32,
                    original: span.text.clone(),
                    replacement: chosen.replacement_text.clone(),
                    term: chosen.term.clone(),
                    alias_id: chosen.alias_id as i32,
                    ranker_prob: chosen.ranker_prob as f64,
                    gate_prob: decision.gate_prob as f64,
                });
            }
        }

        // Build best text by applying edits
        let mut best_text = text.clone();
        let mut offset: i64 = 0;
        for edit in &edits {
            let start = edit.span_start as usize;
            let end = edit.span_end as usize;
            let adj_start = (start as i64 + offset) as usize;
            let adj_end = (end as i64 + offset) as usize;
            best_text.replace_range(adj_start..adj_end, &edit.replacement);
            offset += edit.replacement.len() as i64 - (end - start) as i64;
        }

        CorrectionOutput {
            session_id,
            best_text,
            edits,
        }
    }

    async fn correct_teach(
        &self,
        _session_id: String,
        _resolutions: Vec<EditResolution>,
    ) -> String {
        // TODO: implement teaching from stable session IDs
        String::new()
    }

    async fn correct_save(&self) -> String {
        // TODO: implement persistence (weights + memory + events)
        String::new()
    }
}

// Trigger bootstrap when the vtable is first accessed
#[unsafe(no_mangle)]
pub extern "C" fn bee_ffi_bootstrap() {
    bootstrap_service_once();
}

static INIT_LOGGER: Once = Once::new();

fn init_logger() {
    INIT_LOGGER.call_once(|| {
        env_logger::init();
    });
}

static FFI_LOG_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Set the path for FFI diagnostic logs. Call this before loading any engine.
/// Pass NULL to disable file logging.
#[unsafe(no_mangle)]
pub extern "C" fn asr_set_log_path(path: *const c_char) {
    let mut guard = FFI_LOG_PATH.lock().unwrap();
    if path.is_null() {
        *guard = None;
    } else {
        let s = unsafe { CStr::from_ptr(path) };
        if let Ok(s) = s.to_str() {
            *guard = Some(PathBuf::from(s));
        }
    }
}
