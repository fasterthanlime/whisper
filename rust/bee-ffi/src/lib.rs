//! bee-ffi — vox-ffi service exposing the Bee engine to Swift.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU64, Ordering};

use bee_phonetic::{
    enumerate_transcript_spans_with, feature_tokens_for_ipa, query_index, score_shortlist,
    RetrievalQuery,
};
use bee_rpc::{
    BeeDispatcher, CorrectionEdit, CorrectionOutput, EditResolution, EngineStats, FeedResult,
    RepoDownload,
};
use bee_transcribe::{Language, SessionOptions};
use bee_types::SpanContext;
use tracing::info;
use vox::acceptor_on;
use vox_ffi::declare_link_endpoint;

mod correct;
mod engine;
mod session;
mod stats;

use correct::{load_correction_engine, CorrectionEngine};
use engine::{load_engine, AsrEngine};
use session::AsrSession;

// ── Vox-FFI endpoint ───────────────────────────────────────────────────

declare_link_endpoint!(pub mod bee_ffi_endpoint { export = bee_ffi_v1_vtable; });

#[ctor::ctor]
fn on_load() {
    // Set up tracing subscriber writing to the log file specified by Swift
    if let Ok(path) = std::env::var("BEE_FFI_LOG_PATH") {
        if let Ok(file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let subscriber = tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
                )
                .with_writer(file)
                .with_ansi(false)
                .finish();
            let _ = tracing::subscriber::set_global_default(subscriber);
        }
    }
    info!("bee-ffi: dylib loaded, spawning runtime thread");
    std::thread::spawn(|| {
        info!("bee-ffi: runtime thread started");
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build();
        let Ok(runtime) = runtime else {
            tracing::error!("bee-ffi: failed to create tokio runtime");
            return;
        };
        info!("bee-ffi: tokio runtime created");

        runtime.block_on(async {
            info!("bee-ffi: waiting for accept");
            let link = bee_ffi_endpoint::accept().await;
            match &link {
                Ok(_) => info!("bee-ffi: accept succeeded, got link"),
                Err(e) => {
                    tracing::error!("bee-ffi: accept failed: {e}");
                    return;
                }
            }
            let link = link.unwrap();

            info!("bee-ffi: calling acceptor_on + establish");
            let service = BeeService::new();
            let establish = acceptor_on(link)
                .on_connection(BeeDispatcher::new(service))
                .establish::<vox::NoopClient>()
                .await;

            match establish {
                Ok(client) => {
                    info!("bee-ffi: session established, waiting for caller close");
                    client.caller.closed().await;
                    info!("bee-ffi: caller closed, shutting down");
                    if let Some(session) = client.session.as_ref() {
                        let _ = session.shutdown();
                        info!("bee-ffi: session shutdown complete");
                    }
                }
                Err(error) => {
                    tracing::error!("bee-ffi: establish failed: {error}");
                }
            }
        });
        info!("bee-ffi: runtime thread exiting");
    });
}

// ── BeeService impl ───────────────────────────────────────────────────

struct BeeServiceInner {
    engine: OnceLock<AsrEngine>,
    sessions: Mutex<std::collections::HashMap<String, AsrSession>>,
    next_session_id: AtomicU64,
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
                engine: OnceLock::new(),
                sessions: Mutex::new(std::collections::HashMap::new()),
                next_session_id: AtomicU64::new(1),
                correction: Mutex::new(None),
            }),
        }
    }
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
                let _ = self.inner.engine.set(engine);
                String::new()
            }
            Err(e) => e,
        }
    }

    async fn create_session(&self, language: String) -> String {
        let Some(engine) = self.inner.engine.get() else {
            return String::new();
        };

        let lang = if language.is_empty() {
            Language::default()
        } else {
            Language(language)
        };

        let id_num = self.inner.next_session_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("session-{id_num}");
        tracing::info!("create_session: {id} language={}", lang.0);

        let opts = SessionOptions {
            language: lang,
            ..SessionOptions::default()
        };

        let mut session = engine.inner.session(opts);

        if let Some(ref tensors) = engine.vad_tensors {
            match bee_vad::SileroVad::from_tensors(tensors) {
                Ok(vad) => session.set_vad(vad),
                Err(e) => tracing::warn!("Failed to create VAD: {e}"),
            }
        }

        self.inner.sessions.lock().unwrap().insert(
            id.clone(),
            AsrSession {
                session,
                last_text: String::new(),
                finished: false,
            },
        );

        id
    }

    async fn feed(&self, session_id: String, samples: Vec<f32>) -> FeedResult {
        let empty_result = FeedResult {
            text: String::new(),
            committed_utf16_len: 0,
            alignments: vec![],
            is_final: false,
        };

        // Take session out of map so we can drop the lock during inference
        let mut session = {
            let mut sessions = self.inner.sessions.lock().unwrap();
            match sessions.remove(&session_id) {
                Some(s) if !s.finished => s,
                Some(s) => {
                    tracing::warn!("feed: session {session_id} already finished");
                    sessions.insert(session_id, s);
                    return empty_result;
                }
                None => {
                    tracing::warn!("feed: session {session_id} not found");
                    return empty_result;
                }
            }
        };

        let result = match session.session.feed(&samples) {
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
                tracing::error!("FEED error: {e}");
                FeedResult {
                    text: format!("error: {e}"),
                    committed_utf16_len: 0,
                    alignments: vec![],
                    is_final: false,
                }
            }
        };

        // Put session back
        self.inner.sessions.lock().unwrap().insert(session_id, session);
        result
    }

    async fn finish_session(&self, session_id: String) -> String {
        tracing::info!("finish_session: {session_id}");
        // Take session out of map so we can drop the lock during inference
        let mut session = {
            let mut sessions = self.inner.sessions.lock().unwrap();
            match sessions.remove(&session_id) {
                Some(s) => s,
                None => {
                    tracing::warn!("finish_session: {session_id} not found");
                    return String::new();
                }
            }
        };

        if session.finished {
            return session.last_text;
        }
        session.finished = true;

        match session.session.finish() {
            Ok(update) => update.text,
            Err(e) => format!("error: {e}"),
        }
    }

    async fn set_language(&self, session_id: String, language: String) -> bool {
        if language.is_empty() {
            return false;
        }

        let Some(engine) = self.inner.engine.get() else {
            return false;
        };

        let mut sessions = self.inner.sessions.lock().unwrap();
        let Some(_old_session) = sessions.remove(&session_id) else {
            return false;
        };

        let opts = SessionOptions {
            language: Language(language),
            ..SessionOptions::default()
        };

        let mut new_session = engine.inner.session(opts);

        if let Some(ref tensors) = engine.vad_tensors {
            if let Ok(vad) = bee_vad::SileroVad::from_tensors(tensors) {
                new_session.set_vad(vad);
            }
        }

        sessions.insert(
            session_id,
            AsrSession {
                session: new_session,
                last_text: String::new(),
                finished: false,
            },
        );

        true
    }

    async fn transcribe_samples(&self, samples: Vec<f32>) -> String {
        let Some(engine) = self.inner.engine.get() else {
            return "error: engine not loaded".into();
        };

        let mut session = engine.inner.session(SessionOptions::default());

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
        match self.inner.engine.get() {
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

        // Build best text by applying edits in order (edits are sorted by span_start).
        // Each replacement may shift subsequent positions: if we replace "teh" (3 chars)
        // with "the" (3 chars), offset stays 0. If we replace "NY" (2) with "New York" (8),
        // offset grows by +6 and all later positions must be adjusted.
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

