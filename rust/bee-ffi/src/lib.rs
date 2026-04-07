//! bee-ffi — vox-ffi service exposing the Bee engine to Swift.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use bee_phonetic::{
    enumerate_transcript_spans_with, feature_tokens_for_ipa, query_index, score_shortlist,
    RetrievalQuery,
};
use bee_rpc::{
    BeeDispatcher, BeeError, CorrectionEdit, CorrectionOutput, EditResolution, EngineStats,
    FeedResult, RepoDownload,
};
use bee_transcribe::{Language, SessionOptions};

use bee_types::SpanContext;
use dashmap::DashMap;
use tracing::info;
use vox::acceptor_on;
use vox_ffi::declare_link_endpoint;

mod correct;
mod engine;
mod session;
mod stats;

use correct::{load_correction_engine, CorrectionEngine};
use engine::{load_engine, AsrEngine};
use session::SessionInner;

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

type SessionMap = DashMap<String, Arc<tokio::sync::Mutex<SessionInner>>>;

struct BeeServiceInner {
    engine: OnceLock<AsrEngine>,
    sessions: SessionMap,
    next_session_id: AtomicU64,
    correction: tokio::sync::Mutex<Option<CorrectionEngine>>,
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
                sessions: DashMap::new(),
                next_session_id: AtomicU64::new(1),
                correction: tokio::sync::Mutex::new(None),
            }),
        }
    }

    fn engine(&self) -> Result<&AsrEngine, BeeError> {
        self.inner.engine.get().ok_or(BeeError::EngineNotLoaded)
    }

    fn get_session(
        &self,
        session_id: &str,
    ) -> Result<Arc<tokio::sync::Mutex<SessionInner>>, BeeError> {
        let entry = self.inner.sessions.get(session_id).ok_or_else(|| {
            BeeError::SessionNotFound {
                session_id: session_id.to_owned(),
            }
        })?;
        Ok(entry.value().clone())
    }

    /// Create a new ASR session with VAD, returning the raw Session.
    fn make_session(engine: &AsrEngine, config: &bee_rpc::SessionConfig) -> bee_transcribe::Session<'static> {
        let defaults = SessionOptions::default();
        let lang = if config.language.is_empty() {
            Language::default()
        } else {
            Language(config.language.clone())
        };
        let opts = SessionOptions {
            language: lang,
            chunk_duration: if config.chunk_duration > 0.0 { config.chunk_duration } else { defaults.chunk_duration },
            vad_threshold: if config.vad_threshold > 0.0 { config.vad_threshold } else { defaults.vad_threshold },
            rollback_tokens: if config.rollback_tokens > 0 { config.rollback_tokens as usize } else { defaults.rollback_tokens },
            commit_token_count: if config.commit_token_count > 0 { config.commit_token_count as usize } else { defaults.commit_token_count },
            ..defaults
        };
        info!(
            "make_session: chunk={:.2}s vad_thresh={:.2} rollback={} commit={}",
            opts.chunk_duration, opts.vad_threshold, opts.rollback_tokens, opts.commit_token_count,
        );
        let mut session = engine.inner.session(opts);
        if let Some(ref tensors) = engine.vad_tensors {
            match bee_vad::SileroVad::from_tensors(tensors) {
                Ok(vad) => session.set_vad(vad),
                Err(e) => tracing::error!("VAD creation failed: {e}"),
            }
        }
        session
    }
}

impl bee_rpc::Bee for BeeService {
    async fn required_downloads(&self) -> Vec<RepoDownload> {
        engine::required_downloads()
    }

    async fn load_engine(&self, cache_dir: String) -> Result<bool, BeeError> {
        let cache_base = PathBuf::from(&cache_dir);
        let model_dir = cache_base.join("mlx-community--Qwen3-ASR-1.7B-4bit");

        let engine = load_engine(&model_dir, &cache_base).map_err(|e| {
            tracing::error!("load_engine: {e}");
            BeeError::LoadFailed { message: e }
        })?;
        let _ = self.inner.engine.set(engine);
        info!("load_engine: success");
        Ok(true)
    }

    async fn create_session(&self, config: bee_rpc::SessionConfig) -> Result<String, BeeError> {
        let engine = self.engine()?;

        let id_num = self.inner.next_session_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("session-{id_num}");
        info!("create_session: {id} language={}", config.language);

        let session = Self::make_session(engine, &config);
        self.inner.sessions.insert(
            id.clone(),
            Arc::new(tokio::sync::Mutex::new(SessionInner { session: Some(session), config })),
        );

        Ok(id)
    }

    async fn feed(&self, session_id: String, samples: Vec<f32>) -> Result<Option<FeedResult>, BeeError> {
        let session = self.get_session(&session_id)?;

        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = session.blocking_lock();
            guard.session.as_mut().expect("BUG: feed on finished session").feed(&samples)
        })
        .await
        .expect("spawn_blocking panicked");
        let elapsed = t0.elapsed();

        let update = result.map_err(|e| {
            tracing::error!("feed: session {session_id} error: {e}");
            BeeError::TranscriptionError {
                message: e.to_string(),
            }
        })?;

        let Some(update) = update else {
            tracing::debug!("feed: {session_id} no update ({elapsed:.1?})");
            return Ok(None);
        };

        let committed_utf16_len =
            update.text[..update.committed_len].encode_utf16().count() as u32;
        let text_preview: String = update.text.chars().take(80).collect();
        info!("feed: {session_id} {elapsed:.1?} committed_utf16={committed_utf16_len} text={text_preview:?}");

        Ok(Some(FeedResult {
            text: update.text,
            committed_utf16_len,
            alignments: update.alignments,
            is_final: false,
        }))
    }

    async fn finish_session(&self, session_id: String) -> Result<String, BeeError> {
        info!("finish_session: {session_id}");

        let (_, session_arc) =
            self.inner.sessions.remove(&session_id).ok_or_else(|| {
                BeeError::SessionNotFound {
                    session_id: session_id.clone(),
                }
            })?;

        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = session_arc.blocking_lock();
            let session = guard.session.take().expect("BUG: finish on already-finished session");
            session.finish()
        })
        .await
        .expect("spawn_blocking panicked");
        let elapsed = t0.elapsed();

        let update = result.map_err(|e| {
            tracing::error!("finish_session: {session_id} error: {e} ({elapsed:.1?})");
            BeeError::TranscriptionError {
                message: e.to_string(),
            }
        })?;

        let text_preview: String = update.text.chars().take(80).collect();
        info!("finish_session: {session_id} {elapsed:.1?} → {text_preview:?}");
        Ok(update.text)
    }

    async fn set_language(
        &self,
        session_id: String,
        language: String,
    ) -> Result<bool, BeeError> {
        if language.is_empty() {
            return Err(BeeError::TranscriptionError {
                message: "empty language".into(),
            });
        }

        let engine = self.engine()?;
        let session = self.get_session(&session_id)?;
        let mut guard = session.lock().await;

        info!("set_language: {session_id} → {language}");
        guard.config.language = language;
        guard.session = Some(Self::make_session(engine, &guard.config));
        Ok(true)
    }

    async fn transcribe_samples(&self, samples: Vec<f32>) -> Result<String, BeeError> {
        let engine = self.engine()?;
        let default_config = bee_rpc::SessionConfig {
            language: String::new(),
            chunk_duration: 0.0,
            vad_threshold: 0.0,
            rollback_tokens: 0,
            commit_token_count: 0,
        };
        let mut session = Self::make_session(engine, &default_config);

        // feed + finish are CPU/GPU intensive
        let result = tokio::task::spawn_blocking(move || {
            session.feed(&samples).map_err(|e| {
                BeeError::TranscriptionError {
                    message: format!("feed: {e}"),
                }
            })?;
            session.finish().map_err(|e| {
                BeeError::TranscriptionError {
                    message: format!("finish: {e}"),
                }
            })
        })
        .await
        .expect("spawn_blocking panicked")?;

        Ok(result.text)
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
    ) -> Result<bool, BeeError> {
        let events = if events_path.is_empty() {
            None
        } else {
            Some(PathBuf::from(events_path))
        };
        let engine =
            load_correction_engine(Path::new(&dataset_dir), events, gate_threshold, ranker_threshold)
                .map_err(|e| {
                    tracing::error!("correct_load: {e}");
                    BeeError::CorrectionError { message: e }
                })?;
        info!("correct_load: success");
        *self.inner.correction.lock().await = Some(engine);
        Ok(true)
    }

    async fn correct_process(&self, text: String, app_id: String) -> CorrectionOutput {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut guard = inner.correction.blocking_lock();
            let Some(engine) = guard.as_mut() else {
                tracing::warn!("correct_process: correction engine not loaded, passing through");
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
        })
        .await
        .expect("spawn_blocking panicked")
    }

    async fn correct_teach(
        &self,
        _session_id: String,
        _resolutions: Vec<EditResolution>,
    ) -> Result<bool, BeeError> {
        Err(BeeError::NotImplemented)
    }

    async fn correct_save(&self) -> Result<bool, BeeError> {
        Err(BeeError::NotImplemented)
    }
}
