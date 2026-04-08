//! bee-ffi — vox-ffi service exposing the Bee engine to Swift.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use bee_rpc::{
    BeeDispatcher, BeeError, EditResolution, EngineStats,
    FeedResult, RepoDownload,
};
use bee_transcribe::corrector::Corrector;
use bee_transcribe::{Language, SessionOptions};

use dashmap::DashMap;
use tracing::info;
use vox::acceptor_on;
use vox_ffi::declare_link_endpoint;

mod engine;
mod session;
mod stats;
use engine::{load_engine, AsrEngine};
use session::SessionInner;

// ── Vox-FFI endpoint ───────────────────────────────────────────────────

declare_link_endpoint!(pub mod bee_ffi_endpoint { export = bee_ffi_v1_vtable; });

#[ctor::ctor]
fn on_load() {
    struct BeeLogWriter {
        file: Option<strip_ansi_escapes::Writer<std::fs::File>>,
    }

    impl std::io::Write for BeeLogWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            use std::io::Write as _;

            let mut stderr = std::io::stderr().lock();
            stderr.write_all(buf)?;

            if let Some(file) = &mut self.file {
                file.write_all(buf)?;
            }

            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            use std::io::Write as _;

            std::io::stderr().lock().flush()?;

            if let Some(file) = &mut self.file {
                file.flush()?;
            }

            Ok(())
        }
    }

    let subscriber = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(move || {
            let file = std::env::var("BEE_FFI_LOG_PATH").ok().and_then(|path| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .ok()
                    .map(strip_ansi_escapes::Writer::new)
            });
            BeeLogWriter { file }
        })
        .with_ansi(true)
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
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
    last_corrector: tokio::sync::Mutex<Option<Corrector>>,
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
                last_corrector: tokio::sync::Mutex::new(None),
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
        let entry =
            self.inner
                .sessions
                .get(session_id)
                .ok_or_else(|| BeeError::SessionNotFound {
                    session_id: session_id.to_owned(),
                })?;
        Ok(entry.value().clone())
    }

    /// Create a new ASR session with VAD, returning the raw Session.
    fn make_session(
        &self,
        config: &bee_rpc::SessionConfig,
    ) -> Result<bee_transcribe::Session<'static>, BeeError> {
        let engine = self.engine()?;
        let defaults = SessionOptions::default();
        let lang = if config.language.is_empty() {
            Language::default()
        } else {
            Language(config.language.clone())
        };
        let opts = SessionOptions {
            language: lang,
            chunk_duration: if config.chunk_duration > 0.0 {
                config.chunk_duration
            } else {
                defaults.chunk_duration
            },
            vad_threshold: if config.vad_threshold > 0.0 {
                config.vad_threshold
            } else {
                defaults.vad_threshold
            },
            rollback_tokens: if config.rollback_tokens > 0 {
                config.rollback_tokens as usize
            } else {
                defaults.rollback_tokens
            },
            commit_token_count: if config.commit_token_count > 0 {
                config.commit_token_count as usize
            } else {
                defaults.commit_token_count
            },
            ..defaults
        };
        info!(
            "make_session: chunk={:.2}s vad_thresh={:.2} rollback={} commit={}",
            opts.chunk_duration, opts.vad_threshold, opts.rollback_tokens, opts.commit_token_count,
        );
        let session = engine.inner.session(opts).map_err(|e| BeeError::TranscriptionError {
            message: format!("{e}"),
        })?;
        Ok(session)
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
        let id_num = self.inner.next_session_id.fetch_add(1, Ordering::Relaxed);
        let id = format!("session-{id_num}");
        info!("create_session: {id} language={}", config.language);

        let session = self.make_session(&config)?;
        self.inner.sessions.insert(
            id.clone(),
            Arc::new(tokio::sync::Mutex::new(SessionInner {
                session: Some(session),
                config,
            })),
        );

        Ok(id)
    }

    async fn feed(
        &self,
        session_id: String,
        samples: Vec<f32>,
    ) -> Result<Option<FeedResult>, BeeError> {
        let session = self.get_session(&session_id)?;

        let t0 = std::time::Instant::now();
        let result = tokio::task::spawn_blocking(move || {
            let mut guard = session.blocking_lock();
            guard
                .session
                .as_mut()
                .expect("BUG: feed on finished session")
                .feed(&samples)
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
            tracing::trace!("feed: {session_id} no update ({elapsed:.1?})");
            return Ok(None);
        };

        let committed_utf16_len = update.text[..update.asr_committed_len].encode_utf16().count() as u32;
        let text_preview: String = update.text.chars().take(80).collect();
        info!("feed: {session_id} {elapsed:.1?} committed_utf16={committed_utf16_len} text={text_preview:?}");

        Ok(Some(FeedResult {
            text: update.text,
            committed_utf16_len,
            alignments: update.alignments,
            is_final: false,
            detected_language: update.detected_language,
            correction_edits: vec![],
            correction_session_id: String::new(),
        }))
    }

    async fn finish_session(&self, session_id: String) -> Result<FeedResult, BeeError> {
        info!("finish_session: {session_id}");

        let (_, session_arc) =
            self.inner
                .sessions
                .remove(&session_id)
                .ok_or_else(|| BeeError::SessionNotFound {
                    session_id: session_id.clone(),
                })?;

        let t0 = std::time::Instant::now();
        let finish_result = tokio::task::spawn_blocking(move || {
            let mut guard = session_arc.blocking_lock();
            let session = guard
                .session
                .take()
                .expect("BUG: finish on already-finished session");
            session.finish()
        })
        .await
        .expect("spawn_blocking panicked");
        let elapsed = t0.elapsed();

        let finish = finish_result.map_err(|e| {
            tracing::error!("finish_session: {session_id} error: {e} ({elapsed:.1?})");
            BeeError::TranscriptionError {
                message: e.to_string(),
            }
        })?;

        // Extract correction data before stashing
        let (correction_edits, correction_session_id) =
            if let Some(ref corrector) = finish.corrector {
                (
                    corrector.committed_edits().to_vec(),
                    corrector.session_id().to_string(),
                )
            } else {
                (vec![], String::new())
            };

        // Stash corrector for teach/save
        if let Some(corrector) = finish.corrector {
            *self.inner.last_corrector.lock().await = Some(corrector);
        }

        let update = finish.update;
        let text_preview: String = update.text.chars().take(80).collect();
        info!("finish_session: {session_id} {elapsed:.1?} → {text_preview:?}");
        let committed_utf16_len = update.text.encode_utf16().count() as u32;
        Ok(FeedResult {
            text: update.text,
            committed_utf16_len,
            alignments: update.alignments,
            is_final: true,
            detected_language: update.detected_language,
            correction_edits,
            correction_session_id,
        })
    }

    async fn set_language(&self, session_id: String, language: String) -> Result<bool, BeeError> {
        if language.is_empty() {
            return Err(BeeError::TranscriptionError {
                message: "empty language".into(),
            });
        }

        let session = self.get_session(&session_id)?;
        let mut guard = session.lock().await;

        info!("set_language: {session_id} → {language}");
        guard.config.language = language;
        guard.session = Some(self.make_session(&guard.config)?);
        Ok(true)
    }

    async fn transcribe_samples(&self, samples: Vec<f32>) -> Result<String, BeeError> {
        let default_config = bee_rpc::SessionConfig {
            language: String::new(),
            chunk_duration: 0.0,
            vad_threshold: 0.0,
            rollback_tokens: 0,
            commit_token_count: 0,
        };
        let mut session = self.make_session(&default_config)?;

        // feed + finish are CPU/GPU intensive
        let result = tokio::task::spawn_blocking(move || {
            session
                .feed(&samples)
                .map_err(|e| BeeError::TranscriptionError {
                    message: format!("feed: {e}"),
                })?;
            session.finish().map_err(|e| BeeError::TranscriptionError {
                message: format!("finish: {e}"),
            })
        })
        .await
        .expect("spawn_blocking panicked")?;

        Ok(result.update.text)
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
        _dataset_dir: String,
        _events_path: String,
        _gate_threshold: f32,
        _ranker_threshold: f32,
    ) -> Result<bool, BeeError> {
        // Correction engine is now loaded by Engine at load_engine time.
        // This RPC exists for backwards compat — just report whether it's available.
        let engine = self.engine()?;
        let loaded = engine.inner.correction().is_some();
        if loaded {
            info!("correct_load: correction engine available (loaded with engine)");
        } else {
            info!("correct_load: no correction engine (dataset not found at load time)");
        }
        Ok(loaded)
    }

    async fn correct_teach(
        &self,
        _session_id: String,
        resolutions: Vec<EditResolution>,
    ) -> Result<bool, BeeError> {
        let engine_asr = self.engine()?;
        let correction_arc = engine_asr.inner.correction().cloned().ok_or_else(|| {
            BeeError::CorrectionError {
                message: "correction engine not loaded".into(),
            }
        })?;

        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || {
            let mut corrector_guard = inner.last_corrector.blocking_lock();
            let Some(corrector) = corrector_guard.as_mut() else {
                return Err(BeeError::CorrectionError {
                    message: "no corrector from last session".into(),
                });
            };

            let mut engine = correction_arc.lock().expect("correction engine lock poisoned");
            let pending_edits = corrector.take_pending();

            for res in &resolutions {
                let Some(pending) = pending_edits.get(&res.edit_id) else {
                    tracing::warn!(
                        "correct_teach: unknown edit_id {}",
                        res.edit_id,
                    );
                    continue;
                };

                let chosen = if res.accepted {
                    pending.chosen_alias_id
                } else {
                    None
                };

                let event = engine.judge.teach_span_event(
                    &pending.span,
                    &pending.candidates,
                    chosen,
                    &pending.ctx,
                );
                engine.event_log.push(event);
            }

            tracing::info!(
                resolutions = resolutions.len(),
                "correct_teach: applied"
            );
            Ok(true)
        })
        .await
        .expect("spawn_blocking panicked")
    }

    async fn correct_save(&self) -> Result<bool, BeeError> {
        let engine_asr = self.engine()?;
        let correction_arc = engine_asr.inner.correction().cloned().ok_or_else(|| {
            BeeError::CorrectionError {
                message: "correction engine not loaded".into(),
            }
        })?;

        tokio::task::spawn_blocking(move || {
            let mut engine = correction_arc.lock().expect("correction engine lock poisoned");

            let Some(ref path) = engine.events_path else {
                tracing::warn!("correct_save: no events_path configured, skipping");
                return Ok(true);
            };
            let path = path.clone();

            use std::io::Write;
            let file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| BeeError::CorrectionError {
                    message: format!("open events file: {e}"),
                })?;
            let mut writer = std::io::BufWriter::new(file);
            for event in engine.event_log.drain(..) {
                let json =
                    facet_json::to_string(&event).map_err(|e| BeeError::CorrectionError {
                        message: format!("serialize event: {e}"),
                    })?;
                writeln!(writer, "{json}").map_err(|e| BeeError::CorrectionError {
                    message: format!("write event: {e}"),
                })?;
            }
            writer.flush().map_err(|e| BeeError::CorrectionError {
                message: format!("flush events: {e}"),
            })?;

            tracing::info!("correct_save: events flushed to {}", path.display());
            Ok(true)
        })
        .await
        .expect("spawn_blocking panicked")
    }
}
