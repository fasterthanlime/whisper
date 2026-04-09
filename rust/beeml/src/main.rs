mod offline_eval;
mod rapid_fire;
mod rpc_impl;
mod service;
mod util;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use bee_phonetic::SeedDataset;
use bee_transcribe::{Engine, EngineConfig};
use bee_zipa_mlx::infer::ZipaInference;
use beeml::g2p::CachedEspeakG2p;
use beeml::judge::OnlineJudge;
use beeml::rpc::{BeeMl, OfflineJudgeEvalRequest};
use service::{BeeMlService, BeemlServiceInner};
use tokio::net::TcpListener;
use tracing::{error, info, warn};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};
use util::{load_correction_events, load_counterexample_recordings};
use vox::NoopClient;

fn init_tracing() -> Result<WorkerGuard> {
    let log_dir = env::var("BML_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../logs/beeml"));
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("creating log directory {}", log_dir.display()))?;

    let file_appender = tracing_appender::rolling::daily(&log_dir, "beeml.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    let stderr_writer = std::io::stderr.with_max_level(tracing::Level::INFO);
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,beeml=debug"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(stderr_writer)
                .with_ansi(true),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(file_writer)
                .with_ansi(false),
        )
        .try_init()
        .context("initializing tracing subscriber")?;

    info!(log_dir = %log_dir.display(), "tracing initialized");
    Ok(guard)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let _tracing_guard = init_tracing()?;
    let listen_addr = env::var("BML_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:9944".to_string());
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set")?;
    let tokenizer_dir = env::var("BEE_TOKENIZER_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir.clone());
    let aligner_dir = env::var("BEE_ALIGNER_DIR")
        .map(PathBuf::from)
        .context("BEE_ALIGNER_DIR must be set")?;
    let silero_dir = env::var("BEE_VAD_DIR")
        .map(PathBuf::from)
        .context("BEE_VAD_DIR must be set")?;
    let zipa_bundle_dir = env::var("BEE_ZIPA_BUNDLE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/zipa-mlx-hf")
        });

    info!(model_dir = %model_dir.display(), "loading ASR engine");
    let engine = Engine::load(&EngineConfig {
        model_dir: &model_dir,
        tokenizer_dir: &tokenizer_dir,
        aligner_dir: &aligner_dir,
        share_aligner_audio_tower: false,
        silero_dir: &silero_dir,
        correction_dir: None,
        correction_events_path: None,
        zipa_bundle_dir: &zipa_bundle_dir,
    })
    .context("loading engine")?;

    let dataset =
        SeedDataset::load_canonical().context("loading canonical phonetic seed dataset")?;
    dataset
        .validate()
        .context("validating canonical phonetic seed dataset")?;
    let index = dataset.phonetic_index();
    let counterexamples =
        load_counterexample_recordings().context("loading counterexample phonetic recordings")?;

    let event_log_path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".beeml")
        .join("events.jsonl");
    let zipa_wav_dir = env::var("BEE_PHONETIC_WAV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/bee/data/phonetic-seed/audio-wav")
        });
    let corpus_dir = env::var("BEE_ZIPA_CORPUS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join("bearcove/bee/data/zipa-corpus")
        });

    let mut judge = OnlineJudge::default();
    // Replay correction events from previous sessions
    if event_log_path.exists() {
        match load_correction_events(&event_log_path) {
            Ok(events) => {
                info!(path = %event_log_path.display(), count = events.len(), "loaded correction events");
                judge.replay_events(events);
            }
            Err(e) => {
                tracing::warn!(path = %event_log_path.display(), error = %e, "failed to load correction events, starting fresh");
            }
        }
    }

    info!(bundle = %zipa_bundle_dir.display(), "loading ZIPA inference");
    let zipa = ZipaInference::load_quantized_bundle_dir(&zipa_bundle_dir)
        .with_context(|| format!("loading ZIPA bundle {}", zipa_bundle_dir.display()))?;

    let handler = BeeMlService {
        inner: Arc::new(BeemlServiceInner {
            engine,
            index,
            dataset,
            counterexamples,
            g2p: Mutex::new(
                CachedEspeakG2p::english(
                    Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("../../target")
                        .as_ref(),
                )
                .context("initializing g2p engine")?,
            ),
            zipa: Mutex::new(zipa),
            zipa_wav_dir,
            corpus_dir,
            kokoro_sidecar: Mutex::new(None),
            corpus_eval_jobs: Mutex::new(std::collections::HashMap::new()),
            next_corpus_eval_job_id: std::sync::atomic::AtomicU64::new(1),
            judge: Mutex::new(judge),
            event_log_path,
        }),
    };

    match handler.warm_kokoro_sidecar() {
        Ok(()) => info!("kokoro sidecar warmed"),
        Err(error) => warn!(error = %error, "failed to warm kokoro sidecar at startup"),
    }

    // --export-weights: train on all data and export frozen weights
    if std::env::args().any(|a| a == "--export-weights") {
        let epochs = std::env::args()
            .skip_while(|a| a != "--epochs")
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(4usize);

        let cases = handler.teaching_cases(0, true);
        info!(
            cases = cases.len(),
            epochs, "probing cases for weight export"
        );

        let service = handler.clone();
        let probed_cases: Vec<offline_eval::ProbedCase> = tokio::task::block_in_place(|| {
            use rayon::prelude::*;
            cases
                .into_par_iter()
                .filter_map(|case| match service.probe_case_spans(&case, 3, 100) {
                    Ok(pc) => Some(pc),
                    Err(e) => {
                        tracing::warn!(case_id = %case.case_id, error = %e, "probe failed");
                        None
                    }
                })
                .collect()
        });

        info!(
            probed = probed_cases.len(),
            "probed all cases, training and exporting"
        );

        let output_dir = SeedDataset::canonical_root();
        offline_eval::train_and_export_weights(&probed_cases, epochs, 3, &output_dir)
            .map_err(|e| anyhow::anyhow!("export weights: {e}"))?;

        info!("weights exported to {}", output_dir.display());
        return Ok(());
    }

    // --offline-eval: run full Phase 4 eval suite and exit
    if std::env::args().any(|a| a == "--offline-eval") {
        let folds = std::env::args()
            .skip_while(|a| a != "--folds")
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(5u32);
        let epochs = std::env::args()
            .skip_while(|a| a != "--epochs")
            .nth(1)
            .and_then(|v| v.parse().ok())
            .unwrap_or(4u32);

        handler
            .run_offline_judge_eval(OfflineJudgeEvalRequest {
                folds,
                max_span_words: 3,
                shortlist_limit: 100,
                verify_limit: 20,
                train_epochs: epochs,
            })
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;

        return Ok(());
    }

    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("binding websocket listener on {listen_addr}"))?;

    info!(listen_addr, "beeml vox websocket server listening");

    loop {
        let (stream, peer_addr) = listener
            .accept()
            .await
            .context("accepting websocket socket")?;
        let handler = handler.clone();

        tokio::spawn(async move {
            let link = match vox_websocket::WsLink::server(stream).await {
                Ok(link) => link,
                Err(error) => {
                    warn!(%peer_addr, error = %error, "websocket handshake failed");
                    return;
                }
            };

            let establish = vox_core::acceptor_on(link)
                .on_connection(beeml::rpc::BeeMlDispatcher::new(handler))
                .establish::<NoopClient>()
                .await;

            match establish {
                Ok(client) => {
                    info!(%peer_addr, "client connected");
                    client.caller.closed().await;
                    info!(%peer_addr, "client disconnected");
                }
                Err(error) => {
                    error!(%peer_addr, error = %error, "vox session establish failed");
                }
            }
        });
    }
}
