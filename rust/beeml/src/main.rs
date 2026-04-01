use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use bee_transcribe::{Engine, EngineConfig, SessionOptions};
use beeml::rpc::{BeeMl, TranscribeWavResult};
use tokio::net::TcpListener;
use vox::Caller;

#[derive(Clone)]
struct BeeMlService {
    inner: Arc<BeemlServiceInner>,
}

struct BeemlServiceInner {
    engine: Engine,
}

impl BeeMl for BeeMlService {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String> {
        let samples = bee_transcribe::decode_wav(&wav_bytes)
            .map_err(|e| e.to_string())?;

        let mut session = self.inner.engine.session(SessionOptions::default());

        // Feed all audio in one shot
        session.feed(&samples).map_err(|e| e.to_string())?;
        let update = session.finish().map_err(|e| e.to_string())?;

        Ok(TranscribeWavResult {
            transcript: update.text,
            words: update.alignments,
        })
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let listen_addr = env::var("BML_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:9944".to_string());
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set")?;
    let tokenizer_path = env::var("BEE_TOKENIZER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| model_dir.join("tokenizer.json"));
    let aligner_dir = env::var("BEE_ALIGNER_DIR")
        .map(PathBuf::from)
        .context("BEE_ALIGNER_DIR must be set")?;

    eprintln!("loading ASR engine from {}", model_dir.display());
    let engine = Engine::load(&EngineConfig {
        model_dir: &model_dir,
        tokenizer_path: &tokenizer_path,
        aligner_dir: &aligner_dir,
    }).context("loading engine")?;

    let handler = BeeMlService {
        inner: Arc::new(BeemlServiceInner { engine }),
    };

    let listener = TcpListener::bind(&listen_addr)
        .await
        .with_context(|| format!("binding websocket listener on {listen_addr}"))?;

    eprintln!("beeml vox websocket server listening on ws://{listen_addr}");

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
                    eprintln!("websocket handshake failed for {peer_addr}: {error}");
                    return;
                }
            };

            let establish = vox_core::acceptor_on(link)
                .establish::<vox_core::DriverCaller>(beeml::rpc::BeeMlDispatcher::new(handler))
                .await;

            match establish {
                Ok((caller, _session_handle)) => {
                    eprintln!("client connected: {peer_addr}");
                    caller.closed().await;
                    eprintln!("client disconnected: {peer_addr}");
                }
                Err(error) => {
                    eprintln!("vox session establish failed for {peer_addr}: {error}");
                }
            }
        });
    }
}
