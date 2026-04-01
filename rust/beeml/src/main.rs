use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use bee_qwen3_asr::AsrEngine;
use beeml::rpc::{BeeMl, TranscribeWavResult};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use vox::Caller;

#[derive(Clone)]
struct BeeMlService {
    inner: Arc<BeemlServiceInner>,
}

struct BeemlServiceInner {
    asr_engine: Mutex<AsrEngine>,
}

impl BeeMl for BeeMlService {
    async fn transcribe_wav(&self, wav_bytes: Vec<u8>) -> Result<TranscribeWavResult, String> {
        let engine = self.inner.asr_engine.lock().await;
        let (transcript, qwen_words) = engine
            .transcribe_wav_with_alignments(&wav_bytes)
            .map_err(|e| e.to_string())?;
        Ok(TranscribeWavResult {
            transcript,
            qwen_words,
        })
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let listen_addr = env::var("BML_WS_ADDR").unwrap_or_else(|_| "127.0.0.1:9944".to_string());
    let model_dir = env::var("BEE_ASR_MODEL_DIR")
        .map(PathBuf::from)
        .context("BEE_ASR_MODEL_DIR must be set to an ASR model directory")?;

    eprintln!("loading ASR engine from {}", model_dir.display());
    let engine = AsrEngine::load(&model_dir)?;
    let handler = BeeMlService {
        inner: Arc::new(BeemlServiceInner {
            asr_engine: Mutex::new(engine),
        }),
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
