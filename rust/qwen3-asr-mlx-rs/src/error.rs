use thiserror::Error;

#[derive(Error, Debug)]
pub enum AsrError {
    #[error("Audio decode error: {0}")]
    AudioDecode(#[from] anyhow::Error),

    #[error("Model load error: {0}")]
    ModelLoad(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("Config error: {0}")]
    Config(String),
}
