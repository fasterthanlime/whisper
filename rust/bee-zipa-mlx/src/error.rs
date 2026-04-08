use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ZipaError {
    #[error("missing ZIPA artifact: {path}")]
    MissingArtifact { path: PathBuf },
    #[error("unsupported WAV format in {path}: {reason}")]
    UnsupportedWav { path: PathBuf, reason: String },
    #[error("invalid tokens.txt line: {0}")]
    InvalidTokenLine(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Wav(#[from] hound::Error),
}
