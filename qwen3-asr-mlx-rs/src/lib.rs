pub mod config;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod mel;
pub mod mrope;
pub mod weights;

pub type Result<T> = std::result::Result<T, error::AsrError>;
