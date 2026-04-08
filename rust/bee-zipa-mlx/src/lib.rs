pub mod artifacts;
pub mod audio;
pub mod config;
pub mod encoder;
pub mod error;
pub mod features;
pub mod load;
pub mod model;
pub mod tokenizer;

pub type Result<T> = std::result::Result<T, error::ZipaError>;
