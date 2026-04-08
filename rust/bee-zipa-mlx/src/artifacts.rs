use std::path::{Path, PathBuf};

use crate::config::ZipaVariant;
use crate::error::ZipaError;
use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceArtifacts {
    pub root: PathBuf,
    pub onnx_model: PathBuf,
    pub tokens_txt: PathBuf,
}

impl ReferenceArtifacts {
    pub fn from_dir(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let onnx_model = root.join("model.onnx");
        let tokens_txt = root.join("tokens.txt");

        for path in [&onnx_model, &tokens_txt] {
            if !path.exists() {
                return Err(ZipaError::MissingArtifact { path: path.clone() });
            }
        }

        Ok(Self {
            root,
            onnx_model,
            tokens_txt,
        })
    }

    pub fn default_reference_dir(variant: ZipaVariant) -> PathBuf {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("~"));
        match variant {
            ZipaVariant::SmallCrCtcNsNoDiacritics700k => {
                home.join("bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::ReferenceArtifacts;

    #[test]
    fn default_dir_matches_reference_checkout_layout() {
        let path = ReferenceArtifacts::default_reference_dir(
            crate::config::ZipaVariant::SmallCrCtcNsNoDiacritics700k,
        );
        assert!(path.ends_with("checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp"));
    }

    #[test]
    fn from_dir_requires_model_and_tokens() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("bee-zipa-mlx-{unique}"));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.onnx"), b"model").unwrap();
        std::fs::write(dir.join("tokens.txt"), b"<blk> 0\n").unwrap();

        let artifacts = ReferenceArtifacts::from_dir(&dir).unwrap();
        assert_eq!(artifacts.root, dir);
    }
}
