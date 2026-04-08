use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use facet::Facet;
use serde_json::Value;

use crate::phonetic_index::{PhoneticIndex, build_index};
use crate::phonetic_lexicon::{LexiconAlias, build_phonetic_lexicon};
use crate::types::{ReviewedConfusionSurfaceRow, VocabRow};

#[derive(Debug, Clone, PartialEq)]
pub struct SeedDataset {
    pub root: PathBuf,
    pub terms: Vec<SeedTermRow>,
    pub sentence_examples: Vec<SentenceExampleRow>,
    pub recording_examples: Vec<RecordingExampleRow>,
    pub confusion_forms: Vec<ReviewedConfusionSurfaceRow>,
}

impl SeedDataset {
    pub fn canonical_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/phonetic-seed")
            .clean()
    }

    pub fn load_canonical() -> Result<Self, SeedDatasetError> {
        Self::load(Self::canonical_root())
    }

    pub fn load(root: impl AsRef<Path>) -> Result<Self, SeedDatasetError> {
        let root = root.as_ref().to_path_buf();
        Ok(Self {
            terms: load_jsonl(root.join("vocab.jsonl"))?,
            sentence_examples: load_jsonl(root.join("sentence_examples.jsonl"))?,
            recording_examples: load_recording_examples_jsonl(
                root.join("recording_examples.jsonl"),
            )?,
            confusion_forms: load_jsonl_optional(root.join("confusion_forms.jsonl")),
            root,
        })
    }

    pub fn vocab_rows(&self) -> Vec<VocabRow> {
        self.terms
            .iter()
            .enumerate()
            .map(|(idx, term)| VocabRow {
                id: idx as i64 + 1,
                term: term.term.clone(),
                spoken_auto: term.spoken.clone(),
                spoken_override: Some(term.spoken.clone()),
                reviewed_ipa: Some(term.ipa.clone()),
                reviewed: true,
                description: term.description.clone(),
            })
            .collect()
    }

    pub fn recording_audio_path(&self, row: &RecordingExampleRow) -> PathBuf {
        self.root.join(&row.audio_path)
    }

    pub fn lexicon_aliases(&self) -> Vec<LexiconAlias> {
        let mut confusion_map: std::collections::HashMap<String, Vec<ReviewedConfusionSurfaceRow>> =
            Default::default();
        for form in &self.confusion_forms {
            confusion_map
                .entry(form.term.clone())
                .or_default()
                .push(form.clone());
        }
        build_phonetic_lexicon(&self.vocab_rows(), &confusion_map)
    }

    pub fn phonetic_index(&self) -> PhoneticIndex {
        build_index(self.lexicon_aliases())
    }

    pub fn validate(&self) -> Result<(), SeedDatasetValidationError> {
        for term in &self.terms {
            if term.term.trim().is_empty() {
                return Err(SeedDatasetValidationError::EmptyTerm);
            }
            if term.spoken.trim().is_empty() {
                return Err(SeedDatasetValidationError::MissingSpoken {
                    term: term.term.clone(),
                });
            }
            if term.ipa.trim().is_empty() {
                return Err(SeedDatasetValidationError::MissingIpa {
                    term: term.term.clone(),
                });
            }
        }

        for row in &self.recording_examples {
            if row.text.trim().is_empty() {
                return Err(SeedDatasetValidationError::EmptyRecordingText {
                    term: row.term.clone(),
                    take: row.take,
                });
            }
            if row.transcript.trim().is_empty() {
                return Err(SeedDatasetValidationError::EmptyRecordingTranscript {
                    term: row.term.clone(),
                    take: row.take,
                });
            }
            let audio_path = self.recording_audio_path(row);
            if !audio_path.exists() {
                return Err(SeedDatasetValidationError::MissingAudio {
                    term: row.term.clone(),
                    take: row.take,
                    path: audio_path,
                });
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct SeedTermRow {
    pub term: String,
    pub spoken: String,
    pub ipa: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Facet, PartialEq, Eq)]
pub struct SentenceExampleRow {
    pub term: String,
    pub text: String,
    pub kind: String,
    pub surface_form: Option<String>,
}

#[derive(Debug, Clone, Facet, PartialEq)]
pub struct RecordingExampleRow {
    pub term: String,
    pub text: String,
    pub take: i64,
    pub audio_path: String,
    pub transcript: String,
    /// Per-word alignment with ASR confidence data.
    /// Populated by regen-corpus from audio files.
    pub words: Vec<RecordingWordAlignment>,
}

#[derive(Debug, Clone, Facet, PartialEq)]
pub struct CounterexampleRecordingRow {
    pub term: String,
    pub text: String,
    pub take: i64,
    pub audio_path: String,
    pub transcript: String,
    pub surface_form: String,
    #[facet(default)]
    pub words: Vec<RecordingWordAlignment>,
}

/// Per-word alignment data from ASR, stored alongside recording examples.
#[derive(Debug, Clone, PartialEq, Facet)]
pub struct RecordingWordAlignment {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: bee_types::Confidence,
}

#[derive(Debug)]
pub enum SeedDatasetError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Json {
        path: PathBuf,
        line: usize,
        message: String,
    },
}

impl fmt::Display for SeedDatasetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeedDatasetError::Io { path, source } => {
                write!(f, "read {}: {}", path.display(), source)
            }
            SeedDatasetError::Json {
                path,
                line,
                message,
            } => {
                write!(f, "parse {} line {}: {}", path.display(), line, message)
            }
        }
    }
}

impl std::error::Error for SeedDatasetError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeedDatasetValidationError {
    EmptyTerm,
    MissingSpoken {
        term: String,
    },
    MissingIpa {
        term: String,
    },
    EmptyRecordingText {
        term: String,
        take: i64,
    },
    EmptyRecordingTranscript {
        term: String,
        take: i64,
    },
    MissingAudio {
        term: String,
        take: i64,
        path: PathBuf,
    },
}

impl fmt::Display for SeedDatasetValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SeedDatasetValidationError::EmptyTerm => write!(f, "dataset contains an empty term"),
            SeedDatasetValidationError::MissingSpoken { term } => {
                write!(f, "term '{}' is missing spoken text", term)
            }
            SeedDatasetValidationError::MissingIpa { term } => {
                write!(f, "term '{}' is missing IPA", term)
            }
            SeedDatasetValidationError::EmptyRecordingText { term, take } => {
                write!(
                    f,
                    "recording '{}'/take {} is missing sentence text",
                    term, take
                )
            }
            SeedDatasetValidationError::EmptyRecordingTranscript { term, take } => write!(
                f,
                "recording '{}'/take {} is missing transcript text",
                term, take
            ),
            SeedDatasetValidationError::MissingAudio { term, take, path } => write!(
                f,
                "recording '{}'/take {} is missing audio file {}",
                term,
                take,
                path.display()
            ),
        }
    }
}

impl std::error::Error for SeedDatasetValidationError {}

trait CleanPath {
    fn clean(self) -> PathBuf;
}

impl CleanPath for PathBuf {
    fn clean(self) -> PathBuf {
        let mut out = PathBuf::new();
        for component in self.components() {
            match component {
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                    out.pop();
                }
                _ => out.push(component.as_os_str()),
            }
        }
        out
    }
}

fn load_jsonl_optional<T>(path: PathBuf) -> Vec<T>
where
    T: Facet<'static>,
{
    load_jsonl(path).unwrap_or_default()
}

fn load_jsonl<T>(path: PathBuf) -> Result<Vec<T>, SeedDatasetError>
where
    T: Facet<'static>,
{
    let file = File::open(&path).map_err(|source| SeedDatasetError::Io {
        path: path.clone(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| SeedDatasetError::Io {
            path: path.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let row = facet_json::from_str(&line).map_err(|source| SeedDatasetError::Json {
            path: path.clone(),
            line: idx + 1,
            message: source.to_string(),
        })?;
        rows.push(row);
    }

    Ok(rows)
}

fn load_recording_examples_jsonl(
    path: PathBuf,
) -> Result<Vec<RecordingExampleRow>, SeedDatasetError> {
    let file = File::open(&path).map_err(|source| SeedDatasetError::Io {
        path: path.clone(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| SeedDatasetError::Io {
            path: path.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        rows.push(parse_recording_example_row(&path, idx + 1, &line)?);
    }

    Ok(rows)
}

fn parse_recording_example_row(
    path: &Path,
    line_no: usize,
    line: &str,
) -> Result<RecordingExampleRow, SeedDatasetError> {
    let value: Value = serde_json::from_str(line).map_err(|source| SeedDatasetError::Json {
        path: path.to_path_buf(),
        line: line_no,
        message: source.to_string(),
    })?;

    let words = value
        .get("words")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .map(parse_recording_word_alignment)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
        .map_err(|message| SeedDatasetError::Json {
            path: path.to_path_buf(),
            line: line_no,
            message,
        })?
        .unwrap_or_default();

    Ok(RecordingExampleRow {
        term: required_string(&value, "term")
            .map_err(|message| json_error(path, line_no, message))?,
        text: required_string(&value, "text")
            .map_err(|message| json_error(path, line_no, message))?,
        take: required_i64(&value, "take").map_err(|message| json_error(path, line_no, message))?,
        audio_path: required_string(&value, "audio_path")
            .map_err(|message| json_error(path, line_no, message))?,
        transcript: required_string(&value, "transcript")
            .map_err(|message| json_error(path, line_no, message))?,
        words,
    })
}

fn parse_recording_word_alignment(value: &Value) -> Result<RecordingWordAlignment, String> {
    let confidence_value = if let Some(confidence) = value.get("confidence") {
        confidence.clone()
    } else {
        serde_json::json!({
            "mean_lp": value.get("mean_logprob").and_then(Value::as_f64),
            "min_lp": value.get("min_logprob").and_then(Value::as_f64),
            "mean_m": value.get("mean_margin").and_then(Value::as_f64),
            "min_m": value.get("min_margin").and_then(Value::as_f64),
        })
    };

    Ok(RecordingWordAlignment {
        word: required_string(value, "word")?,
        start: required_f64(value, "start")?,
        end: required_f64(value, "end")?,
        confidence: bee_types::Confidence {
            mean_lp: optional_f32(&confidence_value, "mean_lp").unwrap_or(0.0),
            min_lp: optional_f32(&confidence_value, "min_lp").unwrap_or(0.0),
            mean_m: optional_f32(&confidence_value, "mean_m").unwrap_or(0.0),
            min_m: optional_f32(&confidence_value, "min_m").unwrap_or(0.0),
        },
    })
}

fn required_string(value: &Value, key: &str) -> Result<String, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("missing or invalid string field `{key}`"))
}

fn required_i64(value: &Value, key: &str) -> Result<i64, String> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("missing or invalid integer field `{key}`"))
}

fn required_f64(value: &Value, key: &str) -> Result<f64, String> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| format!("missing or invalid float field `{key}`"))
}

fn optional_f32(value: &Value, key: &str) -> Option<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn json_error(path: &Path, line_no: usize, message: String) -> SeedDatasetError {
    SeedDatasetError::Json {
        path: path.to_path_buf(),
        line: line_no,
        message,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;

    fn temp_seed_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "bee-phonetic-{name}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn loads_seed_dataset_from_jsonl_files() {
        let dir = temp_seed_dir("load-seed-dataset");
        fs::write(
            dir.join("vocab.jsonl"),
            "{\"term\":\"serde\",\"spoken\":\"sirday\",\"ipa\":\"sˈɜːdeɪ\",\"description\":\"serde\"}\n",
        )
        .expect("write vocab");
        fs::write(
            dir.join("sentence_examples.jsonl"),
            "{\"term\":\"serde\",\"text\":\"serde handles config\",\"kind\":\"term\",\"surface_form\":null}\n",
        )
        .expect("write sentence_examples");
        fs::write(
            dir.join("recording_examples.jsonl"),
            "{\"term\":\"serde\",\"text\":\"serde handles config\",\"take\":1,\"audio_path\":\"audio/authored_1_take_1.ogg\",\"transcript\":\"sir day handles config\",\"words\":[]}\n",
        )
        .expect("write recording_examples");
        fs::create_dir_all(dir.join("audio")).expect("create audio dir");
        fs::write(dir.join("audio/authored_1_take_1.ogg"), []).expect("write audio");

        let dataset = SeedDataset::load(&dir).expect("load dataset");
        assert_eq!(dataset.terms.len(), 1);
        assert_eq!(dataset.sentence_examples.len(), 1);
        assert_eq!(dataset.recording_examples.len(), 1);
        assert_eq!(dataset.terms[0].term, "serde");
        assert_eq!(
            dataset.recording_examples[0].audio_path,
            "audio/authored_1_take_1.ogg"
        );
        let vocab_rows = dataset.vocab_rows();
        assert_eq!(vocab_rows[0].spoken(), "sirday");
        let aliases = dataset.lexicon_aliases();
        assert_eq!(aliases.len(), 2);
        dataset.validate().expect("valid dataset");
        assert_eq!(
            dataset.recording_audio_path(&dataset.recording_examples[0]),
            dir.join("audio/authored_1_take_1.ogg")
        );

        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn validation_rejects_missing_audio() {
        let dir = temp_seed_dir("validate-seed-dataset");
        fs::write(
            dir.join("vocab.jsonl"),
            "{\"term\":\"serde\",\"spoken\":\"sirday\",\"ipa\":\"sˈɜːdeɪ\",\"description\":\"serde\"}\n",
        )
        .expect("write vocab");
        fs::write(
            dir.join("sentence_examples.jsonl"),
            "{\"term\":\"serde\",\"text\":\"serde handles config\",\"kind\":\"term\",\"surface_form\":null}\n",
        )
        .expect("write sentence_examples");
        fs::write(
            dir.join("recording_examples.jsonl"),
            "{\"term\":\"serde\",\"text\":\"serde handles config\",\"take\":1,\"audio_path\":\"audio/missing.ogg\",\"transcript\":\"sir day handles config\",\"words\":[]}\n",
        )
        .expect("write recording_examples");

        let dataset = SeedDataset::load(&dir).expect("load dataset");
        let err = dataset.validate().expect_err("missing audio should fail");
        assert!(matches!(
            err,
            SeedDatasetValidationError::MissingAudio { .. }
        ));

        fs::remove_dir_all(dir).expect("cleanup");
    }

    #[test]
    fn canonical_seed_dataset_builds_real_index() {
        let dataset = SeedDataset::load_canonical().expect("load canonical dataset");
        dataset.validate().expect("validate canonical dataset");

        let aliases = dataset.lexicon_aliases();
        let index = dataset.phonetic_index();

        assert_eq!(dataset.terms.len(), 26);
        assert_eq!(dataset.sentence_examples.len(), 140);
        assert_eq!(dataset.recording_examples.len(), 106);
        assert!(
            !dataset
                .sentence_examples
                .iter()
                .any(|row| row.term.eq_ignore_ascii_case("clap"))
        );
        assert!(
            !dataset
                .recording_examples
                .iter()
                .any(|row| row.term.eq_ignore_ascii_case("clap"))
        );
        assert!(aliases.len() >= dataset.terms.len() * 2);
        assert_eq!(index.aliases.len(), aliases.len());
        assert!(!index.postings.is_empty());
        assert!(!index.by_phone_len.is_empty());
        assert!(!index.by_token_count.is_empty());
    }
}
