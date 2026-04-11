use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

use facet::Facet;
use regex::Regex;

#[derive(Debug, Clone, Facet)]
pub struct PhonemizeWordsRequest {
    pub words: Vec<String>,
    pub lang_code: String,
}

#[derive(Debug, Clone, Facet)]
pub struct WordIpa {
    pub word: String,
    pub ipa: String,
}

#[derive(Debug, Clone, Facet)]
pub struct TranscriptWord {
    pub word: String,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct TextWordIpa {
    pub word: String,
    pub ipa: String,
    pub char_start: usize,
    pub char_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct PhonemizeTextRequest {
    pub text: String,
    pub lang_code: String,
}

#[derive(Debug, Clone, Facet)]
pub struct PhonemizeTextResult {
    pub text: String,
    pub word_ipas: Vec<TextWordIpa>,
}

#[derive(Debug, Clone, Facet)]
pub struct PhonemizeWordsResult {
    pub word_ipas: Vec<WordIpa>,
}

#[derive(Debug, Clone, Facet)]
pub struct SidecarReady {
    pub ready: bool,
    pub model: String,
    pub device: String,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeRequest {
    pub text: String,
    pub lang_code: String,
    pub top_k: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeWordSpan {
    pub index: usize,
    pub text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeQwenTokenPiece {
    pub index: usize,
    pub token: String,
    pub char_start: usize,
    pub char_end: usize,
    pub surface: String,
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeWordScore {
    pub word_index: usize,
    pub word_text: String,
    pub char_start: usize,
    pub char_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeQwenPieceScore {
    pub piece_index: usize,
    pub piece_token: String,
    pub piece_surface: String,
    pub char_start: usize,
    pub char_end: usize,
    pub byte_start: usize,
    pub byte_end: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeRankedInput {
    pub input_index: usize,
    pub input_piece: String,
    pub score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeAttentionRow {
    pub output_index: usize,
    pub output_piece: String,
    pub top_input_index: usize,
    pub top_input_piece: String,
    pub top_score: f32,
    pub top_word_index: Option<usize>,
    pub top_word_surface: Option<String>,
    pub top_word_score: Option<f32>,
    pub word_scores: Vec<ProbeWordScore>,
    pub top_qwen_piece_index: Option<usize>,
    pub top_qwen_piece_token: Option<String>,
    pub top_qwen_piece_score: Option<f32>,
    pub qwen_piece_scores: Vec<ProbeQwenPieceScore>,
    pub ranked_inputs: Vec<ProbeRankedInput>,
}

#[derive(Debug, Clone, Facet)]
pub struct ProbeResult {
    pub text: String,
    pub lang_code: String,
    pub prompt: String,
    pub device: String,
    pub text_bytes: Vec<u8>,
    pub word_spans: Vec<ProbeWordSpan>,
    pub qwen_token_pieces: Vec<ProbeQwenTokenPiece>,
    pub input_ids: Vec<i64>,
    pub input_pieces: Vec<String>,
    pub output_ids: Vec<i64>,
    pub output_pieces: Vec<String>,
    pub decoded_output: String,
    pub cross_attention: Vec<ProbeAttentionRow>,
}

#[derive(Debug)]
pub enum SidecarClientError {
    Io(std::io::Error),
    Json(String),
    Spawn(String),
    Protocol(String),
    Sidecar(String),
}

impl fmt::Display for SidecarClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(source) => write!(f, "{source}"),
            Self::Json(message) => write!(f, "{message}"),
            Self::Spawn(message) => write!(f, "{message}"),
            Self::Protocol(message) => write!(f, "{message}"),
            Self::Sidecar(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for SidecarClientError {}

impl From<std::io::Error> for SidecarClientError {
    fn from(source: std::io::Error) -> Self {
        Self::Io(source)
    }
}

#[derive(Debug, Clone, Facet)]
struct SidecarRequest {
    words: Vec<String>,
    lang_code: String,
}

#[derive(Debug, Clone, Facet)]
struct SidecarResponse {
    word_ipas: Option<Vec<String>>,
    error: Option<String>,
}

pub struct CharsiuSidecarClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr: BufReader<ChildStderr>,
    ready: SidecarReady,
}

impl CharsiuSidecarClient {
    pub fn spawn_default() -> Result<Self, SidecarClientError> {
        Self::spawn(script_path())
    }

    pub fn spawn(script_path: impl AsRef<Path>) -> Result<Self, SidecarClientError> {
        let script_path = script_path.as_ref();
        let mut child = Command::new("uv")
            .arg("run")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| {
                SidecarClientError::Spawn(format!(
                    "spawning charsiu sidecar {}: {source}",
                    script_path.display()
                ))
            })?;

        let stdin = child.stdin.take().ok_or_else(|| {
            SidecarClientError::Spawn("charsiu sidecar stdin was not piped".to_string())
        })?;
        let stdout = child.stdout.take().ok_or_else(|| {
            SidecarClientError::Spawn("charsiu sidecar stdout was not piped".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            SidecarClientError::Spawn("charsiu sidecar stderr was not piped".to_string())
        })?;

        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr: BufReader::new(stderr),
            ready: SidecarReady {
                ready: false,
                model: String::new(),
                device: String::new(),
            },
        };
        client.ready = client.read_ready()?;
        if !client.ready.ready {
            return Err(SidecarClientError::Protocol(
                "charsiu sidecar did not report ready=true".to_string(),
            ));
        }
        Ok(client)
    }

    pub fn ready(&self) -> &SidecarReady {
        &self.ready
    }

    pub fn phonemize_words(
        &mut self,
        request: PhonemizeWordsRequest,
    ) -> Result<PhonemizeWordsResult, SidecarClientError> {
        let line = facet_json::to_string(&SidecarRequest {
            words: request.words.clone(),
            lang_code: request.lang_code,
        })
        .map_err(|err| SidecarClientError::Json(format!("encoding request: {err}")))?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;

        let response_line = self.read_response_line("charsiu response")?;
        let response: SidecarResponse = facet_json::from_str(response_line.trim())
            .map_err(|err| SidecarClientError::Json(format!("parsing response: {err}")))?;

        if let Some(error) = response.error {
            return Err(SidecarClientError::Sidecar(error));
        }

        let ipas = response.word_ipas.ok_or_else(|| {
            SidecarClientError::Protocol("charsiu response missing word_ipas".to_string())
        })?;

        if ipas.len() != request.words.len() {
            return Err(SidecarClientError::Protocol(format!(
                "charsiu response length mismatch: sent {} words, got {} ipas",
                request.words.len(),
                ipas.len()
            )));
        }

        let word_ipas = request
            .words
            .into_iter()
            .zip(ipas)
            .map(|(word, ipa)| WordIpa { word, ipa })
            .collect();

        Ok(PhonemizeWordsResult { word_ipas })
    }

    pub fn phonemize_text(
        &mut self,
        request: PhonemizeTextRequest,
    ) -> Result<PhonemizeTextResult, SidecarClientError> {
        let words = transcript_words(&request.text);
        let result = self.phonemize_words(PhonemizeWordsRequest {
            words: words.iter().map(|word| word.word.clone()).collect(),
            lang_code: request.lang_code,
        })?;
        let word_ipas = words
            .into_iter()
            .zip(result.word_ipas)
            .map(|(word, row)| TextWordIpa {
                word: row.word,
                ipa: row.ipa,
                char_start: word.char_start,
                char_end: word.char_end,
            })
            .collect();
        Ok(PhonemizeTextResult {
            text: request.text,
            word_ipas,
        })
    }

    fn read_ready(&mut self) -> Result<SidecarReady, SidecarClientError> {
        let line = self.read_response_line("charsiu ready line")?;
        facet_json::from_str(line.trim())
            .map_err(|err| SidecarClientError::Json(format!("parsing ready line: {err}")))
    }

    fn read_response_line(&mut self, label: &str) -> Result<String, SidecarClientError> {
        let mut line = String::new();
        let read = self.stdout.read_line(&mut line)?;
        if read != 0 {
            return Ok(line);
        }

        let status = self.child.try_wait().map_err(|err| {
            SidecarClientError::Protocol(format!("checking sidecar status: {err}"))
        })?;
        let stderr = read_stderr(&mut self.stderr);
        Err(SidecarClientError::Protocol(match status {
            Some(status) => format!(
                "{label}: sidecar exited with status {status}{}",
                stderr_suffix(&stderr)
            ),
            None => format!(
                "{label}: unexpected EOF from sidecar{}",
                stderr_suffix(&stderr)
            ),
        }))
    }
}

pub fn probe_text_default(request: ProbeRequest) -> Result<ProbeResult, SidecarClientError> {
    probe_text_with_script(probe_script_path(), request)
}

pub fn probe_text_with_script(
    script_path: impl AsRef<Path>,
    request: ProbeRequest,
) -> Result<ProbeResult, SidecarClientError> {
    let output = Command::new("uv")
        .arg("run")
        .arg(script_path.as_ref())
        .arg("--text")
        .arg(&request.text)
        .arg("--lang-code")
        .arg(&request.lang_code)
        .arg("--top-k")
        .arg(request.top_k.to_string())
        .arg("--json")
        .output()
        .map_err(|source| {
            SidecarClientError::Spawn(format!(
                "running charsiu probe {}: {source}",
                script_path.as_ref().display()
            ))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(SidecarClientError::Protocol(format!(
            "charsiu probe exited with status {}{}",
            output.status,
            stderr_suffix(&stderr)
        )));
    }

    let stdout = String::from_utf8(output.stdout).map_err(|source| {
        SidecarClientError::Protocol(format!("probe stdout was not utf-8: {source}"))
    })?;
    facet_json::from_str::<ProbeResult>(stdout.trim())
        .map_err(|err| SidecarClientError::Json(format!("parsing probe json: {err}")))
}

impl Drop for CharsiuSidecarClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("charsiu_g2p_sidecar.py")
}

fn probe_script_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join("charsiu_cross_attention_probe.py")
}

fn read_stderr(stderr: &mut BufReader<ChildStderr>) -> String {
    let mut buf = String::new();
    let _ = stderr.read_to_string(&mut buf);
    buf
}

fn stderr_suffix(stderr: &str) -> String {
    let trimmed = stderr.trim();
    if trimmed.is_empty() {
        String::new()
    } else {
        format!(", stderr:\n{trimmed}")
    }
}

pub fn transcript_words(text: &str) -> Vec<TranscriptWord> {
    word_re()
        .find_iter(text)
        .map(|m| TranscriptWord {
            word: m.as_str().to_string(),
            char_start: m.start(),
            char_end: m.end(),
        })
        .collect()
}

fn word_re() -> &'static Regex {
    static WORD_RE: OnceLock<Regex> = OnceLock::new();
    WORD_RE.get_or_init(|| Regex::new(r"[^\W_]+(?:['’-][^\W_]+)*").expect("valid word regex"))
}

#[cfg(test)]
mod tests {
    use super::transcript_words;

    #[test]
    fn transcript_words_keeps_char_spans() {
        let words = transcript_words("For Jason, this Thursday, use Facet.");
        let rendered: Vec<_> = words
            .into_iter()
            .map(|word| (word.word, word.char_start, word.char_end))
            .collect();
        assert_eq!(
            rendered,
            vec![
                ("For".to_string(), 0, 3),
                ("Jason".to_string(), 4, 9),
                ("this".to_string(), 11, 15),
                ("Thursday".to_string(), 16, 24),
                ("use".to_string(), 26, 29),
                ("Facet".to_string(), 30, 35),
            ]
        );
    }

    #[test]
    fn transcript_words_keeps_internal_apostrophes() {
        let words = transcript_words("we're rock'n'roll but not snake_case");
        let rendered: Vec<_> = words.into_iter().map(|word| word.word).collect();
        assert_eq!(
            rendered,
            vec![
                "we're".to_string(),
                "rock'n'roll".to_string(),
                "but".to_string(),
                "not".to_string(),
                "snake".to_string(),
                "case".to_string(),
            ]
        );
    }
}
