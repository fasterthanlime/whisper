use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::OnceLock;

use bee_phonetic::{
    normalize_ipa_for_comparison, normalize_ipa_for_comparison_with_spans, parse_reviewed_ipa,
};
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
    pub emitted_text: String,
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

#[derive(Debug, Clone, Facet)]
pub struct ProbeTokenOwnershipRun {
    pub qwen_piece_index: usize,
    pub qwen_piece_token: String,
    pub qwen_piece_surface: String,
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub output_start: usize,
    pub output_end: usize,
    pub emitted_texts: Vec<String>,
    pub rendered_output: String,
    pub average_qwen_score: f32,
    pub average_word_score: Option<f32>,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenPieceIpaSpan {
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_step_start: usize,
    pub ipa_step_end: usize,
    pub ipa_text: String,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenPiecePhones {
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_text: String,
    pub ipa_tokens: Vec<String>,
    pub normalized_phones: Vec<String>,
    pub ownership_score: f32,
}

#[derive(Debug, Clone, Facet)]
pub struct TokenPieceComparisonToken {
    pub word_index: Option<usize>,
    pub word_surface: Option<String>,
    pub token_index: usize,
    pub token: String,
    pub token_surface: String,
    pub token_char_start: usize,
    pub token_char_end: usize,
    pub ipa_text: String,
    pub comparison_token: String,
    pub ipa_source_start: usize,
    pub ipa_source_end: usize,
    pub ownership_score: f32,
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

pub fn summarize_probe_runs(result: &ProbeResult) -> Vec<ProbeTokenOwnershipRun> {
    let mut runs = Vec::new();

    let mut pending_rows: Vec<&ProbeAttentionRow> = Vec::new();

    for row in result.cross_attention.iter() {
        if row.output_piece == "</s>" {
            continue;
        }
        pending_rows.push(row);
        if row.emitted_text.is_empty() {
            continue;
        }

        let Some(anchor) = pending_rows
            .iter()
            .rev()
            .find(|row| row.top_qwen_piece_index.is_some())
            .copied()
        else {
            pending_rows.clear();
            continue;
        };
        let Some(qwen_piece_index) = anchor.top_qwen_piece_index else {
            pending_rows.clear();
            continue;
        };
        let qwen_piece_token = anchor.top_qwen_piece_token.clone().unwrap_or_default();
        let Some(piece_meta) = result
            .qwen_token_pieces
            .iter()
            .find(|piece| piece.index == qwen_piece_index)
        else {
            pending_rows.clear();
            continue;
        };

        let extends_last = runs.last().is_some_and(|run: &ProbeTokenOwnershipRun| {
            run.qwen_piece_index == qwen_piece_index && run.word_index == anchor.top_word_index
        });

        let run_start = pending_rows
            .first()
            .map(|row| row.output_index)
            .unwrap_or(anchor.output_index);
        let run_end = pending_rows
            .last()
            .map(|row| row.output_index + 1)
            .unwrap_or(anchor.output_index + 1);
        let step_count = pending_rows.len() as f32;
        let qwen_score_sum: f32 = pending_rows
            .iter()
            .filter_map(|row| row.top_qwen_piece_score)
            .sum();
        let word_score_values: Vec<f32> = pending_rows
            .iter()
            .filter_map(|row| row.top_word_score)
            .collect();
        let avg_qwen = if step_count > 0.0 {
            qwen_score_sum / step_count
        } else {
            0.0
        };
        let avg_word = if word_score_values.is_empty() {
            None
        } else {
            Some(word_score_values.iter().sum::<f32>() / word_score_values.len() as f32)
        };

        if extends_last {
            let run = runs.last_mut().expect("checked above");
            run.output_end = run_end;
            run.emitted_texts.push(row.emitted_text.clone());
            run.rendered_output.push_str(&row.emitted_text);
            let count = run.emitted_texts.len() as f32;
            run.average_qwen_score = ((run.average_qwen_score * (count - 1.0)) + avg_qwen) / count;
            run.average_word_score = match (run.average_word_score, avg_word) {
                (Some(prev), Some(score)) => Some(((prev * (count - 1.0)) + score) / count),
                (Some(prev), None) => Some(prev),
                (None, Some(score)) => Some(score),
                (None, None) => None,
            };
            pending_rows.clear();
            continue;
        }

        runs.push(ProbeTokenOwnershipRun {
            qwen_piece_index,
            qwen_piece_token,
            qwen_piece_surface: piece_meta.surface.clone(),
            word_index: anchor.top_word_index,
            word_surface: anchor.top_word_surface.clone(),
            output_start: run_start,
            output_end: run_end,
            emitted_texts: vec![row.emitted_text.clone()],
            rendered_output: row.emitted_text.clone(),
            average_qwen_score: avg_qwen,
            average_word_score: avg_word,
        });
        pending_rows.clear();
    }

    runs
}

pub fn token_piece_ipa_spans(result: &ProbeResult) -> Vec<TokenPieceIpaSpan> {
    summarize_probe_runs(result)
        .into_iter()
        .filter_map(|run| {
            let piece = result
                .qwen_token_pieces
                .iter()
                .find(|piece| piece.index == run.qwen_piece_index)?;
            Some(TokenPieceIpaSpan {
                word_index: run.word_index,
                word_surface: run.word_surface,
                token_index: run.qwen_piece_index,
                token: run.qwen_piece_token,
                token_surface: piece.surface.clone(),
                token_char_start: piece.char_start,
                token_char_end: piece.char_end,
                ipa_step_start: run.output_start,
                ipa_step_end: run.output_end,
                ipa_text: run.rendered_output,
                ownership_score: run.average_qwen_score,
            })
        })
        .collect()
}

pub fn token_piece_phones(result: &ProbeResult) -> Vec<TokenPiecePhones> {
    token_piece_ipa_spans(result)
        .into_iter()
        .map(|span| {
            let ipa_tokens = parse_reviewed_ipa(&span.ipa_text);
            let normalized_phones = normalize_ipa_for_comparison(&ipa_tokens);
            TokenPiecePhones {
                word_index: span.word_index,
                word_surface: span.word_surface,
                token_index: span.token_index,
                token: span.token,
                token_surface: span.token_surface,
                token_char_start: span.token_char_start,
                token_char_end: span.token_char_end,
                ipa_text: span.ipa_text,
                ipa_tokens,
                normalized_phones,
                ownership_score: span.ownership_score,
            }
        })
        .collect()
}

pub fn token_piece_comparison_tokens(result: &ProbeResult) -> Vec<TokenPieceComparisonToken> {
    token_piece_phones(result)
        .into_iter()
        .flat_map(|span| {
            normalize_ipa_for_comparison_with_spans(&span.ipa_tokens)
                .into_iter()
                .map(move |token| TokenPieceComparisonToken {
                    word_index: span.word_index,
                    word_surface: span.word_surface.clone(),
                    token_index: span.token_index,
                    token: span.token.clone(),
                    token_surface: span.token_surface.clone(),
                    token_char_start: span.token_char_start,
                    token_char_end: span.token_char_end,
                    ipa_text: span.ipa_text.clone(),
                    comparison_token: token.token,
                    ipa_source_start: token.source_start,
                    ipa_source_end: token.source_end,
                    ownership_score: span.ownership_score,
                })
        })
        .collect()
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
    use super::{
        ProbeAttentionRow, ProbeQwenPieceScore, ProbeQwenTokenPiece, ProbeRankedInput, ProbeResult,
        ProbeWordScore, ProbeWordSpan, summarize_probe_runs, token_piece_comparison_tokens,
        token_piece_ipa_spans, transcript_words,
    };

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

    #[test]
    fn summarize_probe_runs_groups_contiguous_rows_by_qwen_piece() {
        let result = ProbeResult {
            text: "Facet".to_string(),
            lang_code: "eng-us".to_string(),
            prompt: "<eng-us>: Facet".to_string(),
            device: "cpu".to_string(),
            text_bytes: vec![],
            word_spans: vec![ProbeWordSpan {
                index: 0,
                text: "Facet".to_string(),
                char_start: 0,
                char_end: 5,
                byte_start: 0,
                byte_end: 5,
            }],
            qwen_token_pieces: vec![
                ProbeQwenTokenPiece {
                    index: 0,
                    token: "Fac".to_string(),
                    char_start: 0,
                    char_end: 3,
                    surface: "Fac".to_string(),
                    byte_start: 0,
                    byte_end: 3,
                },
                ProbeQwenTokenPiece {
                    index: 1,
                    token: "et".to_string(),
                    char_start: 3,
                    char_end: 5,
                    surface: "et".to_string(),
                    byte_start: 3,
                    byte_end: 5,
                },
            ],
            input_ids: vec![],
            input_pieces: vec![],
            output_ids: vec![],
            output_pieces: vec![],
            decoded_output: "ˈfeɪsət".to_string(),
            cross_attention: vec![
                probe_row(2, "f", "f", 0, "Fac", 0.7),
                probe_row(3, "e", "e", 0, "Fac", 0.8),
                probe_row(4, "", "ɪ", 0, "Fac", 0.7),
                probe_row(6, "s", "s", 0, "Fac", 0.6),
                probe_row(7, "", "ə", 1, "et", 0.7),
                probe_row(9, "t", "t", 1, "et", 0.6),
            ],
        };

        let runs = summarize_probe_runs(&result);
        assert_eq!(runs.len(), 2);

        assert_eq!(runs[0].qwen_piece_index, 0);
        assert_eq!(runs[0].rendered_output, "feɪs");
        assert_eq!(runs[0].output_start, 2);
        assert_eq!(runs[0].output_end, 7);

        assert_eq!(runs[1].qwen_piece_index, 1);
        assert_eq!(runs[1].rendered_output, "ət");
        assert_eq!(runs[1].output_start, 7);
        assert_eq!(runs[1].output_end, 10);

        let spans = token_piece_ipa_spans(&result);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].token, "Fac");
        assert_eq!(spans[0].ipa_text, "feɪs");
        assert_eq!(spans[1].token, "et");
        assert_eq!(spans[1].ipa_text, "ət");

        let comparison = token_piece_comparison_tokens(&result);
        let rendered: Vec<_> = comparison
            .into_iter()
            .map(|token| {
                (
                    token.token,
                    token.comparison_token,
                    token.ipa_source_start,
                    token.ipa_source_end,
                )
            })
            .collect();
        assert_eq!(
            rendered,
            vec![
                ("Fac".to_string(), "f".to_string(), 0, 1),
                ("Fac".to_string(), "ɛ".to_string(), 1, 2),
                ("Fac".to_string(), "ɪ".to_string(), 1, 2),
                ("Fac".to_string(), "s".to_string(), 2, 3),
                ("et".to_string(), "ə".to_string(), 0, 1),
                ("et".to_string(), "t".to_string(), 1, 2),
            ]
        );
    }

    fn probe_row(
        output_index: usize,
        output_piece: &str,
        emitted_text: &str,
        qwen_piece_index: usize,
        qwen_piece_token: &str,
        score: f32,
    ) -> ProbeAttentionRow {
        ProbeAttentionRow {
            output_index,
            output_piece: output_piece.to_string(),
            emitted_text: emitted_text.to_string(),
            top_input_index: 0,
            top_input_piece: String::new(),
            top_score: 0.0,
            top_word_index: Some(0),
            top_word_surface: Some("Facet".to_string()),
            top_word_score: Some(score),
            word_scores: vec![ProbeWordScore {
                word_index: 0,
                word_text: "Facet".to_string(),
                char_start: 0,
                char_end: 5,
                byte_start: 0,
                byte_end: 5,
                score,
            }],
            top_qwen_piece_index: Some(qwen_piece_index),
            top_qwen_piece_token: Some(qwen_piece_token.to_string()),
            top_qwen_piece_score: Some(score),
            qwen_piece_scores: vec![ProbeQwenPieceScore {
                piece_index: qwen_piece_index,
                piece_token: qwen_piece_token.to_string(),
                piece_surface: qwen_piece_token.to_string(),
                char_start: 0,
                char_end: 0,
                byte_start: 0,
                byte_end: 0,
                score,
            }],
            ranked_inputs: vec![ProbeRankedInput {
                input_index: 0,
                input_piece: String::new(),
                score: 0.0,
            }],
        }
    }
}
