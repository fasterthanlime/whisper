use anyhow::Result;
use std::sync::Mutex;

/// Resample f32 mono samples to 16kHz. Returns samples unchanged if already 16kHz.
pub fn resample_to_16k(samples: &[f32], from_rate: u32) -> Result<Vec<f32>> {
    if from_rate == 16000 {
        return Ok(samples.to_vec());
    }
    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        16000.0 / from_rate as f64, 2.0, params, samples.len(), 1,
    )?;
    let output = resampler.process(&[samples], None)?;
    Ok(output.into_iter().next().unwrap_or_default())
}

/// Replace one word in a spoken form string (case-insensitive match, preserving surrounding punctuation).
/// Canonicalize a word for vocab lookup: strip non-alphanumeric (keep apostrophes), lowercase.
fn canon(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || *c == '\'')
        .collect::<String>()
        .to_lowercase()
}

/// A token is either a word (alphanumeric + apostrophes) or trivia (everything else).
#[derive(Debug)]
enum Token<'a> {
    Word(&'a str),
    Trivia(&'a str),
}

/// Tokenize text into alternating Word and Trivia segments.
/// "PR, please." → [Word("PR"), Trivia(", "), Word("please"), Trivia(".")]
fn tokenize(text: &str) -> Vec<Token<'_>> {
    let mut tokens = Vec::new();
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < text.len() {
        let ch = text[i..].chars().next().unwrap();
        if ch.is_alphanumeric() || ch == '\'' {
            // Word: consume alphanumeric + apostrophes
            let start = i;
            while i < text.len() {
                let c = text[i..].chars().next().unwrap();
                if c.is_alphanumeric() || c == '\'' {
                    i += c.len_utf8();
                } else {
                    break;
                }
            }
            tokens.push(Token::Word(&text[start..i]));
        } else {
            // Trivia: consume everything else
            let start = i;
            while i < text.len() {
                let c = text[i..].chars().next().unwrap();
                if c.is_alphanumeric() || c == '\'' {
                    break;
                }
                i += c.len_utf8();
            }
            tokens.push(Token::Trivia(&text[start..i]));
        }
    }
    let _ = bytes; // suppress unused warning
    tokens
}

/// Build a spoken form from original text by applying vocab overrides.
/// Tokenizes into Word/Trivia segments, looks up each word in the overrides map,
/// and substitutes matches while preserving all trivia (punctuation, spaces) exactly.
pub fn build_spoken_form(
    text: &str,
    overrides: &std::collections::HashMap<String, String>,
) -> String {
    // Pre-canonicalize override keys for O(1) lookup
    let canon_map: std::collections::HashMap<String, &str> = overrides
        .iter()
        .map(|(term, spoken)| (canon(term), spoken.as_str()))
        .collect();

    let tokens = tokenize(text);
    let mut result = String::with_capacity(text.len());

    for token in &tokens {
        match token {
            Token::Word(w) => {
                let key = canon(w);
                if let Some(&replacement) = canon_map.get(&key) {
                    result.push_str(replacement);
                } else {
                    result.push_str(w);
                }
            }
            Token::Trivia(t) => result.push_str(t),
        }
    }
    result
}

/// Raw audio output from a TTS backend
pub struct TtsAudio {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl TtsAudio {
    /// Peak-normalize to -1 dB
    pub fn normalize(&mut self) {
        let peak = self.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        if peak > 0.0 {
            let target = 10.0f32.powf(-1.0 / 20.0);
            let gain = target / peak;
            for s in &mut self.samples {
                *s *= gain;
            }
        }
    }

    /// Encode as WAV bytes
    pub fn to_wav(&self) -> Result<Vec<u8>> {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &s in &self.samples {
            let clamped = (s * 32767.0f32).clamp(-32768.0, 32767.0);
            writer.write_sample(clamped as i16)?;
        }
        writer.finalize()?;
        Ok(cursor.into_inner())
    }
}

// ==================== Local (sync) backends ====================

/// Sync TTS backend — runs on a blocking thread
trait LocalTtsBackend: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Generate audio. Takes &self — implementations handle their own concurrency.
    fn generate(&self, text: &str) -> Result<TtsAudio>;
}

struct PocketTtsBackend {
    model: pocket_tts::TTSModel,
    voice_state: pocket_tts::ModelState,
    sample_rate: u32,
}

impl LocalTtsBackend for PocketTtsBackend {
    fn name(&self) -> &'static str { "pocket" }

    fn generate(&self, text: &str) -> Result<TtsAudio> {
        let audio = self.model.generate(text, &self.voice_state)
            .map_err(|e| anyhow::anyhow!("pocket-tts: {e}"))?;
        let samples: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        Ok(TtsAudio { samples, sample_rate: self.sample_rate })
    }
}

/// Pool of pocket-tts workers sharing model weights via Arc.
/// Each worker has its own voice_state. generate() picks a free worker.
struct PocketTtsPool {
    workers: Vec<Mutex<PocketTtsWorker>>,
    sample_rate: u32,
}

struct PocketTtsWorker {
    model: std::sync::Arc<pocket_tts::TTSModel>,
    voice_state: pocket_tts::ModelState,
}

impl LocalTtsBackend for PocketTtsPool {
    fn name(&self) -> &'static str { "pocket-hq" }

    fn generate(&self, text: &str) -> Result<TtsAudio> {
        // Find an unlocked worker, or wait on the first one
        let worker_mutex: &Mutex<PocketTtsWorker> = self.workers.iter()
            .find(|w| w.try_lock().is_ok())
            .unwrap_or(&self.workers[0]);
        let worker = worker_mutex.lock().unwrap();
        let audio = worker.model.generate(text, &worker.voice_state)
            .map_err(|e| anyhow::anyhow!("pocket-tts-hq: {e}"))?;
        let samples: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        Ok(TtsAudio { samples, sample_rate: self.sample_rate })
    }
}

struct KokoroBackend {
    model: Mutex<voice_tts::KokoroModel>,
    voice: mlx_rs::Array,
}

unsafe impl Sync for KokoroBackend {}

impl LocalTtsBackend for KokoroBackend {
    fn name(&self) -> &'static str { "kokoro" }

    fn generate(&self, text: &str) -> Result<TtsAudio> {
        // Use espeak-ng directly for the full sentence — the voice-g2p pipeline
        // drops unknown words and botches contractions.
        let fallback = voice_g2p::espeak::EspeakFallback::new();
        if !fallback.is_available() {
            anyhow::bail!("espeak-ng not installed (brew install espeak-ng)");
        }

        // Phonemize each word via espeak and join
        let mut phoneme_parts = Vec::new();
        for word in text.split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
            if clean.is_empty() { continue; }
            match fallback.convert_word(clean) {
                Some((ph, _)) => phoneme_parts.push(ph),
                None => {
                    // Last resort: try the full G2P pipeline for this word
                    if let Ok(ph) = voice_g2p::english_to_phonemes(clean) {
                        let ph = ph.trim().to_string();
                        if !ph.is_empty() {
                            phoneme_parts.push(ph);
                        }
                    }
                }
            }
        }
        let phonemes = phoneme_parts.join(" ");
        eprintln!("kokoro espeak: {text:?} → {phonemes:?}");

        let audio = voice_tts::generate(&mut self.model.lock().unwrap(), &phonemes, &self.voice, 1.0)
            .map_err(|e| anyhow::anyhow!("kokoro: {e}"))?;
        audio.eval().map_err(|e| anyhow::anyhow!("mlx eval: {e}"))?;
        let samples: Vec<f32> = audio.as_slice().to_vec();
        Ok(TtsAudio { samples, sample_rate: 24000 })
    }
}

// ==================== Remote (async) backends ====================

/// Async TTS backend — makes HTTP calls, no mutable state needed
trait RemoteTtsBackend: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn generate(&self, text: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TtsAudio>> + Send + '_>>;
}

struct OpenAiTtsBackend {
    api_key: String,
    client: reqwest::Client,
}

impl RemoteTtsBackend for OpenAiTtsBackend {
    fn name(&self) -> &'static str { "openai" }

    fn generate(&self, text: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TtsAudio>> + Send + '_>> {
        let text = text.to_string();
        Box::pin(async move {
            let resp = self.client
                .post("https://api.openai.com/v1/audio/speech")
                .bearer_auth(&self.api_key)
                .json(&serde_json::json!({
                    "model": "tts-1-hd",
                    "input": text,
                    "voice": "onyx",
                    "response_format": "pcm",
                }))
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("OpenAI TTS {status}: {body}");
            }

            // pcm format = raw 16-bit little-endian PCM at 24kHz
            let pcm_bytes = resp.bytes().await?;
            let samples: Vec<f32> = pcm_bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    sample as f32 / 32768.0
                })
                .collect();
            Ok(TtsAudio { samples, sample_rate: 24000 })
        })
    }
}

struct ElevenLabsTtsBackend {
    api_key: String,
    voice_id: String,
    client: reqwest::Client,
}

impl RemoteTtsBackend for ElevenLabsTtsBackend {
    fn name(&self) -> &'static str { "elevenlabs" }

    fn generate(&self, text: &str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TtsAudio>> + Send + '_>> {
        let text = text.to_string();
        Box::pin(async move {
            let url = format!("https://api.elevenlabs.io/v1/text-to-speech/{}", self.voice_id);
            let resp = self.client
                .post(&url)
                .header("xi-api-key", &self.api_key)
                .header("Accept", "audio/wav")
                .json(&serde_json::json!({
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                }))
                .send()
                .await?;

            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                anyhow::bail!("ElevenLabs TTS {status}: {body}");
            }

            let wav_bytes = resp.bytes().await?;
            decode_wav(&wav_bytes)
        })
    }
}

// ==================== Unknown word detection via CMUdict ====================

use std::collections::HashSet;
use std::sync::OnceLock;

static CMUDICT: OnceLock<HashSet<String>> = OnceLock::new();

/// Must be called at startup. Panics if CMUdict can't be found.
pub fn init_cmudict() {
    CMUDICT.get_or_init(|| {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let candidates = [
            "data/cmudict.txt".to_string(),
            format!("{home}/bearcove/hark/asr-synth/data/cmudict.txt"),
        ];
        let mut errors = Vec::new();
        for p in &candidates {
            match std::fs::read(p) {
            Err(e) => { errors.push(format!("{p}: {e}")); continue; }
            Ok(bytes) => { let content = String::from_utf8_lossy(&bytes);
                let mut dict = HashSet::new();
                for line in content.lines() {
                    if line.starts_with(";;;") || line.is_empty() { continue; }
                    if let Some(word) = line.split_whitespace().next() {
                        let word = word.split('(').next().unwrap_or(word);
                        dict.insert(word.to_lowercase());
                    }
                }
                eprintln!("CMUdict loaded: {} words from {p}", dict.len());
                return dict;
            }}
        }
        panic!("CMUdict not found:\n{}", errors.join("\n"));
    });
}

fn cmudict() -> &'static HashSet<String> {
    CMUDICT.get().expect("CMUdict not initialized — call tts::init_cmudict() at startup")
}

/// Check which words in a sentence are NOT in CMUdict.
/// These are likely technical terms that ASR will struggle with.
/// Canonical word extraction: split text into clean words, stripping punctuation.
/// Used everywhere — CMUdict lookup, unknown detection, frontend matching.
pub fn extract_words(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() >= 2)
        .collect()
}

/// Check if a string looks like a semver version (e.g. "1.2.3", "v0.1.0", "2.0.0-rc.1")
fn is_semver(s: &str) -> bool {
    let s = s.strip_prefix('v').unwrap_or(s);
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() >= 2 && parts.iter().all(|p| {
        // Each part is digits, or digits followed by -prerelease
        let base = p.split('-').next().unwrap_or(p);
        !base.is_empty() && base.chars().all(|c| c.is_ascii_digit())
    })
}

pub fn detect_unknown_words(text: &str) -> Vec<String> {
    let dict = cmudict();
    let mut unknown = Vec::new();

    for word in extract_words(text) {
        if !synth_textgen::corpus::is_valid_vocab_term(&word) { continue; }
        let lower = word.to_lowercase();

        if dict.contains(&lower) { continue; }
        if let Some(stem) = lower.strip_suffix("'s") { if dict.contains(stem) { continue; } }
        if let Some(stem) = lower.strip_suffix('s') { if dict.contains(stem) { continue; } }
        // Check if it's a compound of two known words (e.g., "roadmap" = "road" + "map")
        let is_compound = lower.char_indices()
            .skip(2)
            .take_while(|(i, _)| *i + 2 < lower.len())
            .any(|(i, _)| dict.contains(&lower[..i]) && dict.contains(&lower[i..]));
        if is_compound { continue; }
        if word.chars().all(|c| c.is_ascii_digit()) { continue; }
        if lower.starts_with("0x") { continue; } // hex constants like 0xDEAD
        if lower.chars().all(|c| c.is_ascii_hexdigit()) { continue; } // hex-only (short git commits etc.)
        if is_semver(&lower) { continue; } // semver like 1.2.3, v0.1.0

        unknown.push(word);
    }

    unknown.sort();
    unknown.dedup();
    unknown
}

// ==================== Helpers ====================

fn decode_wav(wav_bytes: &[u8]) -> Result<TtsAudio> {
    let cursor = std::io::Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            reader.samples::<f32>().filter_map(|s| s.ok()).collect()
        }
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max).collect()
        }
    };
    Ok(TtsAudio { samples, sample_rate: spec.sample_rate })
}

// ==================== Manager ====================

/// Holds all TTS backends. Local backends handle their own concurrency.
pub struct TtsManager {
    local: Vec<Box<dyn LocalTtsBackend>>,
    remote: Vec<Box<dyn RemoteTtsBackend>>,
}

impl TtsManager {
    pub fn available_backends(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.local.iter()
            .map(|b| b.name())
            .collect();
        names.extend(self.remote.iter().map(|b| b.name()));
        names
    }

    /// Generate audio. Local backends run synchronously, remote backends run async.
    pub async fn generate(&self, backend_name: &str, text: &str) -> Result<TtsAudio> {
        for local in &self.local {
            if local.name() == backend_name {
                return local.generate(text);
            }
        }

        for remote in &self.remote {
            if remote.name() == backend_name {
                return remote.generate(text).await;
            }
        }

        anyhow::bail!("TTS backend '{backend_name}' not available")
    }
}

/// Number of parallel pocket-tts workers (share weights, separate state).
const POCKET_TTS_WORKERS: usize = 4;

/// Build a TtsManager with all available backends
pub fn init(voice_path: &str, _kokoro_voice: &str) -> TtsManager {
    let mut local: Vec<Box<dyn LocalTtsBackend>> = Vec::new();
    let mut remote: Vec<Box<dyn RemoteTtsBackend>> = Vec::new();

    // Pocket-tts HQ pool — N workers sharing one model
    match PocketTtsPool::load(voice_path, POCKET_TTS_WORKERS) {
        Ok(pool) => {
            eprintln!("pocket-tts-hq ready ({} workers, {} Hz)", POCKET_TTS_WORKERS, pool.sample_rate);
            local.push(Box::new(pool));
        }
        Err(e) => eprintln!("pocket-tts-hq not available: {e}"),
    }

    // OpenAI
    if std::env::var("OPENAI_API_KEY").is_ok() {
        let api_key = std::env::var("OPENAI_API_KEY").unwrap();
        remote.push(Box::new(OpenAiTtsBackend {
            api_key,
            client: reqwest::Client::new(),
        }));
        eprintln!("OpenAI TTS ready");
    }

    // ElevenLabs
    if std::env::var("ELEVENLABS_API_KEY").is_ok() {
        let api_key = std::env::var("ELEVENLABS_API_KEY").unwrap();
        // Default to "Brian" — a pre-made voice available on all plans
        let voice_id = std::env::var("ELEVENLABS_VOICE_ID")
            .unwrap_or_else(|_| "nPczCjzI2devNBz1zQrb".to_string());
        remote.push(Box::new(ElevenLabsTtsBackend {
            api_key,
            voice_id,
            client: reqwest::Client::new(),
        }));
        eprintln!("ElevenLabs TTS ready");
    }

    TtsManager { local, remote }
}

// Keep PocketTtsBackend::load and KokoroBackend::load as private helpers
impl PocketTtsBackend {
    fn load(voice_path: &str) -> Result<Self> {
        eprintln!("Loading pocket-tts (quantized)...");
        let model = pocket_tts::TTSModel::load_quantized("b6369a24")?;
        let voice_state = model
            .get_voice_state(voice_path)
            .map_err(|e| anyhow::anyhow!("loading voice '{voice_path}': {e}"))?;
        let sample_rate = model.sample_rate as u32;
        eprintln!("pocket-tts ready ({sample_rate} Hz)");
        Ok(Self { model, voice_state, sample_rate })
    }
}

impl PocketTtsPool {
    fn load(voice_path: &str, num_workers: usize) -> Result<Self> {
        eprintln!("Loading pocket-tts (full precision)...");
        let model = pocket_tts::TTSModel::load("b6369a24")?;
        let voice_state = model
            .get_voice_state(voice_path)
            .map_err(|e| anyhow::anyhow!("loading voice '{voice_path}': {e}"))?;
        let sample_rate = model.sample_rate as u32;

        let model = std::sync::Arc::new(model);
        let workers: Vec<Mutex<PocketTtsWorker>> = (0..num_workers)
            .map(|_| Mutex::new(PocketTtsWorker {
                model: model.clone(),
                voice_state: voice_state.clone(),
            }))
            .collect();

        Ok(Self { workers, sample_rate })
    }
}

impl KokoroBackend {
    fn load(voice_name: &str) -> Result<Self> {
        eprintln!("Loading Kokoro TTS...");
        let model = voice_tts::load_model("prince-canuma/Kokoro-82M")
            .map_err(|e| anyhow::anyhow!("kokoro model: {e}"))?;
        let voice = voice_tts::load_voice(voice_name, None)
            .map_err(|e| anyhow::anyhow!("kokoro voice '{voice_name}': {e}"))?;
        eprintln!("Kokoro ready (24000 Hz, voice: {voice_name})");
        Ok(Self { model: Mutex::new(model), voice })
    }
}
