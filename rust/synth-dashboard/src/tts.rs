use anyhow::{anyhow, Result};
use ogg::{PacketReader, PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate, Channels, Decoder as OpusDecoder, Encoder as OpusEncoder};
use std::sync::Mutex;

const OPUS_SAMPLE_RATE: u32 = 48_000;
const OPUS_FRAME_SAMPLES: usize = 960; // 20ms @ 48kHz
const OPUS_MAX_PACKET_BYTES: usize = 4_000;
const OPUS_BITRATE_BPS: i32 = 96_000;

/// Encode f32 mono samples as Ogg Opus. Returns the Ogg container bytes.
pub async fn encode_ogg_opus(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let mono_48k = if sample_rate == OPUS_SAMPLE_RATE {
        samples.to_vec()
    } else {
        resample(samples, sample_rate, OPUS_SAMPLE_RATE)?
    };
    encode_ogg_opus_with_libopus(&mono_48k)
}

/// Decode an Ogg Opus payload into mono f32 samples.
pub fn decode_ogg_opus_mono(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    decode_ogg_opus_with_libopus(bytes)
}

/// Resample f32 mono samples to 16kHz. Returns samples unchanged if already 16kHz.
pub fn resample_to_16k(samples: &[f32], from_rate: u32) -> Result<Vec<f32>> {
    resample(samples, from_rate, 16_000)
}

fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }
    use rubato::{
        Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
    };
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        to_rate as f64 / from_rate as f64,
        2.0,
        params,
        samples.len(),
        1,
    )?;
    let output = resampler.process(&[samples], None)?;
    Ok(output.into_iter().next().unwrap_or_default())
}

fn encode_ogg_opus_with_libopus(samples: &[f32]) -> Result<Vec<u8>> {
    let mut encoder = OpusEncoder::new(OPUS_SAMPLE_RATE, Channels::Mono, Application::Audio)
        .map_err(|e| anyhow!("Opus encoder init: {e}"))?;
    encoder
        .set_bitrate(Bitrate::Bits(OPUS_BITRATE_BPS))
        .map_err(|e| anyhow!("Opus set bitrate: {e}"))?;
    encoder
        .set_vbr(true)
        .map_err(|e| anyhow!("Opus set vbr: {e}"))?;
    encoder
        .set_vbr_constraint(false)
        .map_err(|e| anyhow!("Opus set unconstrained vbr: {e}"))?;
    encoder
        .set_complexity(10)
        .map_err(|e| anyhow!("Opus set complexity: {e}"))?;

    let mut output = Vec::new();
    let mut writer = PacketWriter::new(&mut output);
    let serial = rand::random::<u32>().max(1);
    let pre_skip = encoder
        .get_lookahead()
        .map_err(|e| anyhow!("Opus get lookahead: {e}"))? as u16;

    writer.write_packet(
        build_opus_head(pre_skip, OPUS_SAMPLE_RATE, 1),
        serial,
        PacketWriteEndInfo::EndPage,
        0,
    )?;
    writer.write_packet(build_opus_tags(), serial, PacketWriteEndInfo::EndPage, 0)?;

    let total_samples = samples.len();
    let mut emitted_samples = 0usize;
    let mut packet = vec![0u8; OPUS_MAX_PACKET_BYTES];
    while emitted_samples < total_samples {
        let end = (emitted_samples + OPUS_FRAME_SAMPLES).min(total_samples);
        let mut frame = [0.0f32; OPUS_FRAME_SAMPLES];
        let frame_len = end - emitted_samples;
        frame[..frame_len].copy_from_slice(&samples[emitted_samples..end]);
        let encoded_len = encoder
            .encode_float(&frame, &mut packet)
            .map_err(|e| anyhow!("Opus encode: {e}"))?;
        emitted_samples = end;
        let granule_position = (pre_skip as u64) + (emitted_samples as u64);
        let end_info = if emitted_samples >= total_samples {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };
        writer.write_packet(
            packet[..encoded_len].to_vec(),
            serial,
            end_info,
            granule_position,
        )?;
    }

    Ok(output)
}

fn decode_ogg_opus_with_libopus(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    use std::io::Cursor;

    let mut reader = PacketReader::new(Cursor::new(bytes.to_vec()));
    let head = reader
        .read_packet()
        .map_err(|e| anyhow!("Ogg read OpusHead: {e}"))?
        .ok_or_else(|| anyhow!("Missing OpusHead packet"))?;
    let (channels, pre_skip) = parse_opus_head(&head.data)?;
    if channels != 1 {
        return Err(anyhow!(
            "Only mono Opus recordings are supported, got {channels} channels"
        ));
    }

    let tags = reader
        .read_packet()
        .map_err(|e| anyhow!("Ogg read OpusTags: {e}"))?
        .ok_or_else(|| anyhow!("Missing OpusTags packet"))?;
    if !tags.data.starts_with(b"OpusTags") {
        return Err(anyhow!("Invalid OpusTags packet"));
    }

    let mut decoder = OpusDecoder::new(OPUS_SAMPLE_RATE, Channels::Mono)
        .map_err(|e| anyhow!("Opus decoder init: {e}"))?;
    let mut decoded = Vec::new();
    let mut decoded_packet = vec![0.0f32; OPUS_FRAME_SAMPLES * 6];
    let mut final_granule = None;

    while let Some(packet) = reader
        .read_packet()
        .map_err(|e| anyhow!("Ogg read packet: {e}"))?
    {
        let len = decoder
            .decode_float(&packet.data, &mut decoded_packet, false)
            .map_err(|e| anyhow!("Opus decode: {e}"))?;
        decoded.extend_from_slice(&decoded_packet[..len]);
        final_granule = Some(packet.absgp_page());
    }

    let target_len = final_granule
        .map(|g| g.saturating_sub(pre_skip as u64) as usize)
        .unwrap_or_else(|| decoded.len().saturating_sub(pre_skip as usize));
    let skip = pre_skip as usize;
    if skip >= decoded.len() {
        return Ok((Vec::new(), OPUS_SAMPLE_RATE));
    }
    let available = decoded.len() - skip;
    let keep = target_len.min(available);
    Ok((decoded[skip..skip + keep].to_vec(), OPUS_SAMPLE_RATE))
}

fn build_opus_head(pre_skip: u16, input_sample_rate: u32, channels: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(19);
    out.extend_from_slice(b"OpusHead");
    out.push(1);
    out.push(channels);
    out.extend_from_slice(&pre_skip.to_le_bytes());
    out.extend_from_slice(&input_sample_rate.to_le_bytes());
    out.extend_from_slice(&0i16.to_le_bytes()); // output gain
    out.push(0); // channel mapping family 0: mono/stereo
    out
}

fn build_opus_tags() -> Vec<u8> {
    let vendor = b"hark synth-dashboard";
    let mut out = Vec::with_capacity(16 + vendor.len());
    out.extend_from_slice(b"OpusTags");
    out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    out.extend_from_slice(vendor);
    out.extend_from_slice(&0u32.to_le_bytes()); // user comment list length
    out
}

fn parse_opus_head(packet: &[u8]) -> Result<(u8, u16)> {
    if packet.len() < 19 || &packet[..8] != b"OpusHead" {
        return Err(anyhow!("Invalid OpusHead packet"));
    }
    let channels = packet[9];
    let pre_skip = u16::from_le_bytes([packet[10], packet[11]]);
    Ok((channels, pre_skip))
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
    fn generate_phonemes(&self, _phonemes: &str) -> Result<TtsAudio> {
        anyhow::bail!("{} does not support direct phoneme synthesis", self.name());
    }
}

struct PocketTtsBackend {
    model: pocket_tts::TTSModel,
    voice_state: pocket_tts::ModelState,
    sample_rate: u32,
}

impl LocalTtsBackend for PocketTtsBackend {
    fn name(&self) -> &'static str {
        "pocket"
    }

    fn generate(&self, text: &str) -> Result<TtsAudio> {
        let audio = self
            .model
            .generate(text, &self.voice_state)
            .map_err(|e| anyhow::anyhow!("pocket-tts: {e}"))?;
        let samples: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        Ok(TtsAudio {
            samples,
            sample_rate: self.sample_rate,
        })
    }
}

/// Pool of fully isolated pocket-tts workers.
/// Each worker has its own TTSModel on its own Metal device (separate command queue +
/// buffer pool), so concurrent calls are truly isolated — no shared GPU state.
struct PocketTtsPool {
    workers: Vec<Mutex<PocketTtsWorker>>,
    sample_rate: u32,
}

struct PocketTtsWorker {
    model: pocket_tts::TTSModel,
    voice_state: pocket_tts::ModelState,
}

impl LocalTtsBackend for PocketTtsPool {
    fn name(&self) -> &'static str {
        "pocket-hq"
    }

    fn generate(&self, text: &str) -> Result<TtsAudio> {
        // Try to find a free worker, otherwise block on the first one.
        let worker = self
            .workers
            .iter()
            .find_map(|w| w.try_lock().ok())
            .unwrap_or_else(|| self.workers[0].lock().unwrap());

        let audio = worker
            .model
            .generate(text, &worker.voice_state)
            .map_err(|e| anyhow::anyhow!("pocket-tts-hq: {e}"))?;
        let samples: Vec<f32> = audio.flatten_all()?.to_vec1()?;
        Ok(TtsAudio {
            samples,
            sample_rate: self.sample_rate,
        })
    }
}

struct KokoroBackend {
    model: Mutex<voice_tts::KokoroModel>,
    voice: mlx_rs::Array,
}

unsafe impl Sync for KokoroBackend {}

impl LocalTtsBackend for KokoroBackend {
    fn name(&self) -> &'static str {
        "kokoro"
    }

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
            if clean.is_empty() {
                continue;
            }
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

        self.generate_phonemes(&phonemes)
    }

    fn generate_phonemes(&self, phonemes: &str) -> Result<TtsAudio> {
        let phonemes = phonemes.trim();
        if phonemes.is_empty() {
            anyhow::bail!("cannot synthesize empty phoneme string");
        }
        let kokoro_phonemes = kokoro_phonemes_from_ipa(phonemes);
        eprintln!("kokoro ipa: raw={phonemes:?} kokoro={kokoro_phonemes:?}");
        let audio =
            voice_tts::generate(
                &mut self.model.lock().unwrap(),
                &kokoro_phonemes,
                &self.voice,
                1.0,
            )
            .map_err(|e| anyhow::anyhow!("kokoro: {e}"))?;
        audio.eval().map_err(|e| anyhow::anyhow!("mlx eval: {e}"))?;
        let samples: Vec<f32> = audio.as_slice().to_vec();
        Ok(TtsAudio {
            samples,
            sample_rate: 24000,
        })
    }
}

struct EspeakNgBackend;

fn normalize_ipa_for_kokoro(phonemes: &str) -> String {
    fn split_stress(token: &str) -> (String, String, String) {
        let chars: Vec<char> = token.chars().collect();
        let mut start = 0usize;
        let mut end = chars.len();
        while start < chars.len() && matches!(chars[start], 'ˈ' | 'ˌ') {
            start += 1;
        }
        while end > start && matches!(chars[end - 1], 'ˈ' | 'ˌ') {
            end -= 1;
        }
        (
            chars[..start].iter().collect(),
            chars[start..end].iter().collect(),
            chars[end..].iter().collect(),
        )
    }

    fn normalize_core_token(core: &str) -> Vec<String> {
        if core.is_empty() {
            return Vec::new();
        }
        let normalized = core
            .replace('g', "ɡ")
            .replace('r', "ɹ")
            .replace("ɚ", "əɹ")
            .replace("ɝ", "ɜɹ")
            .replace("ɜːɹ", "ɜɹ")
            .replace("ɜː", "ɜɹ")
            .replace("oʊ", "O")
            .replace("aɪ", "I")
            .replace("aʊ", "W")
            .replace("eɪ", "A")
            .replace("ɔɪ", "Y")
            .replace("dʒ", "ʤ")
            .replace("tʃ", "ʧ")
            .replace('ɾ', "T")
            .replace('ː', "");
        if normalized == "a" {
            return vec!["æ".to_string()];
        }
        if normalized == "g" {
            return vec!["ɡ".to_string()];
        }
        vec![normalized]
    }

    let mut raw_tokens: Vec<String> = phonemes
        .split_whitespace()
        .map(str::trim)
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect();

    let mut out = Vec::new();
    let mut pending_stress = String::new();
    let mut i = 0usize;
    while i < raw_tokens.len() {
        let token = raw_tokens[i].trim();
        let (prefix, core, suffix) = split_stress(token);
        if !prefix.is_empty() {
            pending_stress.push_str(&prefix);
        }
        let next_core = raw_tokens
            .get(i + 1)
            .map(|next| split_stress(next).1)
            .unwrap_or_default();
        let combined = match (core.as_str(), next_core.as_str()) {
            ("d", "ʒ") => Some("ʤ".to_string()),
            ("t", "ʃ") => Some("ʧ".to_string()),
            ("a", "ɪ") => Some("I".to_string()),
            ("a", "ʊ") => Some("W".to_string()),
            ("e", "ɪ") => Some("A".to_string()),
            ("o", "ʊ") => Some("O".to_string()),
            ("ɔ", "ɪ") => Some("Y".to_string()),
            _ => None,
        };
        if let Some(mut token_out) = combined {
            if !pending_stress.is_empty() {
                token_out = format!("{pending_stress}{token_out}");
                pending_stress.clear();
            }
            out.push(token_out);
            if !suffix.is_empty() {
                pending_stress.push_str(&suffix);
            }
            i += 2;
            continue;
        }

        for mut token_out in normalize_core_token(&core) {
            if token_out.is_empty() {
                continue;
            }
            if !pending_stress.is_empty() {
                token_out = format!("{pending_stress}{token_out}");
                pending_stress.clear();
            }
            out.push(token_out);
        }
        if !suffix.is_empty() {
            pending_stress.push_str(&suffix);
        }
        i += 1;
    }
    if !pending_stress.is_empty() {
        out.push(pending_stress);
    }
    out.join(" ")
}

pub fn kokoro_phonemes_from_ipa(phonemes: &str) -> String {
    normalize_ipa_for_kokoro(phonemes)
}

fn normalize_ipa_for_espeak(phonemes: &str) -> String {
    let compact = phonemes.split_whitespace().collect::<String>();
    compact
        .replace("oʊ", "əʊ")
        .replace("ɚ", "əɹ")
        .replace("ɝ", "ɜːɹ")
        .replace("g", "ɡ")
}

fn espeak_ipa_voice_params() -> espeak_ng::VoiceParams {
    espeak_ng::VoiceParams {
        speed_percent: 90,
        pitch_hz: 118,
        amplitude: 50,
        ..espeak_ng::VoiceParams::default()
    }
}

impl LocalTtsBackend for EspeakNgBackend {
    fn name(&self) -> &'static str {
        "espeak-ng"
    }

    fn generate(&self, _text: &str) -> Result<TtsAudio> {
        anyhow::bail!("espeak-ng text synthesis path is not wired yet")
    }

    fn generate_phonemes(&self, phonemes: &str) -> Result<TtsAudio> {
        let phonemes = phonemes.trim();
        if phonemes.is_empty() {
            anyhow::bail!("cannot synthesize empty phoneme string");
        }
        let compact = normalize_ipa_for_espeak(phonemes);
        if compact.is_empty() {
            anyhow::bail!("cannot synthesize empty phoneme string");
        }
        let voice = espeak_ipa_voice_params();
        let segments = espeak_ng::synthesize::engine::parse_ipa(&compact, &voice);
        eprintln!(
            "espeak-ng ipa: raw={phonemes:?} compact={compact:?} segments={} voice={{speed_percent:{},pitch_hz:{},amplitude:{}}}",
            segments.len(),
            voice.speed_percent,
            voice.pitch_hz,
            voice.amplitude
        );
        if segments.is_empty() {
            anyhow::bail!("espeak-ng: no recognisable phonemes in {:?}", compact);
        }
        let pcm = espeak_ng::synthesize::engine::synthesize_segments(&segments, &voice);
        if pcm.is_empty() {
            anyhow::bail!("espeak-ng: no samples produced");
        }
        let samples = pcm
            .into_iter()
            .map(|s| (s as f32) / 32768.0)
            .collect::<Vec<_>>();
        Ok(TtsAudio {
            samples,
            sample_rate: 22_050,
        })
    }
}

// ==================== Remote (async) backends ====================

/// Async TTS backend — makes HTTP calls, no mutable state needed
trait RemoteTtsBackend: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn generate(
        &self,
        text: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TtsAudio>> + Send + '_>>;
}

struct OpenAiTtsBackend {
    api_key: String,
    client: reqwest::Client,
}

impl RemoteTtsBackend for OpenAiTtsBackend {
    fn name(&self) -> &'static str {
        "openai"
    }

    fn generate(
        &self,
        text: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TtsAudio>> + Send + '_>> {
        let text = text.to_string();
        Box::pin(async move {
            let resp = self
                .client
                .post("https://api.openai.com/v1/audio/speech")
                .bearer_auth(&self.api_key)
                .json(&serde_json::json!({
                    "model": "tts-1",
                    "input": text,
                    "voice": "alloy",
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
            Ok(TtsAudio {
                samples,
                sample_rate: 24000,
            })
        })
    }
}

struct ElevenLabsTtsBackend {
    api_key: String,
    voice_id: String,
    client: reqwest::Client,
}

impl RemoteTtsBackend for ElevenLabsTtsBackend {
    fn name(&self) -> &'static str {
        "elevenlabs"
    }

    fn generate(
        &self,
        text: &str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<TtsAudio>> + Send + '_>> {
        let text = text.to_string();
        Box::pin(async move {
            let url = format!(
                "https://api.elevenlabs.io/v1/text-to-speech/{}",
                self.voice_id
            );
            let resp = self
                .client
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
                Err(e) => {
                    errors.push(format!("{p}: {e}"));
                    continue;
                }
                Ok(bytes) => {
                    let content = String::from_utf8_lossy(&bytes);
                    let mut dict = HashSet::new();
                    for line in content.lines() {
                        if line.starts_with(";;;") || line.is_empty() {
                            continue;
                        }
                        if let Some(word) = line.split_whitespace().next() {
                            let word = word.split('(').next().unwrap_or(word);
                            dict.insert(word.to_lowercase());
                        }
                    }
                    eprintln!("CMUdict loaded: {} words from {p}", dict.len());
                    return dict;
                }
            }
        }
        panic!("CMUdict not found:\n{}", errors.join("\n"));
    });
}

fn cmudict() -> &'static HashSet<String> {
    CMUDICT
        .get()
        .expect("CMUdict not initialized — call tts::init_cmudict() at startup")
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
    parts.len() >= 2
        && parts.iter().all(|p| {
            // Each part is digits, or digits followed by -prerelease
            let base = p.split('-').next().unwrap_or(p);
            !base.is_empty() && base.chars().all(|c| c.is_ascii_digit())
        })
}

pub fn detect_unknown_words(text: &str) -> Vec<String> {
    let dict = cmudict();
    let mut unknown = Vec::new();

    for word in extract_words(text) {
        if !synth_textgen::corpus::is_valid_vocab_term(&word) {
            continue;
        }
        let lower = word.to_lowercase();

        if dict.contains(&lower) {
            continue;
        }
        if let Some(stem) = lower.strip_suffix("'s") {
            if dict.contains(stem) {
                continue;
            }
        }
        if let Some(stem) = lower.strip_suffix('s') {
            if dict.contains(stem) {
                continue;
            }
        }
        // Check if it's a compound of two known words (e.g., "roadmap" = "road" + "map")
        let is_compound = lower
            .char_indices()
            .skip(2)
            .take_while(|(i, _)| *i + 2 < lower.len())
            .any(|(i, _)| dict.contains(&lower[..i]) && dict.contains(&lower[i..]));
        if is_compound {
            continue;
        }
        if word.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if lower.starts_with("0x") {
            continue;
        } // hex constants like 0xDEAD
        if lower.chars().all(|c| c.is_ascii_hexdigit()) {
            continue;
        } // hex-only (short git commits etc.)
        if is_semver(&lower) {
            continue;
        } // semver like 1.2.3, v0.1.0

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
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / max)
                .collect()
        }
    };
    Ok(TtsAudio {
        samples,
        sample_rate: spec.sample_rate,
    })
}

// ==================== Manager ====================

/// Holds all TTS backends. Local backends are Arc-wrapped so they can be
/// sent into spawn_blocking without blocking the async runtime.
pub struct TtsManager {
    local: Vec<std::sync::Arc<dyn LocalTtsBackend>>,
    remote: Vec<Box<dyn RemoteTtsBackend>>,
}

impl TtsManager {
    pub fn available_backends(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = self.local.iter().map(|b| b.name()).collect();
        names.extend(self.remote.iter().map(|b| b.name()));
        names
    }

    /// Generate audio. Local backends run in spawn_blocking, remote backends run async.
    pub async fn generate(&self, backend_name: &str, text: &str) -> Result<TtsAudio> {
        for local in &self.local {
            if local.name() == backend_name {
                let local = local.clone();
                let text = text.to_string();
                return tokio::task::spawn_blocking(move || local.generate(&text))
                    .await
                    .map_err(|e| anyhow::anyhow!("spawn_blocking: {e}"))?;
            }
        }

        for remote in &self.remote {
            if remote.name() == backend_name {
                return remote.generate(text).await;
            }
        }

        anyhow::bail!("TTS backend '{backend_name}' not available")
    }

    pub async fn generate_phonemes(
        &self,
        backend_name: &str,
        phonemes: &str,
    ) -> Result<TtsAudio> {
        for local in &self.local {
            if local.name() == backend_name {
                let local = local.clone();
                let phonemes = phonemes.to_string();
                return tokio::task::spawn_blocking(move || local.generate_phonemes(&phonemes))
                    .await
                    .map_err(|e| anyhow::anyhow!("spawn_blocking: {e}"))?;
            }
        }

        anyhow::bail!("TTS backend '{backend_name}' not available for phoneme synthesis")
    }
}

/// Build a TtsManager with all available backends
pub fn init(voice_path: &str, kokoro_voice: &str, tts_workers: usize) -> TtsManager {
    let mut local: Vec<std::sync::Arc<dyn LocalTtsBackend>> = Vec::new();
    let mut remote: Vec<Box<dyn RemoteTtsBackend>> = Vec::new();

    // Pocket-tts HQ pool — N isolated Metal workers
    match PocketTtsPool::load(voice_path, tts_workers) {
        Ok(pool) => {
            eprintln!(
                "pocket-tts-hq ready ({tts_workers} workers, {} Hz)",
                pool.sample_rate
            );
            local.push(std::sync::Arc::new(pool));
        }
        Err(e) => eprintln!("pocket-tts-hq not available: {e}"),
    }

    match KokoroBackend::load(kokoro_voice) {
        Ok(backend) => {
            local.push(std::sync::Arc::new(backend));
        }
        Err(e) => eprintln!("kokoro not available: {e}"),
    }

    local.push(std::sync::Arc::new(EspeakNgBackend));

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
        Ok(Self {
            model,
            voice_state,
            sample_rate,
        })
    }
}

impl PocketTtsPool {
    fn load(voice_path: &str, num_workers: usize) -> Result<Self> {
        let num_workers = num_workers.max(1);
        eprintln!("Loading pocket-tts ({num_workers} workers, each on its own Metal device)...");

        let mut workers = Vec::with_capacity(num_workers);
        let mut sample_rate = 0u32;

        for i in 0..num_workers {
            let device = candle_core::Device::new_metal(0)
                .map_err(|e| anyhow::anyhow!("Metal device init for worker {i}: {e}"))?;
            let model = pocket_tts::TTSModel::load_with_params_device(
                "b6369a24",
                pocket_tts::config::defaults::TEMPERATURE,
                pocket_tts::config::defaults::LSD_DECODE_STEPS,
                pocket_tts::config::defaults::EOS_THRESHOLD,
                None,
                &device,
            )?;
            let voice_state = model
                .get_voice_state(voice_path)
                .map_err(|e| anyhow::anyhow!("worker {i} voice '{voice_path}': {e}"))?;
            sample_rate = model.sample_rate as u32;
            eprintln!(
                "  worker {i} ready ({sample_rate} Hz, Metal device {:?})",
                device
            );
            workers.push(Mutex::new(PocketTtsWorker { model, voice_state }));
        }

        Ok(Self {
            workers,
            sample_rate,
        })
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
        Ok(Self {
            model: Mutex::new(model),
            voice,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{decode_ogg_opus_mono, encode_ogg_opus, kokoro_phonemes_from_ipa};
    use std::f32::consts::TAU;

    #[tokio::test]
    async fn ogg_opus_round_trip_produces_audio() {
        let sample_rate = 16_000;
        let samples: Vec<f32> = (0..sample_rate)
            .map(|n| (TAU * 440.0 * n as f32 / sample_rate as f32).sin() * 0.2)
            .collect();

        let encoded = encode_ogg_opus(&samples, sample_rate).await.unwrap();
        assert!(
            encoded.len() > 1000,
            "encoded file should not be header-only"
        );

        let (decoded, decoded_rate) = decode_ogg_opus_mono(&encoded).unwrap();
        assert_eq!(decoded_rate, 48_000);
        assert!(decoded.len() > 40_000);
        assert!(decoded.iter().map(|s| s.abs()).fold(0.0f32, f32::max) > 0.05);
    }

    #[test]
    fn kokoro_normalization_keeps_phone_boundaries() {
        assert_eq!(kokoro_phonemes_from_ipa("ɹ ɪ p ˈg ɹ a b"), "ɹ ɪ p ˈɡ ɹ æ b");
    }

    #[test]
    fn kokoro_normalization_collapses_common_sequences() {
        assert_eq!(kokoro_phonemes_from_ipa("g o ʊ"), "ɡ O");
        assert_eq!(kokoro_phonemes_from_ipa("d ʒ ʌ m p"), "ʤ ʌ m p");
        assert_eq!(kokoro_phonemes_from_ipa("t ʃ ɪ p"), "ʧ ɪ p");
    }

    #[test]
    fn kokoro_normalization_preserves_stress_on_next_phone() {
        assert_eq!(kokoro_phonemes_from_ipa("h ə l ˈo ʊ"), "h ə l ˈO");
    }
}
