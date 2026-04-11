use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use bee_qwen3_asr::config::AsrConfig;
use bee_qwen3_asr::load;
use bee_qwen3_asr::mlx_rs::module::ModuleParametersExt;
use bee_qwen3_asr::model::Qwen3ASRModel;
use bee_roll::{Cutting, FeedOutput, Utterance, ZipaTiming};
use bee_zipa_mlx::audio::load_wav_mono_f32;
use tokenizers::Tokenizer;

const CHUNK_SAMPLES: usize = 3_200;

#[test]
fn real_preview_snapshot_8453579b() -> anyhow::Result<()> {
    insta::assert_snapshot!(
        "real_preview_snapshot_8453579b",
        snapshot_for_artifact("8453579B")?
    );
    Ok(())
}

#[test]
fn real_preview_snapshot_1dce2f4b() -> anyhow::Result<()> {
    insta::assert_snapshot!(
        "real_preview_snapshot_1dce2f4b",
        snapshot_for_artifact("1DCE2F4B")?
    );
    Ok(())
}

fn snapshot_for_artifact(stem: &str) -> anyhow::Result<String> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let wav_path = fixtures.join(format!("{stem}.wav"));
    let transcript_path = fixtures.join(format!("{stem}.transcript.txt"));

    let samples = load_wav_mono_f32(&wav_path)?;
    let expected_text = extract_transcript_text(&transcript_path)?;

    let model_dir = PathBuf::from(env::var("BEE_ASR_MODEL_DIR")?);
    let tokenizer_path = PathBuf::from(env::var("BEE_TOKENIZER_PATH")?);
    let g2p_model_dir = g2p_model_dir()?;
    let zipa_bundle_dir = zipa_bundle_dir()?;

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))?;
    let asr_config = AsrConfig::from_file(&model_dir.join("config.json"))?;
    let mut model = Qwen3ASRModel::new(&asr_config.thinker_config)?;
    load::load_weights(&mut model, &model_dir)?;
    model.eval()?;
    let model = Arc::new(model);

    let mut utterance = Utterance::new(
        asr_config.thinker_config.text_config.num_hidden_layers,
        Cutting::Never,
    );
    utterance.attach_qwen_asr(
        model,
        &tokenizer_path,
        asr_config.thinker_config.audio_config.num_mel_bins,
        "en",
    );
    utterance.attach_phonetics(&g2p_model_dir, &tokenizer_path, &zipa_bundle_dir, "eng-us")?;

    let mut snapshot = String::new();
    snapshot.push_str(&format!("wav: {}\n", wav_path.display()));
    snapshot.push_str(&format!("expected: {expected_text}\n"));
    snapshot.push('\n');

    for (feed_index, chunk) in samples.samples.chunks(CHUNK_SAMPLES).enumerate() {
        let output = utterance.feed(chunk.to_vec());
        render_feed(
            &mut snapshot,
            &tokenizer,
            feed_index + 1,
            (feed_index + 1) * CHUNK_SAMPLES,
            output,
        )?;
    }

    Ok(snapshot)
}

fn render_feed(
    out: &mut String,
    tokenizer: &Tokenizer,
    feed_index: usize,
    sample_count: usize,
    output: FeedOutput<'_>,
) -> anyhow::Result<()> {
    let ids = output
        .tokens()
        .iter()
        .map(|token| token.timed_token().token().as_u32())
        .collect::<Vec<_>>();
    let transcript = tokenizer
        .decode(&ids, true)
        .map_err(|e| anyhow::anyhow!("decoding transcript: {e}"))?;

    out.push_str(&format!(
        "feed {feed_index:02}  samples={sample_count}  secs={:.3}\n",
        sample_count as f64 / 16_000.0
    ));
    out.push_str(&format!(
        "lang {}\n",
        output.detected_language().unwrap_or("<none>")
    ));
    out.push_str(&format!("text {transcript}\n"));

    for token in output.tokens() {
        let timed = token.timed_token();
        let surface = timed
            .token()
            .decode()
            .unwrap_or_else(|_| "<decode-error>".to_owned())
            .replace('\n', "\\n");
        let conf = token
            .asr_confidence()
            .map(|confidence| {
                format!(
                    "{:.3}/{:.3}",
                    confidence.margin(),
                    confidence.concentration()
                )
            })
            .unwrap_or_else(|| "-".to_owned());
        let g2p = token
            .g2p_ipa()
            .map(|ipa| ipa.as_str().to_owned())
            .unwrap_or_else(|| "-".to_owned());
        let phones = if token.transcript_phones().is_empty() {
            "-".to_owned()
        } else {
            token
                .transcript_phones()
                .iter()
                .map(|phone| phone.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        };
        out.push_str(&format!(
            "{:>2} {:<12} conf={} g2p={} phones={} zipa={}\n",
            timed.index().as_usize(),
            quote_surface(&surface),
            conf,
            g2p,
            phones,
            format_zipa_timing(token.zipa_timing()),
        ));
    }

    out.push('\n');
    Ok(())
}

fn format_zipa_timing(timing: &ZipaTiming) -> String {
    match timing {
        ZipaTiming::Aligned(range) => {
            format!("{:.3}..{:.3}", range.start.as_secs(), range.end.as_secs())
        }
        ZipaTiming::Deleted { projected_at } => format!("del@{projected_at}"),
        ZipaTiming::Projected {
            normalized_start,
            normalized_end,
        } => format!("proj {normalized_start}..{normalized_end}"),
        ZipaTiming::Invalid => "invalid".to_owned(),
    }
}

fn quote_surface(surface: &str) -> String {
    format!("{surface:?}")
}

fn extract_transcript_text(path: &Path) -> anyhow::Result<String> {
    let text = fs::read_to_string(path)?;
    for line in text.lines().rev() {
        if let Some(rest) = line.strip_prefix("  text: ") {
            return Ok(rest.trim_matches('"').to_owned());
        }
    }
    anyhow::bail!("no final transcript text in {}", path.display())
}

fn g2p_model_dir() -> anyhow::Result<PathBuf> {
    if let Ok(path) = env::var("BEE_G2P_CHARSIU_MODEL_DIR") {
        return Ok(PathBuf::from(path));
    }
    let fallback = PathBuf::from("/tmp/charsiu-g2p");
    if fallback.join("model.safetensors").exists() {
        return Ok(fallback);
    }
    anyhow::bail!("missing Charsiu model dir")
}

fn zipa_bundle_dir() -> anyhow::Result<PathBuf> {
    if let Ok(path) = env::var("BEE_ZIPA_BUNDLE_DIR") {
        return Ok(PathBuf::from(path));
    }
    let fallback = PathBuf::from(env::var("HOME")?).join("bearcove/zipa-mlx-hf");
    if fallback.join("model.safetensors").exists() {
        return Ok(fallback);
    }
    anyhow::bail!("missing ZIPA bundle dir")
}
