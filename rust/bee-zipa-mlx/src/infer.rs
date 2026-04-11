use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::module::ModuleParameters;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis};

use crate::Result;
use crate::artifacts::ReferenceArtifacts;
use crate::audio::{AudioBuffer, load_wav_mono_f32};
use crate::config::{ZipaModelConfig, ZipaVariant};
use crate::encoder::{Stage0Encoder, Stage1EncoderPrefix, StageEncoder};
use crate::error::ZipaError;
use crate::features::{FbankExtractor, FbankParams};
use crate::load::{
    LoadStats, load_bypass_scale_from_map, load_downsample_weights_from_map,
    load_frontend_and_ctc_weights, load_stage_layer_weights_from_map,
};
use crate::model::ZipaModel;
use crate::tokenizer::TokenTable;

pub struct ZipaInference {
    pub config: ZipaModelConfig,
    pub tokens: TokenTable,
    pub features: FbankExtractor,
    pub model: ZipaModel,
    pub stage0: Stage0Encoder,
    pub stage1: Stage1EncoderPrefix,
    pub stages_2_to_5: Vec<StageEncoder>,
}

pub struct InferenceOutput {
    pub log_probs: Array,
    pub log_probs_len: usize,
    pub token_ids: Vec<usize>,
    pub tokens: Vec<String>,
}

pub struct GreedyInferenceOutput {
    pub frame_count: usize,
    pub token_ids: Vec<usize>,
    pub tokens: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PhoneSpan {
    pub token_id: usize,
    pub token: String,
    pub start_frame: usize,
    pub end_frame: usize,
    pub start_time_secs: f64,
    pub end_time_secs: f64,
}

const QUANTIZED_CHECKPOINT_FORMAT: &str = "zipa-mlx-quantized-v1";

impl ZipaInference {
    pub fn load_reference_small_no_diacritics() -> Result<Self> {
        let variant = ZipaVariant::SmallCrCtcNsNoDiacritics700k;
        let artifacts =
            ReferenceArtifacts::from_dir(ReferenceArtifacts::default_reference_dir(variant))?;
        let weights = artifacts.root.join("frontend_ctc.safetensors");
        Self::load_from_reference_dir(&artifacts, &weights)
    }

    pub fn load_quantized_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let (tensors, metadata) = Array::load_safetensors_with_metadata(path)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("load safetensors: {e}")))?;
        let format = metadata.get("format").ok_or_else(|| {
            ZipaError::Other(anyhow::anyhow!("missing quantized checkpoint format"))
        })?;
        if format != QUANTIZED_CHECKPOINT_FORMAT {
            return Err(ZipaError::Other(anyhow::anyhow!(
                "unsupported quantized checkpoint format: {format}"
            )));
        }
        let bits = metadata
            .get("bits")
            .ok_or_else(|| ZipaError::Other(anyhow::anyhow!("missing quantized checkpoint bits")))?
            .parse::<i32>()
            .map_err(|e| ZipaError::Other(anyhow::Error::new(e)))?;
        let group_size = metadata
            .get("group_size")
            .ok_or_else(|| {
                ZipaError::Other(anyhow::anyhow!("missing quantized checkpoint group_size"))
            })?
            .parse::<i32>()
            .map_err(|e| ZipaError::Other(anyhow::Error::new(e)))?;

        let variant = ZipaVariant::SmallCrCtcNsNoDiacritics700k;
        let artifacts =
            ReferenceArtifacts::from_dir(ReferenceArtifacts::default_reference_dir(variant))?;
        let config = ZipaModelConfig::for_variant(variant);
        let tokens = TokenTable::from_file(&artifacts.tokens_txt)?;
        let features = FbankExtractor::new(FbankParams::default());

        let model = ZipaModel::new(&config)?;
        let stage0 = Stage0Encoder::new(&config)?;
        let stage1 = Stage1EncoderPrefix::new(&config)?;
        let mut stages_2_to_5 = Vec::new();
        for stage in 2..=5 {
            stages_2_to_5.push(StageEncoder::new(&config, stage)?);
        }

        let mut inference = Self {
            config,
            tokens,
            features,
            model,
            stage0,
            stage1,
            stages_2_to_5,
        };
        inference.quantize_linears(group_size, bits)?;
        inference.load_flattened_quantized_tensors(&tensors)?;
        Ok(inference)
    }

    pub fn load_quantized_bundle_dir(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let checkpoint = path.join("model.safetensors");
        if !checkpoint.is_file() {
            return Err(ZipaError::Other(anyhow::anyhow!(
                "quantized bundle dir is missing model.safetensors: {}",
                checkpoint.display()
            )));
        }
        Self::load_quantized_safetensors(checkpoint)
    }

    pub fn load_from_reference_dir(
        artifacts: &ReferenceArtifacts,
        weights_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let tokens = TokenTable::from_file(&artifacts.tokens_txt)?;
        let features = FbankExtractor::new(FbankParams::default());

        let mut model = ZipaModel::new(&config)?;
        load_frontend_and_ctc_weights(&mut model, &weights_path)?;

        let tensors = Array::load_safetensors(weights_path)
            .map_err(|e| mlx_rs::error::Exception::custom(format!("load safetensors: {e}")))?;
        let stage0 = load_stage0(&config, &tensors)?;
        let stage1 = load_stage1(&config, &tensors)?;
        let mut stages_2_to_5 = Vec::new();
        for stage in 2..=5 {
            stages_2_to_5.push(load_stage(&config, stage, &tensors)?);
        }

        Ok(Self {
            config,
            tokens,
            features,
            model,
            stage0,
            stage1,
            stages_2_to_5,
        })
    }

    pub fn features_from_audio(&self, audio: &AudioBuffer) -> Result<Array> {
        let features = self.features.extract(audio)?;
        Ok(Array::from_slice(
            &features.data,
            &[1, features.num_frames as i32, features.num_filters as i32],
        ))
    }

    fn forward_encoder(&self, features: &Array) -> Result<Array> {
        let mut x = self.model.forward_frontend(features)?;
        x = x.transpose_axes(&[1, 0, 2])?;
        let mut outputs = Vec::with_capacity(6);
        x = self.stage0.forward(&x)?;
        outputs.push(x.clone());
        x = self.stage1.forward(&x)?;
        outputs.push(x.clone());
        for stage in &self.stages_2_to_5 {
            x = stage.forward(&x)?;
            outputs.push(x.clone());
        }
        get_full_dim_output(&outputs, &self.config.encoder_dim)
    }

    pub fn forward_features(&self, features: &Array) -> Result<(Array, usize)> {
        let full_dim = self.forward_encoder(features)?;
        let log_probs = self
            .model
            .ctc_head
            .forward(&full_dim)?
            .transpose_axes(&[1, 0, 2])?;
        let log_probs_len = log_probs.shape()[1] as usize;
        Ok((log_probs, log_probs_len))
    }

    pub fn infer_audio(&self, audio: &AudioBuffer) -> Result<InferenceOutput> {
        let features = self.features_from_audio(audio)?;
        self.infer_features(&features)
    }

    pub fn infer_wav(&self, wav_path: impl AsRef<Path>) -> Result<InferenceOutput> {
        let audio = load_wav_mono_f32(wav_path)?;
        self.infer_audio(&audio)
    }

    pub fn infer_audio_greedy(&self, audio: &AudioBuffer) -> Result<GreedyInferenceOutput> {
        let features = self.features_from_audio(audio)?;
        self.infer_features_greedy(&features)
    }

    pub fn infer_wav_greedy(&self, wav_path: impl AsRef<Path>) -> Result<GreedyInferenceOutput> {
        let audio = load_wav_mono_f32(wav_path)?;
        self.infer_audio_greedy(&audio)
    }

    pub fn infer_features(&self, features: &Array) -> Result<InferenceOutput> {
        let (log_probs, log_probs_len) = self.forward_features(features)?;
        let best_ids = argmax_axis(log_probs.index((0, .., ..)), -1, false)?;
        let token_ids = best_ids
            .as_slice::<u32>()
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let tokens = self.tokens.decode_ctc_greedy(&token_ids, 0);
        Ok(InferenceOutput {
            log_probs,
            log_probs_len,
            token_ids,
            tokens,
        })
    }

    pub fn infer_features_greedy(&self, features: &Array) -> Result<GreedyInferenceOutput> {
        let full_dim = self.forward_encoder(features)?;
        let frame_count = full_dim.shape()[0] as usize;
        let logits = self
            .model
            .ctc_head
            .forward_logits(&full_dim)?
            .transpose_axes(&[1, 0, 2])?;
        let best_ids = argmax_axis(logits.index((0, .., ..)), -1, false)?;
        let token_ids = best_ids
            .as_slice::<u32>()
            .iter()
            .map(|&id| id as usize)
            .collect::<Vec<_>>();
        let tokens = self.tokens.decode_ctc_greedy(&token_ids, 0);
        Ok(GreedyInferenceOutput {
            frame_count,
            token_ids,
            tokens,
        })
    }

    pub fn quantize_linears(&mut self, group_size: i32, bits: i32) -> Result<()> {
        self.model.quantize_linears(group_size, bits)?;
        self.stage0.quantize_linears(group_size, bits)?;
        self.stage1.quantize_linears(group_size, bits)?;
        for stage in &mut self.stages_2_to_5 {
            stage.quantize_linears(group_size, bits)?;
        }
        Ok(())
    }

    pub fn save_quantized_safetensors(
        &self,
        path: impl AsRef<Path>,
        group_size: i32,
        bits: i32,
    ) -> Result<()> {
        let mut tensors = HashMap::new();
        collect_prefixed_parameters("model", &self.model, &mut tensors);
        collect_prefixed_parameters("stage0", &self.stage0, &mut tensors);
        collect_prefixed_parameters("stage1", &self.stage1, &mut tensors);
        for (index, stage) in self.stages_2_to_5.iter().enumerate() {
            collect_prefixed_parameters(&format!("stage{}", index + 2), stage, &mut tensors);
        }

        let mut metadata = HashMap::new();
        metadata.insert("format".to_owned(), QUANTIZED_CHECKPOINT_FORMAT.to_owned());
        metadata.insert(
            "variant".to_owned(),
            "small-crctc-ns-no-diacritics-700k".to_owned(),
        );
        metadata.insert("group_size".to_owned(), group_size.to_string());
        metadata.insert("bits".to_owned(), bits.to_string());

        Array::save_safetensors(&tensors, Some(&metadata), path)
            .map_err(|e| ZipaError::Other(anyhow::Error::new(e)))?;
        Ok(())
    }

    fn load_flattened_quantized_tensors(&mut self, tensors: &HashMap<String, Array>) -> Result<()> {
        load_prefixed_parameters("model", &mut self.model, tensors)?;
        load_prefixed_parameters("stage0", &mut self.stage0, tensors)?;
        load_prefixed_parameters("stage1", &mut self.stage1, tensors)?;
        for (index, stage) in self.stages_2_to_5.iter_mut().enumerate() {
            load_prefixed_parameters(&format!("stage{}", index + 2), stage, tensors)?;
        }
        Ok(())
    }
}

impl InferenceOutput {
    pub fn derive_phone_spans(
        &self,
        token_table: &TokenTable,
        total_duration_secs: f64,
        blank_id: usize,
    ) -> Vec<PhoneSpan> {
        derive_phone_spans_from_ids(
            self.log_probs_len,
            &self.token_ids,
            token_table,
            total_duration_secs,
            blank_id,
        )
    }
}

impl GreedyInferenceOutput {
    pub fn derive_phone_spans(
        &self,
        token_table: &TokenTable,
        total_duration_secs: f64,
        blank_id: usize,
    ) -> Vec<PhoneSpan> {
        derive_phone_spans_from_ids(
            self.frame_count,
            &self.token_ids,
            token_table,
            total_duration_secs,
            blank_id,
        )
    }
}

fn derive_phone_spans_from_ids(
    frame_count: usize,
    token_ids: &[usize],
    token_table: &TokenTable,
    total_duration_secs: f64,
    blank_id: usize,
) -> Vec<PhoneSpan> {
    if frame_count == 0 || token_ids.is_empty() || total_duration_secs <= 0.0 {
        return Vec::new();
    }

    let seconds_per_frame = total_duration_secs / frame_count as f64;
    let mut spans = Vec::new();
    let mut frame = 0usize;
    let mut prev_nonblank = None;

    while frame < token_ids.len() {
        let token_id = token_ids[frame];
        if token_id == blank_id {
            prev_nonblank = None;
            frame += 1;
            continue;
        }

        if prev_nonblank == Some(token_id) {
            frame += 1;
            continue;
        }

        let start_frame = frame;
        frame += 1;
        while frame < token_ids.len() && token_ids[frame] == token_id {
            frame += 1;
        }
        let end_frame = frame;
        prev_nonblank = Some(token_id);

        let Some(token) = token_table.get(token_id) else {
            continue;
        };
        if token.is_empty() {
            continue;
        }

        spans.push(PhoneSpan {
            token_id,
            token: token.to_owned(),
            start_frame,
            end_frame,
            start_time_secs: start_frame as f64 * seconds_per_frame,
            end_time_secs: end_frame as f64 * seconds_per_frame,
        });
    }

    spans
}

fn collect_prefixed_parameters<M: ModuleParameters>(
    prefix: &str,
    module: &M,
    out: &mut HashMap<String, Array>,
) {
    for (name, value) in module.parameters().flatten() {
        out.insert(format!("{prefix}.{name}"), value.clone());
    }
}

fn load_prefixed_parameters<M: ModuleParameters>(
    prefix: &str,
    module: &mut M,
    tensors: &HashMap<String, Array>,
) -> Result<()> {
    let mut params = module.parameters_mut().flatten();
    for (name, param) in &mut params {
        let key = format!("{prefix}.{name}");
        let value = tensors
            .get(&key)
            .ok_or_else(|| ZipaError::Other(anyhow::anyhow!("missing quantized tensor: {key}")))?;
        **param = value.clone();
    }
    Ok(())
}

fn load_stage0(
    config: &ZipaModelConfig,
    tensors: &HashMap<String, Array>,
) -> Result<Stage0Encoder> {
    let mut stage0 = Stage0Encoder::new(config)?;
    let stats =
        load_stage_layer_weights_from_map(&mut stage0.layer0, "encoder.stage0.layer0", tensors)?;
    require_no_missing(stats, "stage0 layer0")?;
    let stats =
        load_stage_layer_weights_from_map(&mut stage0.layer1, "encoder.stage0.layer1", tensors)?;
    require_no_missing(stats, "stage0 layer1")?;
    Ok(stage0)
}

fn load_stage1(
    config: &ZipaModelConfig,
    tensors: &HashMap<String, Array>,
) -> Result<Stage1EncoderPrefix> {
    let mut stage1 = Stage1EncoderPrefix::new(config)?;
    let stats = load_downsample_weights_from_map(
        &mut stage1.downsample,
        "encoder.stage1.downsample.weights",
        tensors,
    )?;
    require_no_missing(stats, "stage1 downsample")?;
    let stats =
        load_stage_layer_weights_from_map(&mut stage1.layer0, "encoder.stage1.layer0", tensors)?;
    require_no_missing(stats, "stage1 layer0")?;
    let stats =
        load_stage_layer_weights_from_map(&mut stage1.layer1, "encoder.stage1.layer1", tensors)?;
    require_no_missing(stats, "stage1 layer1")?;
    let stats = load_bypass_scale_from_map(
        &mut stage1.out_combiner,
        "encoder.stage1.out_combiner.bypass_scale",
        tensors,
    )?;
    require_no_missing(stats, "stage1 out_combiner")?;
    Ok(stage1)
}

fn load_stage(
    config: &ZipaModelConfig,
    stage_index: usize,
    tensors: &HashMap<String, Array>,
) -> Result<StageEncoder> {
    let mut stage = StageEncoder::new(config, stage_index)?;
    let stats = load_downsample_weights_from_map(
        &mut stage.downsample,
        &format!("encoder.stage{stage_index}.downsample.weights"),
        tensors,
    )?;
    require_no_missing(stats, &format!("stage{stage_index} downsample"))?;
    for (layer_index, layer) in stage.layers.iter_mut().enumerate() {
        let prefix = format!("encoder.stage{stage_index}.layer{layer_index}");
        let stats = load_stage_layer_weights_from_map(layer, &prefix, tensors)?;
        require_no_missing(stats, &prefix)?;
    }
    let stats = load_bypass_scale_from_map(
        &mut stage.out_combiner,
        &format!("encoder.stage{stage_index}.out_combiner.bypass_scale"),
        tensors,
    )?;
    require_no_missing(stats, &format!("stage{stage_index} out_combiner"))?;
    Ok(stage)
}

fn require_no_missing(stats: LoadStats, label: &str) -> Result<()> {
    if stats.missing.is_empty() {
        return Ok(());
    }
    Err(ZipaError::Other(anyhow::anyhow!(
        "missing {label} weights: {:?}",
        stats.missing
    )))
}

fn get_full_dim_output(outputs: &[Array], encoder_dims: &[usize]) -> Result<Array> {
    if outputs.len() != encoder_dims.len() {
        return Err(ZipaError::Other(anyhow::anyhow!(
            "expected {} encoder outputs, got {}",
            encoder_dims.len(),
            outputs.len()
        )));
    }
    let output_dim = encoder_dims.iter().copied().max().unwrap_or(0) as i32;
    let mut output_pieces = vec![outputs[outputs.len() - 1].clone()];
    let mut cur_dim = encoder_dims[encoder_dims.len() - 1] as i32;

    for i in (0..outputs.len() - 1).rev() {
        let dim = encoder_dims[i] as i32;
        if dim > cur_dim {
            output_pieces.push(outputs[i].index((.., .., cur_dim..dim)));
            cur_dim = dim;
        }
    }

    if cur_dim != output_dim {
        return Err(ZipaError::Other(anyhow::anyhow!(
            "failed to reconstruct full encoder width: got {cur_dim}, expected {output_dim}"
        )));
    }
    Ok(mlx_rs::ops::concatenate_axis(&output_pieces, 2)?)
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use std::path::PathBuf;

    use mlx_rs::Array;
    use mlx_rs::ops::indexing::IndexOp;

    use super::{InferenceOutput, ZipaInference};
    use crate::tokenizer::TokenTable;

    #[test]
    fn derives_phone_spans_from_ctc_argmax_frames() {
        let table = TokenTable::from_str("<blk> 0\nə 1\nn 2\n").unwrap();
        let output = InferenceOutput {
            log_probs: Array::from_slice(&[0.0f32; 7 * 3], &[1, 7, 3]),
            log_probs_len: 7,
            token_ids: vec![0, 1, 1, 0, 2, 2, 0],
            tokens: vec!["ə".to_string(), "n".to_string()],
        };

        let spans = output.derive_phone_spans(&table, 0.7, 0);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].token, "ə");
        assert_eq!(spans[0].start_frame, 1);
        assert_eq!(spans[0].end_frame, 3);
        assert!((spans[0].start_time_secs - 0.1).abs() < 1e-6);
        assert!((spans[0].end_time_secs - 0.3).abs() < 1e-6);

        assert_eq!(spans[1].token, "n");
        assert_eq!(spans[1].start_frame, 4);
        assert_eq!(spans[1].end_frame, 6);
        assert!((spans[1].start_time_secs - 0.4).abs() < 1e-6);
        assert!((spans[1].end_time_secs - 0.6).abs() < 1e-6);
    }

    #[test]
    fn end_to_end_log_probs_match_onnx_reference_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let reference = home.join(
            "bearcove/zipa/checkpoints/zipa-cr-ns-small-nodiacritics-700k/exp/authored_282_take_1_frontend_ref.safetensors",
        );
        if !reference.exists() {
            return;
        }

        let inference = ZipaInference::load_reference_small_no_diacritics().unwrap();
        let tensors = Array::load_safetensors(&reference).unwrap();
        let features = tensors.get("features").unwrap();
        let expected = tensors.get("log_probs").unwrap();
        let expected_len = tensors.get("log_probs_len").unwrap().index(0).item::<i64>() as usize;

        let actual = inference.infer_features(features).unwrap();
        assert_eq!(actual.log_probs.shape(), expected.shape());
        assert_eq!(actual.log_probs_len, expected_len);
        assert!(
            actual
                .log_probs
                .all_close(expected, 1e-4, 1e-4, None)
                .unwrap()
                .item::<bool>(),
            "end-to-end log_probs diverged from ONNX reference"
        );
    }

    #[test]
    fn end_to_end_decode_produces_non_empty_tokens_for_reference_sample_when_local_artifacts_exist()
    {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let wav = home.join("bearcove/bee/data/phonetic-seed/audio-wav/authored_282_take_1.wav");
        if !wav.exists() {
            return;
        }

        let inference = ZipaInference::load_reference_small_no_diacritics().unwrap();
        let actual = inference.infer_wav(&wav).unwrap();
        assert!(!actual.tokens.is_empty());
        assert_eq!(actual.tokens[0], "▁");
    }

    #[test]
    fn quantized_checkpoint_roundtrips_for_reference_sample_when_local_artifacts_exist() {
        let home = match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home),
            None => return,
        };
        let wav = home.join("bearcove/bee/data/phonetic-seed/audio-wav/authored_282_take_1.wav");
        if !wav.exists() {
            return;
        }

        let mut inference = ZipaInference::load_reference_small_no_diacritics().unwrap();
        inference.quantize_linears(64, 8).unwrap();

        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("zipa-q8-roundtrip-{unique}.safetensors"));
        inference.save_quantized_safetensors(&path, 64, 8).unwrap();

        let reloaded = ZipaInference::load_quantized_safetensors(&path).unwrap();
        let original = inference.infer_wav(&wav).unwrap();
        let loaded = reloaded.infer_wav(&wav).unwrap();

        let _ = std::fs::remove_file(&path);

        assert_eq!(original.tokens, loaded.tokens);
        assert_eq!(original.log_probs_len, loaded.log_probs_len);
        assert!(
            original
                .log_probs
                .all_close(&loaded.log_probs, 1e-5, 1e-5, None)
                .unwrap()
                .item::<bool>(),
            "reloaded quantized checkpoint diverged from saved inference"
        );
    }
}
