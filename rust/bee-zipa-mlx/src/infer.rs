use std::collections::HashMap;
use std::path::Path;

use mlx_rs::ops::indexing::{argmax_axis, IndexOp};
use mlx_rs::Array;

use crate::artifacts::ReferenceArtifacts;
use crate::audio::{load_wav_mono_f32, AudioBuffer};
use crate::config::{ZipaModelConfig, ZipaVariant};
use crate::encoder::{Stage0Encoder, Stage1EncoderPrefix, StageEncoder};
use crate::error::ZipaError;
use crate::features::{FbankExtractor, FbankParams};
use crate::load::{
    load_bypass_scale_from_map, load_downsample_weights_from_map, load_frontend_and_ctc_weights,
    load_stage_layer_weights_from_map, LoadStats,
};
use crate::model::ZipaModel;
use crate::tokenizer::TokenTable;
use crate::Result;

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

impl ZipaInference {
    pub fn load_reference_small_no_diacritics() -> Result<Self> {
        let variant = ZipaVariant::SmallCrCtcNsNoDiacritics700k;
        let artifacts =
            ReferenceArtifacts::from_dir(ReferenceArtifacts::default_reference_dir(variant))?;
        let weights = artifacts.root.join("frontend_ctc.safetensors");
        Self::load_from_reference_dir(&artifacts, &weights)
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

    pub fn forward_features(&self, features: &Array) -> Result<(Array, usize)> {
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
        let full_dim = get_full_dim_output(&outputs, &self.config.encoder_dim)?;
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

    pub fn quantize_linears(&mut self, group_size: i32, bits: i32) -> Result<()> {
        self.model.quantize_linears(group_size, bits)?;
        self.stage0.quantize_linears(group_size, bits)?;
        self.stage1.quantize_linears(group_size, bits)?;
        for stage in &mut self.stages_2_to_5 {
            stage.quantize_linears(group_size, bits)?;
        }
        Ok(())
    }
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
    use std::path::PathBuf;

    use mlx_rs::ops::indexing::IndexOp;
    use mlx_rs::Array;

    use super::ZipaInference;

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
}
