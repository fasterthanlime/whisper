use std::collections::HashMap;
use std::path::Path;

use mlx_rs::Array;
use mlx_rs::error::Exception;
use mlx_rs::module::ModuleParameters;

use crate::encoder::{BypassModule, Downsample2, ZipformerEncoderLayer};
use crate::model::ZipaModel;

const DIRECT_KEYS: &[(&str, &str)] = &[
    ("encoder_embed.conv0.bias", "encoder_embed.conv0.bias"),
    ("encoder_embed.conv1.bias", "encoder_embed.conv1.bias"),
    ("encoder_embed.conv2.bias", "encoder_embed.conv2.bias"),
    (
        "encoder_embed.convnext.depthwise_conv.bias",
        "encoder_embed.convnext.depthwise_conv.bias",
    ),
    (
        "encoder_embed.convnext.pointwise_conv1.bias",
        "encoder_embed.convnext.pointwise_conv1.bias",
    ),
    (
        "encoder_embed.convnext.pointwise_conv2.bias",
        "encoder_embed.convnext.pointwise_conv2.bias",
    ),
    ("encoder_embed.out.bias", "encoder_embed.out.bias"),
    (
        "encoder_embed.out_norm.log_scale",
        "encoder_embed.out_norm.log_scale",
    ),
    ("encoder_embed.out_norm.bias", "encoder_embed.out_norm.bias"),
    ("ctc_output.linear.bias", "ctc_head.linear.bias"),
];

const CONV2D_KEYS: &[(&str, &str)] = &[
    ("encoder_embed.conv0.weight", "encoder_embed.conv0.weight"),
    ("encoder_embed.conv1.weight", "encoder_embed.conv1.weight"),
    ("encoder_embed.conv2.weight", "encoder_embed.conv2.weight"),
    (
        "encoder_embed.convnext.depthwise_conv.weight",
        "encoder_embed.convnext.depthwise_conv.weight",
    ),
    (
        "encoder_embed.convnext.pointwise_conv1.weight",
        "encoder_embed.convnext.pointwise_conv1.weight",
    ),
    (
        "encoder_embed.convnext.pointwise_conv2.weight",
        "encoder_embed.convnext.pointwise_conv2.weight",
    ),
];

const LINEAR_KEYS: &[(&str, &str)] = &[
    ("encoder_embed.out.weight", "encoder_embed.out.weight"),
    ("ctc_output.linear.weight", "ctc_head.linear.weight"),
];

/// Load the ZIPA frontend + CTC head weights from a safetensors file produced by
/// `scripts/zipa/export_onnx_initializers.py`.
pub fn load_frontend_and_ctc_weights(
    model: &mut ZipaModel,
    safetensors_path: impl AsRef<Path>,
) -> Result<LoadStats, Exception> {
    let loaded = Array::load_safetensors(safetensors_path)
        .map_err(|e| Exception::custom(format!("load safetensors: {e}")))?;
    load_frontend_and_ctc_weights_from_map(model, &loaded)
}

pub fn load_frontend_and_ctc_weights_from_map(
    model: &mut ZipaModel,
    tensors: &HashMap<String, Array>,
) -> Result<LoadStats, Exception> {
    let mut params = model.parameters_mut().flatten();
    let mut loaded_count = 0usize;
    let mut missing = Vec::new();

    for (src, dst) in DIRECT_KEYS {
        match tensors.get(*src) {
            Some(value) => {
                if let Some(param) = params.get_mut(*dst) {
                    **param = value.clone();
                    loaded_count += 1;
                }
            }
            None => missing.push((*src).to_owned()),
        }
    }

    for (src, dst) in CONV2D_KEYS {
        match tensors.get(*src) {
            Some(value) => {
                let transposed = value.transpose_axes(&[0, 2, 3, 1])?;
                if let Some(param) = params.get_mut(*dst) {
                    **param = transposed;
                    loaded_count += 1;
                }
            }
            None => missing.push((*src).to_owned()),
        }
    }

    for (src, dst) in LINEAR_KEYS {
        match tensors.get(*src) {
            Some(value) => {
                let transposed = value.transpose_axes(&[1, 0])?;
                if let Some(param) = params.get_mut(*dst) {
                    **param = transposed;
                    loaded_count += 1;
                }
            }
            None => missing.push((*src).to_owned()),
        }
    }

    Ok(LoadStats {
        loaded: loaded_count,
        missing,
    })
}

const STAGE_LAYER_DIRECT_SUFFIXES: &[&str] = &[
    "self_attn_weights.in_proj.bias",
    "feed_forward1.in_proj.bias",
    "feed_forward1.out_proj.bias",
    "nonlin_attention.in_proj.bias",
    "nonlin_attention.out_proj.bias",
    "self_attn1.in_proj.bias",
    "self_attn1.out_proj.bias",
    "conv_module1.in_proj.bias",
    "conv_module1.depthwise_conv.bias",
    "conv_module1.out_proj.bias",
    "feed_forward2.in_proj.bias",
    "feed_forward2.out_proj.bias",
    "bypass_mid.bypass_scale",
    "self_attn2.in_proj.bias",
    "self_attn2.out_proj.bias",
    "conv_module2.in_proj.bias",
    "conv_module2.depthwise_conv.bias",
    "conv_module2.out_proj.bias",
    "feed_forward3.in_proj.bias",
    "feed_forward3.out_proj.bias",
    "norm.log_scale",
    "norm.bias",
    "bypass.bypass_scale",
];

const STAGE_LAYER_LINEAR_SUFFIXES: &[&str] = &[
    "self_attn_weights.in_proj.weight",
    "self_attn_weights.linear_pos.weight",
    "feed_forward1.in_proj.weight",
    "feed_forward1.out_proj.weight",
    "nonlin_attention.in_proj.weight",
    "nonlin_attention.out_proj.weight",
    "self_attn1.in_proj.weight",
    "self_attn1.out_proj.weight",
    "conv_module1.in_proj.weight",
    "conv_module1.out_proj.weight",
    "feed_forward2.in_proj.weight",
    "feed_forward2.out_proj.weight",
    "self_attn2.in_proj.weight",
    "self_attn2.out_proj.weight",
    "conv_module2.in_proj.weight",
    "conv_module2.out_proj.weight",
    "feed_forward3.in_proj.weight",
    "feed_forward3.out_proj.weight",
];

const STAGE_LAYER_CONV1D_SUFFIXES: &[&str] = &[
    "conv_module1.depthwise_conv.weight",
    "conv_module2.depthwise_conv.weight",
];

fn stage_layer_key(prefix: &str, suffix: &str) -> String {
    format!("{prefix}.{suffix}")
}

pub fn load_stage0_layer_weights_from_map(
    layer: &mut ZipformerEncoderLayer,
    tensors: &HashMap<String, Array>,
) -> Result<LoadStats, Exception> {
    load_stage_layer_weights_from_map(layer, "encoder.stage0.layer0", tensors)
}

pub fn load_stage_layer_weights_from_map(
    layer: &mut ZipformerEncoderLayer,
    source_prefix: &str,
    tensors: &HashMap<String, Array>,
) -> Result<LoadStats, Exception> {
    let mut params = layer.parameters_mut().flatten();
    let mut loaded_count = 0usize;
    let mut missing = Vec::new();

    for suffix in STAGE_LAYER_DIRECT_SUFFIXES {
        let src = stage_layer_key(source_prefix, suffix);
        match tensors.get(&src) {
            Some(value) => {
                if let Some(param) = params.get_mut(*suffix) {
                    **param = value.clone();
                    loaded_count += 1;
                }
            }
            None => missing.push(src),
        }
    }

    for suffix in STAGE_LAYER_LINEAR_SUFFIXES {
        let src = stage_layer_key(source_prefix, suffix);
        match tensors.get(&src) {
            Some(value) => {
                let transposed = value.transpose_axes(&[1, 0])?;
                if let Some(param) = params.get_mut(*suffix) {
                    **param = transposed;
                    loaded_count += 1;
                }
            }
            None => missing.push(src),
        }
    }

    for suffix in STAGE_LAYER_CONV1D_SUFFIXES {
        let src = stage_layer_key(source_prefix, suffix);
        match tensors.get(&src) {
            Some(value) => {
                let transposed = value.transpose_axes(&[0, 2, 1])?;
                if let Some(param) = params.get_mut(*suffix) {
                    **param = transposed;
                    loaded_count += 1;
                }
            }
            None => missing.push(src),
        }
    }

    Ok(LoadStats {
        loaded: loaded_count,
        missing,
    })
}

pub fn load_downsample_weights_from_map(
    downsample: &mut Downsample2,
    source_key: &str,
    tensors: &HashMap<String, Array>,
) -> Result<LoadStats, Exception> {
    let mut params = downsample.parameters_mut().flatten();
    let mut missing = Vec::new();
    let mut loaded = 0usize;

    match tensors.get(source_key) {
        Some(value) => {
            if let Some(param) = params.get_mut("weights") {
                **param = value.clone();
                loaded += 1;
            }
        }
        None => missing.push(source_key.to_owned()),
    }

    Ok(LoadStats { loaded, missing })
}

pub fn load_bypass_scale_from_map(
    bypass: &mut BypassModule,
    source_key: &str,
    tensors: &HashMap<String, Array>,
) -> Result<LoadStats, Exception> {
    let mut params = bypass.parameters_mut().flatten();
    let mut missing = Vec::new();
    let mut loaded = 0usize;

    match tensors.get(source_key) {
        Some(value) => {
            if let Some(param) = params.get_mut("bypass_scale") {
                **param = value.clone();
                loaded += 1;
            }
        }
        None => missing.push(source_key.to_owned()),
    }

    Ok(LoadStats { loaded, missing })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadStats {
    pub loaded: usize,
    pub missing: Vec<String>,
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use mlx_rs::Array;
    use mlx_rs::module::ModuleParameters;

    use crate::config::{ZipaModelConfig, ZipaVariant};
    use crate::load::load_frontend_and_ctc_weights_from_map;
    use crate::model::ZipaModel;

    #[test]
    fn loads_frontend_subset_and_transposes_conv_weights() {
        let config = ZipaModelConfig::for_variant(ZipaVariant::SmallCrCtcNsNoDiacritics700k);
        let mut model = ZipaModel::new(&config).unwrap();

        let mut tensors = HashMap::new();
        tensors.insert(
            "encoder_embed.conv0.weight".to_string(),
            Array::zeros::<f32>(&[8, 1, 3, 3]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.conv0.bias".to_string(),
            Array::zeros::<f32>(&[8]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.conv1.weight".to_string(),
            Array::zeros::<f32>(&[32, 8, 3, 3]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.conv1.bias".to_string(),
            Array::zeros::<f32>(&[32]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.conv2.weight".to_string(),
            Array::zeros::<f32>(&[128, 32, 3, 3]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.conv2.bias".to_string(),
            Array::zeros::<f32>(&[128]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.convnext.depthwise_conv.weight".to_string(),
            Array::zeros::<f32>(&[128, 1, 7, 7]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.convnext.depthwise_conv.bias".to_string(),
            Array::zeros::<f32>(&[128]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.convnext.pointwise_conv1.weight".to_string(),
            Array::zeros::<f32>(&[384, 128, 1, 1]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.convnext.pointwise_conv1.bias".to_string(),
            Array::zeros::<f32>(&[384]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.convnext.pointwise_conv2.weight".to_string(),
            Array::zeros::<f32>(&[128, 384, 1, 1]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.convnext.pointwise_conv2.bias".to_string(),
            Array::zeros::<f32>(&[128]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.out.weight".to_string(),
            Array::zeros::<f32>(&[2432, 192]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.out.bias".to_string(),
            Array::zeros::<f32>(&[192]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.out_norm.log_scale".to_string(),
            Array::zeros::<f32>(&[]).unwrap(),
        );
        tensors.insert(
            "encoder_embed.out_norm.bias".to_string(),
            Array::zeros::<f32>(&[192]).unwrap(),
        );
        tensors.insert(
            "ctc_output.linear.weight".to_string(),
            Array::zeros::<f32>(&[512, 127]).unwrap(),
        );
        tensors.insert(
            "ctc_output.linear.bias".to_string(),
            Array::zeros::<f32>(&[127]).unwrap(),
        );

        let stats = load_frontend_and_ctc_weights_from_map(&mut model, &tensors).unwrap();
        assert_eq!(stats.loaded, 18);

        let flat = model.parameters().flatten();
        assert_eq!(flat["encoder_embed.conv0.weight"].shape(), vec![8, 3, 3, 1]);
        assert_eq!(
            flat["encoder_embed.convnext.depthwise_conv.weight"].shape(),
            vec![128, 7, 7, 1]
        );
        assert_eq!(flat["encoder_embed.out.weight"].shape(), vec![192, 2432]);
        assert_eq!(flat["ctc_head.linear.weight"].shape(), vec![127, 512]);
    }
}
