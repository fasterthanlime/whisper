#!/usr/bin/env python3

import argparse
from pathlib import Path

import onnx
from onnx import numpy_helper
from safetensors.numpy import save_file


NAME_MAP = {
    "encoder_embed.conv.0.weight": "encoder_embed.conv0.weight",
    "encoder_embed.conv.0.bias": "encoder_embed.conv0.bias",
    "encoder_embed.conv.4.weight": "encoder_embed.conv1.weight",
    "encoder_embed.conv.4.bias": "encoder_embed.conv1.bias",
    "encoder_embed.conv.7.weight": "encoder_embed.conv2.weight",
    "encoder_embed.conv.7.bias": "encoder_embed.conv2.bias",
    "encoder_embed.convnext.depthwise_conv.weight": "encoder_embed.convnext.depthwise_conv.weight",
    "encoder_embed.convnext.depthwise_conv.bias": "encoder_embed.convnext.depthwise_conv.bias",
    "encoder_embed.convnext.pointwise_conv1.weight": "encoder_embed.convnext.pointwise_conv1.weight",
    "encoder_embed.convnext.pointwise_conv1.bias": "encoder_embed.convnext.pointwise_conv1.bias",
    "encoder_embed.convnext.pointwise_conv2.weight": "encoder_embed.convnext.pointwise_conv2.weight",
    "encoder_embed.convnext.pointwise_conv2.bias": "encoder_embed.convnext.pointwise_conv2.bias",
    "onnx::MatMul_11174": "encoder_embed.out.weight",
    "encoder_embed.out.bias": "encoder_embed.out.bias",
    "encoder_embed.out_norm.log_scale": "encoder_embed.out_norm.log_scale",
    "encoder_embed.out_norm.bias": "encoder_embed.out_norm.bias",
    "onnx::MatMul_12110": "ctc_output.linear.weight",
    "ctc_output.1.bias": "ctc_output.linear.bias",
}

STAGE0_LAYER_MATMULS = {
    0: {
        "self_attn_weights.in_proj.weight": "onnx::MatMul_11184",
        "self_attn_weights.linear_pos.weight": "onnx::MatMul_11203",
        "feed_forward1.in_proj.weight": "onnx::MatMul_11210",
        "feed_forward1.out_proj.weight": "onnx::MatMul_11211",
        "nonlin_attention.in_proj.weight": "onnx::MatMul_11216",
        "nonlin_attention.out_proj.weight": "onnx::MatMul_11220",
        "self_attn1.in_proj.weight": "onnx::MatMul_11221",
        "self_attn1.out_proj.weight": "onnx::MatMul_11223",
        "conv_module1.in_proj.weight": "onnx::MatMul_11224",
        "conv_module1.out_proj.weight": "onnx::MatMul_11225",
        "feed_forward2.in_proj.weight": "onnx::MatMul_11226",
        "feed_forward2.out_proj.weight": "onnx::MatMul_11227",
        "self_attn2.in_proj.weight": "onnx::MatMul_11228",
        "self_attn2.out_proj.weight": "onnx::MatMul_11230",
        "conv_module2.in_proj.weight": "onnx::MatMul_11231",
        "conv_module2.out_proj.weight": "onnx::MatMul_11232",
        "feed_forward3.in_proj.weight": "onnx::MatMul_11233",
        "feed_forward3.out_proj.weight": "onnx::MatMul_11234",
    },
    1: {
        "self_attn_weights.in_proj.weight": "onnx::MatMul_11235",
        "self_attn_weights.linear_pos.weight": "onnx::MatMul_11254",
        "feed_forward1.in_proj.weight": "onnx::MatMul_11261",
        "feed_forward1.out_proj.weight": "onnx::MatMul_11262",
        "nonlin_attention.in_proj.weight": "onnx::MatMul_11267",
        "nonlin_attention.out_proj.weight": "onnx::MatMul_11271",
        "self_attn1.in_proj.weight": "onnx::MatMul_11272",
        "self_attn1.out_proj.weight": "onnx::MatMul_11274",
        "conv_module1.in_proj.weight": "onnx::MatMul_11275",
        "conv_module1.out_proj.weight": "onnx::MatMul_11276",
        "feed_forward2.in_proj.weight": "onnx::MatMul_11277",
        "feed_forward2.out_proj.weight": "onnx::MatMul_11278",
        "self_attn2.in_proj.weight": "onnx::MatMul_11279",
        "self_attn2.out_proj.weight": "onnx::MatMul_11281",
        "conv_module2.in_proj.weight": "onnx::MatMul_11282",
        "conv_module2.out_proj.weight": "onnx::MatMul_11283",
        "feed_forward3.in_proj.weight": "onnx::MatMul_11284",
        "feed_forward3.out_proj.weight": "onnx::MatMul_11285",
    },
}

STAGE1_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11301",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11320",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11327",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11328",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11333",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11337",
    "self_attn1.in_proj.weight": "onnx::MatMul_11338",
    "self_attn1.out_proj.weight": "onnx::MatMul_11340",
    "conv_module1.in_proj.weight": "onnx::MatMul_11341",
    "conv_module1.out_proj.weight": "onnx::MatMul_11342",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11343",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11344",
    "self_attn2.in_proj.weight": "onnx::MatMul_11345",
    "self_attn2.out_proj.weight": "onnx::MatMul_11347",
    "conv_module2.in_proj.weight": "onnx::MatMul_11348",
    "conv_module2.out_proj.weight": "onnx::MatMul_11349",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11350",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11351",
}


def add_stage0_layer(layer_index: int) -> None:
    prefix = f"encoder.stage0.layer{layer_index}"
    source_prefix = f"encoder.encoders.0.layers.{layer_index}"
    matmuls = STAGE0_LAYER_MATMULS[layer_index]

    for dst_suffix, src_name in matmuls.items():
        NAME_MAP[src_name] = f"{prefix}.{dst_suffix}"

    direct_suffixes = [
        "self_attn_weights.in_proj.bias",
        "feed_forward1.in_proj.bias",
        "feed_forward1.out_proj.bias",
        "nonlin_attention.in_proj.bias",
        "nonlin_attention.out_proj.bias",
        "self_attn1.in_proj.bias",
        "self_attn1.out_proj.bias",
        "conv_module1.in_proj.bias",
        "conv_module1.depthwise_conv.weight",
        "conv_module1.depthwise_conv.bias",
        "conv_module1.out_proj.bias",
        "feed_forward2.in_proj.bias",
        "feed_forward2.out_proj.bias",
        "bypass_mid.bypass_scale",
        "self_attn2.in_proj.bias",
        "self_attn2.out_proj.bias",
        "conv_module2.in_proj.bias",
        "conv_module2.depthwise_conv.weight",
        "conv_module2.depthwise_conv.bias",
        "conv_module2.out_proj.bias",
        "feed_forward3.in_proj.bias",
        "feed_forward3.out_proj.bias",
        "norm.log_scale",
        "norm.bias",
        "bypass.bypass_scale",
    ]
    for suffix in direct_suffixes:
        NAME_MAP[f"{source_prefix}.{suffix}"] = f"{prefix}.{suffix}"


for stage0_layer_index in range(2):
    add_stage0_layer(stage0_layer_index)

NAME_MAP["onnx::Mul_11296"] = "encoder.stage1.downsample.weights"

for dst_suffix, src_name in STAGE1_LAYER0_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage1.layer0.{dst_suffix}"

for suffix in [
    "self_attn_weights.in_proj.bias",
    "feed_forward1.in_proj.bias",
    "feed_forward1.out_proj.bias",
    "nonlin_attention.in_proj.bias",
    "nonlin_attention.out_proj.bias",
    "self_attn1.in_proj.bias",
    "self_attn1.out_proj.bias",
    "conv_module1.in_proj.bias",
    "conv_module1.depthwise_conv.weight",
    "conv_module1.depthwise_conv.bias",
    "conv_module1.out_proj.bias",
    "feed_forward2.in_proj.bias",
    "feed_forward2.out_proj.bias",
    "bypass_mid.bypass_scale",
    "self_attn2.in_proj.bias",
    "self_attn2.out_proj.bias",
    "conv_module2.in_proj.bias",
    "conv_module2.depthwise_conv.weight",
    "conv_module2.depthwise_conv.bias",
    "conv_module2.out_proj.bias",
    "feed_forward3.in_proj.bias",
    "feed_forward3.out_proj.bias",
    "norm.log_scale",
    "norm.bias",
    "bypass.bypass_scale",
]:
    NAME_MAP[f"encoder.encoders.1.encoder.layers.0.{suffix}"] = (
        f"encoder.stage1.layer0.{suffix}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract ZIPA ONNX initializers into a safetensors file for MLX loading."
    )
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    model = onnx.load(args.onnx_model)
    tensors = {}
    missing = []

    for initializer in model.graph.initializer:
        mapped = NAME_MAP.get(initializer.name)
        if mapped is None:
            continue
        tensors[mapped] = numpy_helper.to_array(initializer)

    for source_name in NAME_MAP:
        if NAME_MAP[source_name] not in tensors:
            missing.append(source_name)

    if missing:
        raise SystemExit(f"missing required initializers: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output))
    print(f"wrote {len(tensors)} tensors to {args.output}")


if __name__ == "__main__":
    main()
