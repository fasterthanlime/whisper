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
    "onnx::MatMul_11184": "encoder.stage0.layer0.self_attn_weights.in_proj.weight",
    "encoder.encoders.0.layers.0.self_attn_weights.in_proj.bias": "encoder.stage0.layer0.self_attn_weights.in_proj.bias",
    "onnx::MatMul_11203": "encoder.stage0.layer0.self_attn_weights.linear_pos.weight",
    "onnx::MatMul_11210": "encoder.stage0.layer0.feed_forward1.in_proj.weight",
    "encoder.encoders.0.layers.0.feed_forward1.in_proj.bias": "encoder.stage0.layer0.feed_forward1.in_proj.bias",
    "onnx::MatMul_11211": "encoder.stage0.layer0.feed_forward1.out_proj.weight",
    "encoder.encoders.0.layers.0.feed_forward1.out_proj.bias": "encoder.stage0.layer0.feed_forward1.out_proj.bias",
    "onnx::MatMul_11216": "encoder.stage0.layer0.nonlin_attention.in_proj.weight",
    "encoder.encoders.0.layers.0.nonlin_attention.in_proj.bias": "encoder.stage0.layer0.nonlin_attention.in_proj.bias",
    "onnx::MatMul_11220": "encoder.stage0.layer0.nonlin_attention.out_proj.weight",
    "encoder.encoders.0.layers.0.nonlin_attention.out_proj.bias": "encoder.stage0.layer0.nonlin_attention.out_proj.bias",
    "onnx::MatMul_11221": "encoder.stage0.layer0.self_attn1.in_proj.weight",
    "encoder.encoders.0.layers.0.self_attn1.in_proj.bias": "encoder.stage0.layer0.self_attn1.in_proj.bias",
    "onnx::MatMul_11223": "encoder.stage0.layer0.self_attn1.out_proj.weight",
    "encoder.encoders.0.layers.0.self_attn1.out_proj.bias": "encoder.stage0.layer0.self_attn1.out_proj.bias",
    "onnx::MatMul_11224": "encoder.stage0.layer0.conv_module1.in_proj.weight",
    "encoder.encoders.0.layers.0.conv_module1.in_proj.bias": "encoder.stage0.layer0.conv_module1.in_proj.bias",
    "encoder.encoders.0.layers.0.conv_module1.depthwise_conv.weight": "encoder.stage0.layer0.conv_module1.depthwise_conv.weight",
    "encoder.encoders.0.layers.0.conv_module1.depthwise_conv.bias": "encoder.stage0.layer0.conv_module1.depthwise_conv.bias",
    "onnx::MatMul_11225": "encoder.stage0.layer0.conv_module1.out_proj.weight",
    "encoder.encoders.0.layers.0.conv_module1.out_proj.bias": "encoder.stage0.layer0.conv_module1.out_proj.bias",
    "onnx::MatMul_11226": "encoder.stage0.layer0.feed_forward2.in_proj.weight",
    "encoder.encoders.0.layers.0.feed_forward2.in_proj.bias": "encoder.stage0.layer0.feed_forward2.in_proj.bias",
    "onnx::MatMul_11227": "encoder.stage0.layer0.feed_forward2.out_proj.weight",
    "encoder.encoders.0.layers.0.feed_forward2.out_proj.bias": "encoder.stage0.layer0.feed_forward2.out_proj.bias",
    "encoder.encoders.0.layers.0.bypass_mid.bypass_scale": "encoder.stage0.layer0.bypass_mid.bypass_scale",
    "onnx::MatMul_11228": "encoder.stage0.layer0.self_attn2.in_proj.weight",
    "encoder.encoders.0.layers.0.self_attn2.in_proj.bias": "encoder.stage0.layer0.self_attn2.in_proj.bias",
    "onnx::MatMul_11230": "encoder.stage0.layer0.self_attn2.out_proj.weight",
    "encoder.encoders.0.layers.0.self_attn2.out_proj.bias": "encoder.stage0.layer0.self_attn2.out_proj.bias",
    "onnx::MatMul_11231": "encoder.stage0.layer0.conv_module2.in_proj.weight",
    "encoder.encoders.0.layers.0.conv_module2.in_proj.bias": "encoder.stage0.layer0.conv_module2.in_proj.bias",
    "encoder.encoders.0.layers.0.conv_module2.depthwise_conv.weight": "encoder.stage0.layer0.conv_module2.depthwise_conv.weight",
    "encoder.encoders.0.layers.0.conv_module2.depthwise_conv.bias": "encoder.stage0.layer0.conv_module2.depthwise_conv.bias",
    "onnx::MatMul_11232": "encoder.stage0.layer0.conv_module2.out_proj.weight",
    "encoder.encoders.0.layers.0.conv_module2.out_proj.bias": "encoder.stage0.layer0.conv_module2.out_proj.bias",
    "onnx::MatMul_11233": "encoder.stage0.layer0.feed_forward3.in_proj.weight",
    "encoder.encoders.0.layers.0.feed_forward3.in_proj.bias": "encoder.stage0.layer0.feed_forward3.in_proj.bias",
    "onnx::MatMul_11234": "encoder.stage0.layer0.feed_forward3.out_proj.weight",
    "encoder.encoders.0.layers.0.feed_forward3.out_proj.bias": "encoder.stage0.layer0.feed_forward3.out_proj.bias",
    "encoder.encoders.0.layers.0.norm.log_scale": "encoder.stage0.layer0.norm.log_scale",
    "encoder.encoders.0.layers.0.norm.bias": "encoder.stage0.layer0.norm.bias",
    "encoder.encoders.0.layers.0.bypass.bypass_scale": "encoder.stage0.layer0.bypass.bypass_scale",
}


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
