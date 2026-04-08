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

STAGE1_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11352",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11371",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11378",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11379",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11384",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11388",
    "self_attn1.in_proj.weight": "onnx::MatMul_11389",
    "self_attn1.out_proj.weight": "onnx::MatMul_11391",
    "conv_module1.in_proj.weight": "onnx::MatMul_11392",
    "conv_module1.out_proj.weight": "onnx::MatMul_11393",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11394",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11395",
    "self_attn2.in_proj.weight": "onnx::MatMul_11396",
    "self_attn2.out_proj.weight": "onnx::MatMul_11398",
    "conv_module2.in_proj.weight": "onnx::MatMul_11399",
    "conv_module2.out_proj.weight": "onnx::MatMul_11400",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11401",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11402",
}

STAGE2_LAYER0_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11422",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11441",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11448",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11449",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11454",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11458",
    "self_attn1.in_proj.weight": "onnx::MatMul_11459",
    "self_attn1.out_proj.weight": "onnx::MatMul_11461",
    "conv_module1.in_proj.weight": "onnx::MatMul_11462",
    "conv_module1.out_proj.weight": "onnx::MatMul_11463",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11464",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11465",
    "self_attn2.in_proj.weight": "onnx::MatMul_11466",
    "self_attn2.out_proj.weight": "onnx::MatMul_11468",
    "conv_module2.in_proj.weight": "onnx::MatMul_11469",
    "conv_module2.out_proj.weight": "onnx::MatMul_11470",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11471",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11472",
}

STAGE2_LAYER1_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11473",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11492",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11499",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11500",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11505",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11509",
    "self_attn1.in_proj.weight": "onnx::MatMul_11510",
    "self_attn1.out_proj.weight": "onnx::MatMul_11512",
    "conv_module1.in_proj.weight": "onnx::MatMul_11513",
    "conv_module1.out_proj.weight": "onnx::MatMul_11514",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11515",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11516",
    "self_attn2.in_proj.weight": "onnx::MatMul_11517",
    "self_attn2.out_proj.weight": "onnx::MatMul_11519",
    "conv_module2.in_proj.weight": "onnx::MatMul_11520",
    "conv_module2.out_proj.weight": "onnx::MatMul_11521",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11522",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11523",
}

STAGE2_LAYER2_MATMULS = {
    "self_attn_weights.in_proj.weight": "onnx::MatMul_11524",
    "self_attn_weights.linear_pos.weight": "onnx::MatMul_11543",
    "feed_forward1.in_proj.weight": "onnx::MatMul_11550",
    "feed_forward1.out_proj.weight": "onnx::MatMul_11551",
    "nonlin_attention.in_proj.weight": "onnx::MatMul_11556",
    "nonlin_attention.out_proj.weight": "onnx::MatMul_11560",
    "self_attn1.in_proj.weight": "onnx::MatMul_11561",
    "self_attn1.out_proj.weight": "onnx::MatMul_11563",
    "conv_module1.in_proj.weight": "onnx::MatMul_11564",
    "conv_module1.out_proj.weight": "onnx::MatMul_11565",
    "feed_forward2.in_proj.weight": "onnx::MatMul_11566",
    "feed_forward2.out_proj.weight": "onnx::MatMul_11567",
    "self_attn2.in_proj.weight": "onnx::MatMul_11568",
    "self_attn2.out_proj.weight": "onnx::MatMul_11570",
    "conv_module2.in_proj.weight": "onnx::MatMul_11571",
    "conv_module2.out_proj.weight": "onnx::MatMul_11572",
    "feed_forward3.in_proj.weight": "onnx::MatMul_11573",
    "feed_forward3.out_proj.weight": "onnx::MatMul_11574",
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

for dst_suffix, src_name in STAGE1_LAYER1_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage1.layer1.{dst_suffix}"

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
    NAME_MAP[f"encoder.encoders.1.encoder.layers.1.{suffix}"] = (
        f"encoder.stage1.layer1.{suffix}"
    )

NAME_MAP["encoder.encoders.1.out_combiner.bypass_scale"] = "encoder.stage1.out_combiner.bypass_scale"

NAME_MAP["onnx::Mul_11417"] = "encoder.stage2.downsample.weights"

for dst_suffix, src_name in STAGE2_LAYER0_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage2.layer0.{dst_suffix}"
for dst_suffix, src_name in STAGE2_LAYER1_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage2.layer1.{dst_suffix}"
for dst_suffix, src_name in STAGE2_LAYER2_MATMULS.items():
    NAME_MAP[src_name] = f"encoder.stage2.layer2.{dst_suffix}"

for layer_index in range(3):
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
        NAME_MAP[f"encoder.encoders.2.encoder.layers.{layer_index}.{suffix}"] = (
            f"encoder.stage2.layer{layer_index}.{suffix}"
        )

NAME_MAP["encoder.encoders.2.out_combiner.bypass_scale"] = "encoder.stage2.out_combiner.bypass_scale"


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
