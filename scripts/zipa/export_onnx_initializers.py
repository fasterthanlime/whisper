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
