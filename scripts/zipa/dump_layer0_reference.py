#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from lhotse.features.kaldi.extractors import Fbank, FbankConfig
from onnx import helper, TensorProto
from safetensors.numpy import save_file

FRONTEND_OUTPUT = "/encoder_embed/out_norm/Mul_1_output_0"
LAYER0_INPUT = "/encoder/Slice_output_0"
POS_EMB = "/encoder/0/encoder_pos/Unsqueeze_35_output_0"
ATTN_WEIGHTS = "/encoder/0/layers.0/self_attn_weights/Softmax_output_0"
ADD0 = "/encoder/0/layers.0/Add_output_0"
ADD1 = "/encoder/0/layers.0/Add_1_output_0"
ADD2 = "/encoder/0/layers.0/Add_2_output_0"
ADD3 = "/encoder/0/layers.0/Add_3_output_0"
ADD4 = "/encoder/0/layers.0/Add_4_output_0"
MID = "/encoder/0/layers.0/bypass_mid/Add_output_0"
ADD5 = "/encoder/0/layers.0/Add_5_output_0"
ADD6 = "/encoder/0/layers.0/Add_6_output_0"
ADD7 = "/encoder/0/layers.0/Add_7_output_0"
NORM = "/encoder/0/layers.0/norm/Mul_1_output_0"
LAYER0_OUTPUT = "/encoder/0/layers.0/bypass/Add_output_0"
LAYER1_ATTN_WEIGHTS = "/encoder/0/layers.1/self_attn_weights/Softmax_output_0"
LAYER1_OUTPUT = "/encoder/0/layers.1/bypass/Add_output_0"


def ensure_outputs(model_path: Path, output_path: Path) -> Path:
    model = onnx.load(model_path)
    existing = {o.name for o in model.graph.output}
    for name in [
        FRONTEND_OUTPUT,
        LAYER0_INPUT,
        POS_EMB,
        ATTN_WEIGHTS,
        ADD0,
        ADD1,
        ADD2,
        ADD3,
        ADD4,
        MID,
        ADD5,
        ADD6,
        ADD7,
        NORM,
        LAYER0_OUTPUT,
        LAYER1_ATTN_WEIGHTS,
        LAYER1_OUTPUT,
    ]:
        if name not in existing:
            model.graph.output.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )
    onnx.save(model, output_path)
    return output_path


def load_features(wav_path: Path) -> np.ndarray:
    samples, sr = sf.read(wav_path)
    if samples.ndim > 1:
        samples = samples[:, 0]
    feats = Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False)).extract(
        torch.from_numpy(samples).float(), sr
    )
    return feats.unsqueeze(0).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump ZIPA stage-0 encoder reference tensors for MLX comparison."
    )
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    patched = args.output.with_suffix(".patched.onnx")
    model_for_run = ensure_outputs(args.onnx_model, patched)

    features = load_features(args.wav)
    feat_lens = np.array([features.shape[1]], dtype=np.int64)

    session = ort.InferenceSession(str(model_for_run))
    outputs = session.run(
        [
            LAYER0_INPUT,
            POS_EMB,
            ATTN_WEIGHTS,
            ADD0,
            ADD1,
            ADD2,
            ADD3,
            ADD4,
            MID,
            ADD5,
            ADD6,
            ADD7,
            NORM,
            LAYER0_OUTPUT,
            LAYER1_ATTN_WEIGHTS,
            LAYER1_OUTPUT,
        ],
        {"x": features, "x_lens": feat_lens},
    )
    (
        layer0_in,
        pos_emb,
        attn_weights,
        add0,
        add1,
        add2,
        add3,
        add4,
        mid,
        add5,
        add6,
        add7,
        norm,
        layer0_out,
        layer1_attn_weights,
        layer1_out,
    ) = outputs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "layer0_in": layer0_in.astype(np.float32),
            "pos_emb": pos_emb.astype(np.float32),
            "attn_weights": attn_weights.astype(np.float32),
            "add0": add0.astype(np.float32),
            "add1": add1.astype(np.float32),
            "add2": add2.astype(np.float32),
            "add3": add3.astype(np.float32),
            "add4": add4.astype(np.float32),
            "mid": mid.astype(np.float32),
            "add5": add5.astype(np.float32),
            "add6": add6.astype(np.float32),
            "add7": add7.astype(np.float32),
            "norm": norm.astype(np.float32),
            "layer0_out": layer0_out.astype(np.float32),
            "layer1_attn_weights": layer1_attn_weights.astype(np.float32),
            "layer1_out": layer1_out.astype(np.float32),
            "stage0_out": layer1_out.astype(np.float32),
        },
        str(args.output),
    )
    print(f"wrote stage0 reference tensors to {args.output}")


if __name__ == "__main__":
    main()
