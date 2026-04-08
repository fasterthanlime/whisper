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


def ensure_frontend_output(model_path: Path, output_path: Path) -> Path:
    model = onnx.load(model_path)
    if any(o.name == FRONTEND_OUTPUT for o in model.graph.output):
        return model_path

    graph = model.graph
    graph.output.append(
        helper.make_tensor_value_info(FRONTEND_OUTPUT, TensorProto.FLOAT, None)
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
        description="Dump ZIPA frontend input/output tensors for MLX numeric comparison."
    )
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("wav", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    patched = args.output.with_suffix(".patched.onnx")
    model_for_run = ensure_frontend_output(args.onnx_model, patched)

    features = load_features(args.wav)
    feat_lens = np.array([features.shape[1]], dtype=np.int64)

    session = ort.InferenceSession(str(model_for_run))
    outputs = session.run([FRONTEND_OUTPUT, "log_probs", "log_probs_len"], {"x": features, "x_lens": feat_lens})
    frontend_out, log_probs, log_probs_len = outputs

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "features": features.astype(np.float32),
            "frontend_out": frontend_out.astype(np.float32),
            "log_probs": log_probs.astype(np.float32),
            "log_probs_len": log_probs_len.astype(np.int64),
        },
        str(args.output),
    )
    print(f"wrote frontend reference tensors to {args.output}")


if __name__ == "__main__":
    main()
