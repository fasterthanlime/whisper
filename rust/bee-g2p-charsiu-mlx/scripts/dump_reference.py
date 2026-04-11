#!/usr/bin/env python3
"""Dump intermediate T5 activations for layer-by-layer Rust/MLX validation."""
# /// script
# dependencies = ["torch", "transformers", "safetensors", "packaging", "numpy"]
# ///

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def dump_reference(model_id: str, out_dir: Path, text: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.eval()

    # Encode input
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded["input_ids"]  # [1, seq_len]

    tensors: dict[str, torch.Tensor] = {}
    tensors["input_ids"] = input_ids.to(torch.int32).contiguous()

    # =========================================================================
    # 1. Encoder layer-by-layer
    # =========================================================================
    with torch.no_grad():
        # Embedding
        encoder = model.encoder
        x = encoder.embed_tokens(input_ids)
        tensors["encoder.embed_out"] = x[0].contiguous()  # [seq_len, d_model]

        # Position bias (from first block only)
        seq_len = x.shape[1]
        position_bias = encoder.block[0].layer[0].SelfAttention.compute_bias(seq_len, seq_len)
        tensors["encoder.position_bias"] = position_bias[0].contiguous()  # [heads, qlen, klen]

        # Each encoder block
        for i, block in enumerate(encoder.block):
            # Self-attention sublayer
            normed = block.layer[0].layer_norm(x)
            attn_out = block.layer[0].SelfAttention(
                normed, position_bias=position_bias
            )
            attn_hidden = attn_out[0]
            x_after_attn = x + attn_hidden
            tensors[f"encoder.block.{i}.attn_out"] = attn_hidden[0].contiguous()
            tensors[f"encoder.block.{i}.after_attn"] = x_after_attn[0].contiguous()

            # FFN sublayer
            normed_ff = block.layer[1].layer_norm(x_after_attn)
            ff_out = block.layer[1].DenseReluDense(normed_ff)
            x = x_after_attn + ff_out
            tensors[f"encoder.block.{i}.ff_out"] = ff_out[0].contiguous()
            tensors[f"encoder.block.{i}.out"] = x[0].contiguous()

        # Final layer norm
        encoder_output = encoder.final_layer_norm(x)
        tensors["encoder.final_out"] = encoder_output[0].contiguous()

    # =========================================================================
    # 2. Full generation to get decoder reference
    # =========================================================================
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            num_beams=1,
            max_length=64,
            do_sample=False,
        )
        output_ids = generated[0].tolist()
        tensors["generated_ids"] = generated[0].to(torch.int32).contiguous()

    # =========================================================================
    # 3. Decoder layer-by-layer (teacher-forced with generated output)
    # =========================================================================
    with torch.no_grad():
        decoder = model.decoder
        # Use the generated sequence as decoder input (shifted right)
        decoder_input_ids = model._shift_right(generated)
        tensors["decoder_input_ids"] = decoder_input_ids[0].to(torch.int32).contiguous()

        x = decoder.embed_tokens(decoder_input_ids)
        tensors["decoder.embed_out"] = x[0].contiguous()

        dec_len = x.shape[1]
        enc_len = encoder_output.shape[1]

        # Self-attention position bias (is_decoder=True, so already causal)
        self_attn_bias = decoder.block[0].layer[0].SelfAttention.compute_bias(dec_len, dec_len)
        tensors["decoder.self_attn_bias"] = self_attn_bias[0].contiguous()

        # No separate causal mask needed — the decoder position bias already
        # encodes causality (future positions mapped to bucket 0 with large negative values).

        for i, block in enumerate(decoder.block):
            # Self-attention (no mask — causality is in the position bias)
            normed = block.layer[0].layer_norm(x)
            self_attn_out = block.layer[0].SelfAttention(
                normed,
                position_bias=self_attn_bias,
            )
            x_after_self_attn = x + self_attn_out[0]
            tensors[f"decoder.block.{i}.self_attn_out"] = self_attn_out[0][0].contiguous()

            # Cross-attention
            normed = block.layer[1].layer_norm(x_after_self_attn)
            cross_attn_out = block.layer[1].EncDecAttention(
                normed,
                key_value_states=encoder_output,
            )
            x_after_cross_attn = x_after_self_attn + cross_attn_out[0]
            tensors[f"decoder.block.{i}.cross_attn_out"] = cross_attn_out[0][0].contiguous()

            # FFN
            normed = block.layer[2].layer_norm(x_after_cross_attn)
            ff_out = block.layer[2].DenseReluDense(normed)
            x = x_after_cross_attn + ff_out
            tensors[f"decoder.block.{i}.out"] = x[0].contiguous()

        decoder_output = decoder.final_layer_norm(x)
        tensors["decoder.final_out"] = decoder_output[0].contiguous()

        # LM head logits
        logits = model.lm_head(decoder_output)
        tensors["logits"] = logits[0].contiguous()

    # =========================================================================
    # Save
    # =========================================================================
    out_path = out_dir / "reference.safetensors"
    # Clone everything to avoid shared memory
    tensors = {k: v.clone().contiguous() for k, v in tensors.items()}
    save_file(tensors, str(out_path))

    # Print summary
    print(f"Saved {len(tensors)} reference tensors to {out_path}")
    print(f"Input: {text!r}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Generated IDs: {output_ids}")
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Decoded output: {decoded!r}")
    for k, v in sorted(tensors.items()):
        print(f"  {k}: {list(v.shape)} {v.dtype}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="charsiu/g2p_multilingual_byT5_tiny_16_layers",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.home() / ".bearcove" / "charsiu-g2p" / "reference",
    )
    parser.add_argument(
        "--text",
        default="<eng-us>: Facet",
    )
    args = parser.parse_args()
    dump_reference(args.model_id, args.out_dir, args.text)


if __name__ == "__main__":
    main()
