#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentencepiece",
#   "torch",
#   "transformers",
# ]
# ///
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def decode_pieces(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in token_ids]


@dataclass
class AttentionSummary:
    output_index: int
    output_piece: str
    top_input_index: int
    top_input_piece: str
    top_score: float
    top_qwen_piece_index: int | None
    top_qwen_piece_token: str | None
    top_qwen_piece_score: float | None
    qwen_piece_scores: list[dict[str, object]]
    ranked_inputs: list[dict[str, object]]


@dataclass
class QwenTokenPiece:
    index: int
    token: str
    char_start: int
    char_end: int
    surface: str
    byte_start: int
    byte_end: int


def average_cross_attention(cross_attentions) -> torch.Tensor:
    # tuple[num_decode_steps][num_layers][batch, heads, tgt=1, src]
    per_step = []
    for step in cross_attentions:
        layers = torch.stack([layer[0].mean(dim=0).mean(dim=0) for layer in step], dim=0)
        per_step.append(layers.mean(dim=0).squeeze(0))
    return torch.stack(per_step, dim=0)


def qwen_token_pieces(word: str, qwen_tokenizer) -> list[QwenTokenPiece]:
    encoded = qwen_tokenizer(word, add_special_tokens=False, return_offsets_mapping=True)
    return [
        QwenTokenPiece(
            index=index,
            token=token,
            char_start=start,
            char_end=end,
            surface=word[start:end],
            byte_start=len(word[:start].encode("utf-8")),
            byte_end=len(word[:end].encode("utf-8")),
        )
        for index, (token, (start, end)) in enumerate(
            zip(encoded.tokens(), encoded["offset_mapping"], strict=True)
        )
    ]


@torch.no_grad()
def probe_word(
    word: str,
    *,
    lang_code: str,
    tokenizer,
    model,
    qwen_tokenizer,
    device: torch.device,
    max_length: int,
    top_k: int,
) -> dict[str, object]:
    prompt = f"<{lang_code}>: {word}"
    encoded = tokenizer(
        [prompt],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    generated = model.generate(
        **encoded,
        num_beams=1,
        max_length=max_length,
        do_sample=False,
        return_dict_in_generate=True,
        output_attentions=True,
    )

    input_ids = encoded["input_ids"][0].tolist()
    output_ids = generated.sequences[0].tolist()
    input_pieces = decode_pieces(tokenizer, input_ids)
    output_pieces = decode_pieces(tokenizer, output_ids)
    cross = average_cross_attention(generated.cross_attentions)
    word_bytes = list(word.encode("utf-8"))
    prompt_bytes = prompt.encode("utf-8")
    word_byte_start = len(prompt_bytes) - len(word_bytes)
    qwen_pieces = qwen_token_pieces(word, qwen_tokenizer)

    summaries: list[AttentionSummary] = []
    for output_index, weights in enumerate(cross):
        qwen_piece_scores = []
        for piece in qwen_pieces:
            start = word_byte_start + piece.byte_start
            end = word_byte_start + piece.byte_end
            score = float(sum(weights[start:end]).item())
            qwen_piece_scores.append(
                {
                    "piece_index": piece.index,
                    "piece_token": piece.token,
                    "piece_surface": piece.surface,
                    "score": score,
                }
            )
        top_qwen = max(qwen_piece_scores, key=lambda item: item["score"], default=None)
        ranked = sorted(
            [
                {
                    "input_index": input_index,
                    "input_piece": input_pieces[input_index],
                    "score": float(score),
                }
                for input_index, score in enumerate(weights.tolist())
            ],
            key=lambda item: item["score"],
            reverse=True,
        )
        top = ranked[0]
        summaries.append(
            AttentionSummary(
                output_index=output_index,
                output_piece=output_pieces[output_index + 1]
                if output_index + 1 < len(output_pieces)
                else "",
                top_input_index=int(top["input_index"]),
                top_input_piece=str(top["input_piece"]),
                top_score=float(top["score"]),
                top_qwen_piece_index=None if top_qwen is None else int(top_qwen["piece_index"]),
                top_qwen_piece_token=None if top_qwen is None else str(top_qwen["piece_token"]),
                top_qwen_piece_score=None if top_qwen is None else float(top_qwen["score"]),
                qwen_piece_scores=qwen_piece_scores,
                ranked_inputs=ranked[:top_k],
            )
        )

    return {
        "word": word,
        "lang_code": lang_code,
        "prompt": prompt,
        "device": str(device),
        "word_bytes": word_bytes,
        "qwen_token_pieces": [asdict(piece) for piece in qwen_pieces],
        "input_ids": input_ids,
        "input_pieces": input_pieces,
        "output_ids": output_ids,
        "output_pieces": output_pieces,
        "decoded_output": tokenizer.decode(output_ids, skip_special_tokens=True),
        "cross_attention": [asdict(summary) for summary in summaries],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Charsiu decoder cross-attention for one word."
    )
    parser.add_argument("--word", required=True)
    parser.add_argument("--lang-code", default="eng-us")
    parser.add_argument("--model-id", default="charsiu/g2p_multilingual_byT5_tiny_16_layers")
    parser.add_argument("--tokenizer-id", default="google/byt5-small")
    parser.add_argument(
        "--qwen-tokenizer-path",
        default=os.environ.get("BEE_ASR_MODEL_DIR", ""),
        help="Local Qwen tokenizer/model dir; defaults to BEE_ASR_MODEL_DIR",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def print_report(result: dict[str, object]) -> None:
    print(f"word         : {result['word']}")
    print(f"lang_code    : {result['lang_code']}")
    print(f"device       : {result['device']}")
    print(f"prompt       : {result['prompt']}")
    print(f"decoded ipa  : {result['decoded_output']}")
    print(f"word bytes   : {result['word_bytes']}")
    print("qwen pieces  :")
    for piece in result["qwen_token_pieces"]:
        print(
            f"  [{piece['index']}] {piece['token']!r} "
            f"{piece['char_start']}..{piece['char_end']} "
            f"bytes {piece['byte_start']}..{piece['byte_end']} -> {piece['surface']!r}"
        )
    print(f"input pieces : {' | '.join(result['input_pieces'])}")
    print(f"output pieces: {' | '.join(result['output_pieces'])}")
    print()
    for row in result["cross_attention"]:
        print(
            f"out[{row['output_index']:>2}] {row['output_piece']!r} -> "
            f"in[{row['top_input_index']}] {row['top_input_piece']!r} "
            f"score={row['top_score']:.4f}"
        )
        if row["top_qwen_piece_index"] is not None:
            print(
                f"  qwen piece: [{row['top_qwen_piece_index']}] "
                f"{row['top_qwen_piece_token']!r} score={row['top_qwen_piece_score']:.4f}"
            )
            piece_scores = " ; ".join(
                f"{item['piece_index']}:{item['piece_token']!r}={item['score']:.4f}"
                for item in row["qwen_piece_scores"]
            )
            print(f"  qwen mass: {piece_scores}")
        ranked = " ; ".join(
            f"{item['input_index']}:{item['input_piece']!r}={item['score']:.4f}"
            for item in row["ranked_inputs"]
        )
        print(f"  top: {ranked}")


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if not args.qwen_tokenizer_path:
        raise SystemExit("missing --qwen-tokenizer-path or BEE_ASR_MODEL_DIR")
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()
    result = probe_word(
        args.word,
        lang_code=args.lang_code,
        tokenizer=tokenizer,
        model=model,
        qwen_tokenizer=qwen_tokenizer,
        device=device,
        max_length=args.max_length,
        top_k=args.top_k,
    )
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
