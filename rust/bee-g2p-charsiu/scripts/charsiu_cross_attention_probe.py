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
import re
from dataclasses import asdict, dataclass

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


WORD_RE = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)


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


def emitted_texts(tokenizer, output_ids: list[int]) -> list[str]:
    emitted = []
    previous = ""
    for step_end in range(2, len(output_ids) + 1):
        current = tokenizer.decode(output_ids[:step_end], skip_special_tokens=True)
        emitted.append(current[len(previous) :] if current.startswith(previous) else current)
        previous = current
    return emitted


@dataclass
class AttentionSummary:
    output_index: int
    output_piece: str
    emitted_text: str
    top_input_index: int
    top_input_piece: str
    top_score: float
    top_word_index: int | None
    top_word_surface: str | None
    top_word_score: float | None
    word_scores: list[dict[str, object]]
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


@dataclass
class WordSpan:
    index: int
    text: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int


def average_cross_attention(cross_attentions) -> torch.Tensor:
    # tuple[num_decode_steps][num_layers][batch, heads, tgt=1, src]
    per_step = []
    for step in cross_attentions:
        layers = torch.stack([layer[0].mean(dim=0).mean(dim=0) for layer in step], dim=0)
        per_step.append(layers.mean(dim=0).squeeze(0))
    return torch.stack(per_step, dim=0)


def average_teacher_forced_cross_attention(model, *, input_ids, attention_mask, decoder_input_ids) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    layers = torch.stack(
        [layer[0].mean(dim=0) for layer in outputs.cross_attentions],
        dim=0,
    )
    return layers.mean(dim=0)


def qwen_token_pieces(text: str, qwen_tokenizer) -> list[QwenTokenPiece]:
    encoded = qwen_tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return [
        QwenTokenPiece(
            index=index,
            token=token,
            char_start=start,
            char_end=end,
            surface=text[start:end],
            byte_start=len(text[:start].encode("utf-8")),
            byte_end=len(text[:end].encode("utf-8")),
        )
        for index, (token, (start, end)) in enumerate(
            zip(encoded.tokens(), encoded["offset_mapping"], strict=True)
        )
    ]


def word_spans(text: str) -> list[WordSpan]:
    return [
        WordSpan(
            index=index,
            text=match.group(0),
            char_start=match.start(),
            char_end=match.end(),
            byte_start=len(text[:match.start()].encode("utf-8")),
            byte_end=len(text[:match.end()].encode("utf-8")),
        )
        for index, match in enumerate(WORD_RE.finditer(text))
    ]


def span_scores(weights: torch.Tensor, *, text_byte_start: int, spans: list[object], label_key: str) -> list[dict[str, object]]:
    scores = []
    for span in spans:
        start = text_byte_start + span.byte_start
        end = text_byte_start + span.byte_end
        scores.append(
            {
                "index": span.index,
                label_key: getattr(span, label_key),
                "char_start": span.char_start,
                "char_end": span.char_end,
                "byte_start": span.byte_start,
                "byte_end": span.byte_end,
                "score": float(weights[start:end].sum().item()),
            }
        )
    return scores


@torch.no_grad()
def probe_text(
    text: str,
    *,
    lang_code: str,
    tokenizer,
    model,
    qwen_tokenizer,
    device: torch.device,
    max_length: int,
    top_k: int,
) -> dict[str, object]:
    prompt = f"<{lang_code}>: {text}"
    encoded = tokenizer(
        [prompt],
        padding=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **encoded,
        num_beams=1,
        max_length=max_length,
        do_sample=False,
    )

    input_ids = encoded["input_ids"][0].tolist()
    output_ids = generated_ids[0].tolist()
    input_pieces = decode_pieces(tokenizer, input_ids)
    output_pieces = decode_pieces(tokenizer, output_ids)
    output_emitted = emitted_texts(tokenizer, output_ids)
    teacher_forced_cross = average_teacher_forced_cross_attention(
        model,
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        decoder_input_ids=generated_ids[:, :-1],
    )
    text_bytes = list(text.encode("utf-8"))
    prompt_bytes = prompt.encode("utf-8")
    text_byte_start = len(prompt_bytes) - len(text_bytes)
    qwen_pieces = qwen_token_pieces(text, qwen_tokenizer)
    words = word_spans(text)

    summaries: list[AttentionSummary] = []
    for output_index, weights in enumerate(teacher_forced_cross):
        qwen_piece_scores = [
            {
                "piece_index": item["index"],
                "piece_token": item["token"],
                "piece_surface": qwen_pieces[item["index"]].surface,
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "byte_start": item["byte_start"],
                "byte_end": item["byte_end"],
                "score": item["score"],
            }
            for item in span_scores(weights, text_byte_start=text_byte_start, spans=qwen_pieces, label_key="token")
        ]
        word_scores = [
            {
                "word_index": item["index"],
                "word_text": item["text"],
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "byte_start": item["byte_start"],
                "byte_end": item["byte_end"],
                "score": item["score"],
            }
            for item in span_scores(weights, text_byte_start=text_byte_start, spans=words, label_key="text")
        ]
        top_qwen = max(qwen_piece_scores, key=lambda item: item["score"], default=None)
        top_word = max(word_scores, key=lambda item: item["score"], default=None)
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
                output_piece=output_pieces[output_index + 1],
                emitted_text=output_emitted[output_index],
                top_input_index=int(top["input_index"]),
                top_input_piece=str(top["input_piece"]),
                top_score=float(top["score"]),
                top_word_index=None if top_word is None else int(top_word["word_index"]),
                top_word_surface=None if top_word is None else str(top_word["word_text"]),
                top_word_score=None if top_word is None else float(top_word["score"]),
                word_scores=word_scores,
                top_qwen_piece_index=None if top_qwen is None else int(top_qwen["piece_index"]),
                top_qwen_piece_token=None if top_qwen is None else str(top_qwen["piece_token"]),
                top_qwen_piece_score=None if top_qwen is None else float(top_qwen["score"]),
                qwen_piece_scores=qwen_piece_scores,
                ranked_inputs=ranked[:top_k],
            )
        )

    return {
        "text": text,
        "lang_code": lang_code,
        "prompt": prompt,
        "device": str(device),
        "text_bytes": text_bytes,
        "word_spans": [asdict(word) for word in words],
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
        description="Inspect Charsiu decoder cross-attention for one word or phrase."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--word")
    group.add_argument("--text")
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
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def print_report(result: dict[str, object]) -> None:
    print(f"text         : {result['text']}")
    print(f"lang_code    : {result['lang_code']}")
    print(f"device       : {result['device']}")
    print(f"prompt       : {result['prompt']}")
    print(f"decoded ipa  : {result['decoded_output']}")
    print(f"text bytes   : {result['text_bytes']}")
    print("word spans   :")
    for word in result["word_spans"]:
        print(
            f"  [{word['index']}] {word['text']!r} "
            f"{word['char_start']}..{word['char_end']} "
            f"bytes {word['byte_start']}..{word['byte_end']}"
        )
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
                f"  word span : [{row['top_word_index']}] "
                f"{row['top_word_surface']!r} score={row['top_word_score']:.4f}"
            )
            word_scores = " ; ".join(
                f"{item['word_index']}:{item['word_text']!r}={item['score']:.4f}"
                for item in row["word_scores"]
            )
            print(f"  word mass : {word_scores}")
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


def print_summary(result: dict[str, object]) -> None:
    print(f"text        : {result['text']}")
    print(f"decoded ipa : {result['decoded_output']}")
    print("word spans  :")
    for word in result["word_spans"]:
        print(f"  [{word['index']}] {word['text']!r} {word['char_start']}..{word['char_end']}")
    print("qwen pieces :")
    for piece in result["qwen_token_pieces"]:
        print(f"  [{piece['index']}] {piece['token']!r} {piece['char_start']}..{piece['char_end']} -> {piece['surface']!r}")
    print()
    print("ownership:")
    for row in result["cross_attention"]:
        piece = row["output_piece"]
        if piece in {"", "</s>"}:
            continue
        print(
            f"  out[{row['output_index']:>2}] {piece!r} "
            f"word[{row['top_word_index']}]={row['top_word_surface']!r}:{row['top_word_score']:.4f} "
            f"qwen[{row['top_qwen_piece_index']}]={row['top_qwen_piece_token']!r}:{row['top_qwen_piece_score']:.4f}"
        )


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if not args.qwen_tokenizer_path:
        raise SystemExit("missing --qwen-tokenizer-path or BEE_ASR_MODEL_DIR")
    qwen_tokenizer = AutoTokenizer.from_pretrained(args.qwen_tokenizer_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()
    text = args.word if args.word is not None else args.text
    result = probe_text(
        text,
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
    elif args.summary:
        print_summary(result)
    else:
        print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
