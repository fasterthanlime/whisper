#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def strip_eot(text: str) -> str:
    return text.replace("<|endoftext|>", "").strip()


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def load_jsonl(path: Path, max_examples: int | None = None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            rows.append(
                {
                    "prompt": obj["prompt"].strip(),
                    "completion": strip_eot(obj["completion"]),
                }
            )
            if max_examples is not None and len(rows) >= max_examples:
                break
    return rows


class PromptCompletionDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.rows[index]


@dataclass
class BatchCollator:
    tokenizer: Any
    max_source_length: int
    max_target_length: int

    def __call__(self, rows: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompts = [row["prompt"] for row in rows]
        completions = [row["completion"] for row in rows]
        model_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=completions,
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def exact_match_rate(
    model: AutoModelForSeq2SeqLM,
    tokenizer: Any,
    rows: list[dict[str, str]],
    device: torch.device,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
) -> tuple[float, list[dict[str, str]]]:
    model.eval()
    examples: list[dict[str, str]] = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            prompts = [row["prompt"] for row in chunk]
            expected = [row["completion"] for row in chunk]
            encoded = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_source_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generated = model.generate(
                **encoded,
                max_new_tokens=max_target_length,
                do_sample=False,
                num_beams=1,
            )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for prompt, pred, gold in zip(prompts, decoded, expected, strict=True):
                pred_norm = normalize_text(pred)
                gold_norm = normalize_text(gold)
                if pred_norm == gold_norm:
                    correct += 1
                elif len(examples) < 8:
                    examples.append(
                        {
                            "prompt": prompt,
                            "predicted": pred_norm,
                            "expected": gold_norm,
                        }
                    )
    return (correct / max(len(rows), 1) * 100.0), examples


def mean_eval_loss(
    model: AutoModelForSeq2SeqLM,
    data_loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            outputs = model(**move_batch(batch, device))
            total_loss += float(outputs.loss.item())
            total_batches += 1
    if total_batches == 0:
        return math.nan
    return total_loss / total_batches


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone ByT5 fine-tune for ASR correction.")
    parser.add_argument("--model", default="google/byt5-small")
    parser.add_argument("--train-file", default="training/data/train.jsonl")
    parser.add_argument("--valid-file", default="training/data/valid.jsonl")
    parser.add_argument("--test-file", default="training/data/test.jsonl")
    parser.add_argument("--output-dir", default="training/byt5-small-run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-source-length", type=int, default=768)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--max-train-examples", type=int)
    parser.add_argument("--max-valid-examples", type=int)
    parser.add_argument("--max-test-examples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = pick_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(Path(args.train_file), args.max_train_examples)
    valid_rows = load_jsonl(Path(args.valid_file), args.max_valid_examples)
    test_rows = load_jsonl(Path(args.test_file), args.max_test_examples)

    print(f"device={device}")
    print(
        f"train={len(train_rows)} valid={len(valid_rows)} test={len(test_rows)} "
        f"model={args.model}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.to(device)

    collator = BatchCollator(
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
    )
    train_loader = DataLoader(
        PromptCompletionDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=False,
    )
    valid_loader = DataLoader(
        PromptCompletionDataset(valid_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=False,
    )
    test_loader = DataLoader(
        PromptCompletionDataset(test_rows),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=False,
    )

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_valid_loss = float("inf")
    history: list[dict[str, Any]] = []

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        epoch_started = time.time()

        for step, batch in enumerate(train_loader, start=1):
            outputs = model(**move_batch(batch, device))
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += float(outputs.loss.item())
            if step % args.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        train_loss = running_loss / max(len(train_loader), 1)
        valid_loss = mean_eval_loss(model, valid_loader, device)
        valid_exact, bad_examples = exact_match_rate(
            model,
            tokenizer,
            valid_rows,
            device,
            args.eval_batch_size,
            args.max_source_length,
            args.max_target_length,
        )
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_exact_match": valid_exact,
            "steps": global_step,
            "elapsed_sec": round(time.time() - epoch_started, 2),
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            with (output_dir / "best_metrics.json").open("w") as f:
                json.dump(
                    {
                        **epoch_metrics,
                        "bad_examples": bad_examples,
                        "train_rows": len(train_rows),
                        "valid_rows": len(valid_rows),
                        "test_rows": len(test_rows),
                    },
                    f,
                    indent=2,
                )

        if args.save_every_epoch:
            epoch_dir = output_dir / f"epoch-{epoch}"
            epoch_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_dir)
            tokenizer.save_pretrained(epoch_dir)

    best_model = AutoModelForSeq2SeqLM.from_pretrained(output_dir).to(device)
    test_exact, test_bad_examples = exact_match_rate(
        best_model,
        tokenizer,
        test_rows,
        device,
        args.eval_batch_size,
        args.max_source_length,
        args.max_target_length,
    )
    test_loss = mean_eval_loss(best_model, test_loader, device)
    summary = {
        "model": args.model,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_source_length": args.max_source_length,
        "max_target_length": args.max_target_length,
        "best_valid_loss": best_valid_loss,
        "test_exact_match": test_exact,
        "test_loss": test_loss,
        "history": history,
        "test_bad_examples": test_bad_examples,
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
