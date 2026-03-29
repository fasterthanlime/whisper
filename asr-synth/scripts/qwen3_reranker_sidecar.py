#!/usr/bin/env python3
import json
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
MAX_LENGTH = 8192
PREFIX = (
    '<|im_start|>system\n'
    'Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
    'Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n'
    '<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_instruction(instruction, query, doc):
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class Reranker:
    def __init__(self):
        device = pick_device()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
        model_kwargs = {}
        if device in {"mps", "cuda"}:
            model_kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs).eval().to(device)
        self.prefix_tokens = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=MAX_LENGTH - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def score(self, instruction, query, documents):
        pairs = [format_instruction(instruction, query, doc) for doc in documents]
        t0 = time.perf_counter()
        inputs = self.process_inputs(pairs)
        t1 = time.perf_counter()
        logits = self.model(**inputs).logits[:, -1, :]
        true_vector = logits[:, self.token_true_id]
        false_vector = logits[:, self.token_false_id]
        stacked = torch.stack([false_vector, true_vector], dim=1)
        scores = torch.nn.functional.log_softmax(stacked, dim=1)[:, 1].exp().tolist()
        t2 = time.perf_counter()
        return {
            "scores": scores,
            "timing_ms": {
                "tokenize": round((t1 - t0) * 1000),
                "forward": round((t2 - t1) * 1000),
                "total": round((t2 - t0) * 1000),
            },
        }


def main():
    reranker = Reranker()
    print(json.dumps({"ready": True, "model": MODEL_ID, "device": reranker.device}), flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            instruction = req["instruction"]
            query = req["query"]
            documents = req["documents"]
            out = reranker.score(instruction, query, documents)
            out["model"] = MODEL_ID
            out["device"] = reranker.device
            print(json.dumps(out), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
