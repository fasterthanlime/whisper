#!/usr/bin/env python3
import json
import sys
import time
from typing import List

import mlx.core as mx
from mlx_lm import load


def _usage() -> None:
    print("usage: prototype_reranker_sidecar.py MODEL_ID [ADAPTER_PATH]", file=sys.stderr)


if len(sys.argv) < 2:
    _usage()
    sys.exit(2)

MODEL_ID = sys.argv[1]
ADAPTER_PATH = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None


def encode_prompt(tokenizer, prompt: str) -> List[int]:
    bos = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos is None or not prompt.startswith(bos)
    return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)


class Reranker:
    def __init__(self):
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(MODEL_ID, adapter_path=ADAPTER_PATH)
        self.model.eval()
        self.yes_ids = self.tokenizer.encode(" yes", add_special_tokens=False)
        self.no_ids = self.tokenizer.encode(" no", add_special_tokens=False)
        if not self.yes_ids or not self.no_ids:
            raise RuntimeError("failed to encode yes/no labels for reranker scoring")
        t1 = time.perf_counter()
        self.load_ms = round((t1 - t0) * 1000)

    def sequence_logprob(self, prompt_ids: List[int], continuation_ids: List[int]) -> float:
        input_ids = prompt_ids + continuation_ids[:-1]
        logits = self.model(mx.array([input_ids]))
        logits = logits[:, -len(continuation_ids) :, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target = mx.array(continuation_ids)[None, :, None]
        picked = mx.take_along_axis(logprobs, target, axis=-1).squeeze(-1)
        mx.eval(picked)
        return float(picked.sum().item())

    def score_prompt(self, prompt: str):
        prompt_ids = encode_prompt(self.tokenizer, prompt)
        yes_lp = self.sequence_logprob(prompt_ids, self.yes_ids)
        no_lp = self.sequence_logprob(prompt_ids, self.no_ids)
        norm = max(yes_lp, no_lp)
        yes_prob = float(mx.exp(mx.array(yes_lp - norm)).item())
        no_prob = float(mx.exp(mx.array(no_lp - norm)).item())
        denom = yes_prob + no_prob
        if denom <= 0:
            yes_prob, no_prob = 0.5, 0.5
        else:
            yes_prob /= denom
            no_prob /= denom
        return {
            "yes_prob": yes_prob,
            "no_prob": no_prob,
            "answer": "yes" if yes_prob >= no_prob else "no",
            "label_logprobs": {
                "yes": yes_lp,
                "no": no_lp,
            },
        }

    def score(self, prompts: List[str]):
        t0 = time.perf_counter()
        results = [self.score_prompt(prompt) for prompt in prompts]
        t1 = time.perf_counter()
        return {
            "results": results,
            "timing_ms": {
                "load": self.load_ms,
                "score": round((t1 - t0) * 1000),
                "total": self.load_ms + round((t1 - t0) * 1000),
            },
        }


def main():
    reranker = Reranker()
    print(
        json.dumps(
            {
                "ready": True,
                "model": MODEL_ID,
                "adapters": ADAPTER_PATH,
                "backend": "mlx-lm",
                "timing_ms": {"load": reranker.load_ms},
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            prompts = req["prompts"]
            out = reranker.score(prompts)
            out["model"] = MODEL_ID
            out["adapters"] = ADAPTER_PATH
            out["backend"] = "mlx-lm"
            print(json.dumps(out), flush=True)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
