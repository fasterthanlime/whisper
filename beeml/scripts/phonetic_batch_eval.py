#!/usr/bin/env python3
import argparse
import json
import random
import re
import sqlite3
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DB_URI = f"file:{ROOT / 'corpus.db'}?mode=ro&immutable=1"
BIN = ROOT / "target" / "debug" / "synth-dashboard"
ESPEAK = Path("/opt/homebrew/bin/espeak-ng")


def espeak_ipa(text: str) -> str:
    out = subprocess.check_output([str(ESPEAK), "-q", "--ipa=3", text], text=True).strip()
    out = out.replace("ˈ", " ").replace("ˌ", " ").replace("\u200d", "").replace("‍", "")
    return " ".join(out.split())


def score_pair(obs_text: str, cand_text: str):
    obs_ipa = espeak_ipa(obs_text)
    cand_ipa = espeak_ipa(cand_text)
    raw = subprocess.check_output([str(BIN), "phonetic-score", obs_ipa, cand_ipa], text=True)
    data = json.loads(raw)
    if data is None:
        return None
    return {
        "score": float(data["blended_score"]),
        "obs_ipa": obs_ipa,
        "cand_ipa": cand_ipa,
    }


def summarize(items):
    xs = sorted(x["score"] for x in items)

    def q(p: float) -> float:
        return xs[min(len(xs) - 1, int((len(xs) - 1) * p))]

    return {
        "n": len(xs),
        "min": round(xs[0], 3),
        "p25": round(q(0.25), 3),
        "median": round(q(0.50), 3),
        "p75": round(q(0.75), 3),
        "max": round(xs[-1], 3),
        "mean": round(statistics.mean(xs), 3),
    }


def looks_acronymish(text: str) -> bool:
    if not text:
        return False
    compact = re.sub(r"[^A-Za-z0-9]", "", text)
    if len(compact) >= 2 and compact.isupper():
        return True
    upper_count = sum(1 for ch in text if ch.isupper())
    digit_count = sum(1 for ch in text if ch.isdigit())
    if upper_count + digit_count >= 2 and len(compact) <= 10:
        return True
    return False


def summarize_bucket(items):
    if not items:
        return None
    return {
        "count": len(items),
        "summary": summarize(items),
    }


def policy_eval(positives, negatives, acronym_threshold: float, normal_threshold: float):
    pos_kept = 0
    neg_kept = 0
    for item in positives:
        threshold = acronym_threshold if item["term_acronymish"] else normal_threshold
        if item["score"] >= threshold:
            pos_kept += 1
    for item in negatives:
        threshold = acronym_threshold if item["wrong_term_acronymish"] else normal_threshold
        if item["score"] >= threshold:
            neg_kept += 1
    return {
        "acronym_threshold": acronym_threshold,
        "normal_threshold": normal_threshold,
        "positives_kept": pos_kept,
        "negatives_kept": neg_kept,
    }


def load_examples(sample_size: int):
    conn = sqlite3.connect(DB_URI, uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        select vc.term, vc.qwen_heard, vc.parakeet_heard,
               v.spoken_auto, coalesce(v.spoken_override, '') as spoken_override
        from vocab_confusions vc
        join vocab v on lower(v.term) = lower(vc.term)
        where v.reviewed = 1
        """
    ).fetchall()

    examples = []
    seen = set()
    for row in rows:
        observed = (row["qwen_heard"] or "").strip() or (row["parakeet_heard"] or "").strip()
        spoken = (row["spoken_override"] or "").strip() or (row["spoken_auto"] or "").strip()
        term = (row["term"] or "").strip()
        if not observed or not spoken or not term:
            continue
        key = (term.lower(), observed.lower())
        if key in seen:
            continue
        seen.add(key)
        examples.append(
            {
                "term": term,
                "observed": observed,
                "spoken": spoken,
                "term_acronymish": looks_acronymish(term),
            }
        )

    rng = random.Random(42)
    rng.shuffle(examples)
    return examples[:sample_size]


def main():
    parser = argparse.ArgumentParser(description="Batch-evaluate phonetic scorer on reviewed confusion pairs.")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--thresholds", default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    args = parser.parse_args()

    sample = load_examples(args.sample_size)

    term_prons = []
    term_seen = set()
    for ex in sample:
        key = ex["term"].lower()
        if key in term_seen:
            continue
        term_seen.add(key)
        term_prons.append({"term": ex["term"], "spoken": ex["spoken"]})

    rng2 = random.Random(99)
    positives = []
    negatives = []
    skipped_positive = []
    skipped_negative = []

    for ex in sample:
        pos = score_pair(ex["observed"], ex["spoken"])
        if pos is None:
            skipped_positive.append(
                {
                    "term": ex["term"],
                    "observed": ex["observed"],
                    "spoken": ex["spoken"],
                }
            )
        else:
            positives.append(
                {
                    "score": pos["score"],
                    "term": ex["term"],
                    "observed": ex["observed"],
                    "spoken": ex["spoken"],
                    "obs_ipa": pos["obs_ipa"],
                    "cand_ipa": pos["cand_ipa"],
                    "term_acronymish": ex["term_acronymish"],
                }
            )

        others = [t for t in term_prons if t["term"].lower() != ex["term"].lower()]
        wrong = rng2.choice(others)
        neg = score_pair(ex["observed"], wrong["spoken"])
        if neg is None:
            skipped_negative.append(
                {
                    "true_term": ex["term"],
                    "wrong_term": wrong["term"],
                    "observed": ex["observed"],
                    "wrong_spoken": wrong["spoken"],
                }
            )
        else:
            negatives.append(
                {
                    "score": neg["score"],
                    "true_term": ex["term"],
                    "wrong_term": wrong["term"],
                    "observed": ex["observed"],
                    "wrong_spoken": wrong["spoken"],
                    "obs_ipa": neg["obs_ipa"],
                    "cand_ipa": neg["cand_ipa"],
                    "true_term_acronymish": ex["term_acronymish"],
                    "wrong_term_acronymish": looks_acronymish(wrong["term"]),
                }
            )

    positives.sort(key=lambda x: x["score"], reverse=True)
    negatives.sort(key=lambda x: x["score"], reverse=True)

    thresholds = []
    for raw_th in args.thresholds.split(","):
        th = float(raw_th)
        thresholds.append(
            {
                "threshold": th,
                "positives_kept": sum(1 for x in positives if x["score"] >= th),
                "negatives_kept": sum(1 for x in negatives if x["score"] >= th),
            }
        )

    result = {
        "sample_size": len(sample),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "skipped_positive_count": len(skipped_positive),
        "skipped_negative_count": len(skipped_negative),
        "positive_summary": summarize(positives) if positives else None,
        "negative_summary": summarize(negatives) if negatives else None,
        "positive_buckets": {
            "acronymish": summarize_bucket([x for x in positives if x["term_acronymish"]]),
            "non_acronymish": summarize_bucket([x for x in positives if not x["term_acronymish"]]),
        },
        "negative_buckets": {
            "true_term_acronymish": summarize_bucket([x for x in negatives if x["true_term_acronymish"]]),
            "true_term_non_acronymish": summarize_bucket([x for x in negatives if not x["true_term_acronymish"]]),
            "wrong_term_acronymish": summarize_bucket([x for x in negatives if x["wrong_term_acronymish"]]),
            "wrong_term_non_acronymish": summarize_bucket([x for x in negatives if not x["wrong_term_acronymish"]]),
            "both_acronymish": summarize_bucket(
                [x for x in negatives if x["true_term_acronymish"] and x["wrong_term_acronymish"]]
            ),
            "neither_acronymish": summarize_bucket(
                [x for x in negatives if not x["true_term_acronymish"] and not x["wrong_term_acronymish"]]
            ),
        },
        "thresholds": thresholds,
        "policy_sweeps": [
            policy_eval(positives, negatives, 0.80, 0.70),
            policy_eval(positives, negatives, 0.82, 0.70),
            policy_eval(positives, negatives, 0.85, 0.70),
            policy_eval(positives, negatives, 0.80, 0.75),
            policy_eval(positives, negatives, 0.82, 0.75),
            policy_eval(positives, negatives, 0.85, 0.75),
        ],
        "top_positive_examples": positives[:12],
        "top_negative_examples": negatives[:20],
        "skipped_positive_examples": skipped_positive[:10],
        "skipped_negative_examples": skipped_negative[:10],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
