#!/usr/bin/env python3
"""
Generate term-focused sentences for TTS→ASR pipeline.

For each hard term, generates many sentence contexts using the term
in different positions and surrounding contexts.
"""

import json
import random

random.seed(42)

TEMPLATES = [
    # Term at start
    "{term} is what we need for this.",
    "{term} handles that already.",
    "{term} should work here.",
    "{term} needs to be updated.",
    "{term} is broken, can you fix it?",
    "{term} was working yesterday but not today.",
    # Term in middle
    "I think {term} is the right choice here.",
    "Can you check if {term} is installed?",
    "We should switch to {term} for this.",
    "The issue is in {term}, not in our code.",
    "Have you tried using {term} instead?",
    "Let me look at {term} real quick.",
    "After upgrading {term} everything broke.",
    "I'm having trouble with {term} again.",
    "We need to add {term} to the dependencies.",
    "Make sure {term} is in the config.",
    "The problem is that {term} doesn't support this yet.",
    "I just pushed a fix for {term}.",
    "Can you review the {term} changes?",
    "Let's revert the {term} update.",
    # Term at end
    "The bug is definitely in {term}.",
    "We're currently using {term}.",
    "I just upgraded to the latest {term}.",
    "This whole thing depends on {term}.",
    "Please take a look at {term}.",
    "The tests are failing because of {term}.",
    "I wrote a wrapper around {term}.",
    "Everything works except {term}.",
    # Technical contexts
    "Add {term} to your Cargo.toml and run cargo build.",
    "The {term} crate provides exactly what we need.",
    "Check the {term} documentation for details.",
    "I'm going to refactor the {term} integration.",
    "We replaced the old approach with {term}.",
    "The {term} configuration needs to change.",
    "Run the {term} tests before merging.",
    "The performance of {term} is much better now.",
    # With other terms mixed in
    "Use {term} with serde for serialization.",
    "The {term} setup works on both Linux and macOS.",
    "Deploy {term} to the staging server first.",
    "I committed the {term} changes to main.",
    "The CI is failing because {term} is misconfigured.",
]

# Spoken form overrides for TTS
SPOKEN_OVERRIDES = {}
with open("data/pronunciations.jsonl") as f:
    for line in f:
        d = json.loads(line)
        SPOKEN_OVERRIDES[d["term"].lower()] = d.get("note", d["term"])

def spoken_form(term):
    """Get how a human would say this term."""
    low = term.lower()
    if low in SPOKEN_OVERRIDES:
        note = SPOKEN_OVERRIDES[low]
        # Use the "like X" form if available
        if note.startswith("like "):
            return note[5:]
        return note
    # Default: just the term with hyphens/underscores as spaces
    return term.replace("-", " ").replace("_", " ")

# Load hard terms
hard_terms = []
with open("data/real_corruption_map.jsonl") as f:
    for line in f:
        d = json.loads(line)
        hard_terms.append(d["term"])

# Add terms from pronunciations that aren't in confusions
seen = {t.lower() for t in hard_terms}
with open("data/pronunciations.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d["term"].lower() not in seen:
            hard_terms.append(d["term"])
            seen.add(d["term"].lower())

# Take top 50
hard_terms = hard_terms[:50]
print(f"Generating sentences for {len(hard_terms)} terms", flush=True)

sentences = []
for term in hard_terms:
    spoken = spoken_form(term)
    # Generate 40 contexts per term
    selected = random.sample(TEMPLATES, min(40, len(TEMPLATES)))
    for template in selected:
        text = template.format(term=term)
        spoken_text = template.format(term=spoken)
        sentences.append({
            "text": text,
            "spoken": spoken_text,
            "vocab_terms": [term],
        })

random.shuffle(sentences)
print(f"Generated {len(sentences)} sentences")

with open("data/focused_sentences.jsonl", "w") as f:
    for s in sentences:
        f.write(json.dumps(s) + "\n")

print(f"Wrote to data/focused_sentences.jsonl")
