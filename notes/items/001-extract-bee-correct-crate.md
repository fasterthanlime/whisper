# 001: Extract bee-correct crate from beeml

## Goal

Create `rust/bee-correct/` containing the core correction logic, so that
`bee-ffi` can depend on it without pulling in Vox RPC, tokio, or WebSocket
machinery.

## What moves to bee-correct

| File | From | Notes |
|------|------|-------|
| `judge.rs` | `beeml/src/judge.rs` | Entire file (OnlineJudge, features, scoring, training, gate/ranker) |
| `sparse_ftrl.rs` | `beeml/src/sparse_ftrl.rs` | Entire file (FTRL-Proximal optimizer) |
| `g2p.rs` | `beeml/src/g2p.rs` | Entire file (espeak-ng G2P cache) |
| `decision_set.rs` | Extract from `beeml/src/main.rs` | Decision set building: `build_rapid_fire_decision_set`, `collect_admitted_edits`, `dedupe_edit_candidates`, `build_conflict_components`, `build_component`, `compose_sentence_hypotheses`, `enumerate_sentence_hypotheses`, `beam_sentence_hypotheses`, `build_sentence_hypothesis`, `apply_atomic_edits`, `prune_sentence_hypotheses`, helper structs (`EditCandidate`, `ComponentHypothesis`, `SentenceHypothesis`, `Component`) |

## What stays in beeml

- `main.rs`: RPC handlers (`run_probe`, `run_offline_judge_eval`), WebSocket server, eval harness
- `rpc.rs`: Vox RPC trait definitions, debug/trace types, eval types
- `bin/`: codegen and utility binaries
- All eval functions (k-fold, two-stage grid, training traces)

## Dependencies for bee-correct

```toml
bee-phonetic = { path = "../bee-phonetic" }
bee-transcribe = { path = "../bee-transcribe" }
anyhow = { workspace = true }
espeak-ng = { version = "0.1.0", features = ["bundled-data-en"] }
fnv = "1.0"
serde = { workspace = true }
serde_json = { workspace = true }
tracing = "0.1"
```

## beeml becomes

```toml
[dependencies]
bee-correct = { path = "../bee-correct" }
# ... plus vox, tokio, rayon, etc. for RPC/eval
```

beeml re-exports from bee-correct as needed. All `use beeml::judge::*`
becomes `use bee_correct::judge::*` (or beeml re-exports them).

## Validation

- `cargo build -p bee-correct` compiles
- `cargo build -p beeml` compiles (uses bee-correct)
- `cargo run -p beeml -- --offline-eval` produces same results
- `bee-ffi` can add `bee-correct` as a dependency without pulling in vox/tokio
