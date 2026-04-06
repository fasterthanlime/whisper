# 001: Extract bee-correct crate from beeml

## Goal

Create `rust/bee-correct/` containing the core correction logic, so that
`bee-ffi` can depend on it without pulling in Vox RPC, tokio, or WebSocket
machinery.

## Depends on

- 000 (bee-types crate for shared data model)

## Key boundary rule

bee-correct must NOT depend on bee-transcribe. It depends on bee-types
for transcript/span structs and bee-phonetic for retrieval/scoring.

## What moves to bee-correct

| File | From | Notes |
|------|------|-------|
| `judge.rs` | `beeml/src/judge.rs` | OnlineJudge, features, scoring, training, gate/ranker |
| `sparse_ftrl.rs` | `beeml/src/sparse_ftrl.rs` | FTRL-Proximal optimizer |
| `g2p.rs` | `beeml/src/g2p.rs` | espeak-ng G2P cache |
| `decision_set.rs` | Extract from `beeml/src/main.rs` | Decision set building pipeline |
| `types.rs` | New | Re-export from bee-types + bee-correct-specific types |

## What stays in beeml

- `main.rs`: RPC handlers, WebSocket server, eval harness
- `rpc.rs`: Vox RPC trait definitions, debug/trace types
- `bin/`: codegen and utility binaries
- All eval functions (k-fold, two-stage grid, training traces)

## Dependencies for bee-correct

```toml
bee-types = { path = "../bee-types" }
bee-phonetic = { path = "../bee-phonetic" }
anyhow = { workspace = true }
espeak-ng = { version = "0.1.0", features = ["bundled-data-en"] }
fnv = "1.0"
serde = { workspace = true }
serde_json = { workspace = true }
tracing = "0.1"
```

NO bee-transcribe. NO vox. NO tokio.

## beeml becomes

```toml
[dependencies]
bee-correct = { path = "../bee-correct" }
bee-types = { path = "../bee-types" }
# ... plus vox, tokio, rayon, etc. for RPC/eval
```

## Parity testing

Before and after extraction, snapshot these for fixed test cases:
- Best text output
- Choice count in decision set
- Top 3 choices with scores
- Gate and ranker probabilities for specific spans

`cargo run -p beeml -- --offline-eval` must produce identical results.

## Validation

- `cargo build -p bee-correct` compiles
- `cargo build -p beeml` compiles (uses bee-correct)
- bee-correct does NOT depend on bee-transcribe (verify in Cargo.toml)
- Offline eval produces identical results to pre-extraction
- Parity snapshots match
