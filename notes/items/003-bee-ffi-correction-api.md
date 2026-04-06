# 003: Add correction API to bee-ffi

## Goal

Expose the two-stage judge and correction pipeline through C FFI so the
Swift bee app can call it.

## Depends on

- 001 (bee-correct crate)
- 002 (TwoStageJudge struct)

## API surface

Minimum viable FFI:

```c
// Lifecycle
bee_correct_engine_t* bee_correct_engine_load(
    const char* index_path,     // phonetic index
    const char* events_path     // correction event log (NULL = fresh)
);
void bee_correct_engine_free(bee_correct_engine_t* engine);

// Core correction
bee_correction_result_t* bee_correct_process(
    bee_correct_engine_t* engine,
    const char* transcript_json,    // transcript + aligned words
    uint8_t max_span_words,
    uint16_t shortlist_limit,
    uint16_t verify_limit
);
void bee_correction_result_free(bee_correction_result_t* result);

// Result access
uint32_t bee_correction_result_choice_count(const bee_correction_result_t* result);
const char* bee_correction_result_choice_json(const bee_correction_result_t* result, uint32_t index);
const char* bee_correction_result_best_text(const bee_correction_result_t* result);

// Teaching (user accepted a correction)
void bee_correct_teach(
    bee_correct_engine_t* engine,
    const char* transcript_json,
    const char* teach_json          // which span, which alias chosen
);

// Persistence
void bee_correct_save_events(bee_correct_engine_t* engine, const char* path);
```

## Design decisions

- JSON for complex structured data (transcript, choices) — avoids
  proliferating C struct types
- Opaque pointers for engine and result — Rust owns memory
- `process` returns a decision set (multiple sentence hypotheses)
- `teach` updates the judge model + memory + event log
- Events can be saved/loaded for persistence across sessions

## Swift wrapper

The Swift side wraps these C calls in a `CorrectionEngine` class that:
- Manages lifecycle (load on init, free on deinit)
- Converts Swift types to/from JSON
- Provides async interface if processing is slow

## Validation

- `cargo build -p bee-ffi` compiles with bee-correct dependency
- C header generated (cbindgen or manual)
- Simple C test program can load engine, process transcript, read results
