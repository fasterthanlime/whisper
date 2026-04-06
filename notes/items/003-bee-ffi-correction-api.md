# 003: Add correction API to bee-ffi

## Goal

Expose the two-stage judge and correction pipeline through C FFI so the
Swift bee app can call it.

## Depends on

- 000 (bee-types for shared data model)
- 001 (bee-correct crate)
- 002 (TwoStageJudge struct)

## API surface

### Lifecycle

```c
bee_correct_engine_t* bee_correct_engine_load(
    const char* index_path,     // phonetic index
    const char* weights_path,   // judge weights (NULL = seed)
    const char* memory_path,    // term memory (NULL = fresh)
    const char* events_path     // event log path for append
);
void bee_correct_engine_free(bee_correct_engine_t* engine);
```

### Core correction

```c
// Returns a result with a stable session_id for teaching
bee_correction_result_t* bee_correct_process(
    bee_correct_engine_t* engine,
    const char* transcript_json,    // { text, words, app_id? }
    uint8_t max_span_words,
    uint16_t shortlist_limit,
    uint16_t verify_limit
);
void bee_correction_result_free(bee_correction_result_t* result);

// Result access
const char* bee_correction_result_session_id(const bee_correction_result_t* result);
uint32_t bee_correction_result_edit_count(const bee_correction_result_t* result);
const char* bee_correction_result_json(const bee_correction_result_t* result);
const char* bee_correction_result_best_text(const bee_correction_result_t* result);
```

### Teaching (stable IDs, not transcript re-processing)

```c
// Teach against the exact choices the engine surfaced
void bee_correct_teach(
    bee_correct_engine_t* engine,
    const char* session_id,         // from process result
    const char* teach_json          // { edits: [{edit_id, accepted}], user_edits: [...] }
);
```

### Persistence

```c
void bee_correct_save(bee_correct_engine_t* engine);  // saves weights + memory + flushes events
```

## Teaching event model

The teach payload references **stable IDs** from the process result,
not raw transcript positions. This avoids drift if the transcript is
reprocessed or engine state changed between process and teach.

```json
{
  "session_id": "abc123",
  "schema_version": 1,
  "edits": [
    { "edit_id": "e1", "resolution": "accepted" },
    { "edit_id": "e2", "resolution": "rejected" }
  ],
  "user_edits": [
    { "span_start": 15, "span_end": 22, "original": "foo bar", "replacement": "foobar" }
  ]
}
```

Teaching signals by resolution:
- **accepted**: positive for gate (open was correct) + positive for ranker (right candidate)
- **rejected**: negative for gate (should not have opened)
- **user_edit**: vocabulary expansion signal (separate from judge training)

## Design decisions

- JSON for complex structured data — avoids proliferating C struct types
- Opaque pointers for engine and result — Rust owns memory
- Session IDs tie process results to teaching events
- Schema versioned from day one
- Thread safety: engine is NOT thread-safe. Caller serializes access.
  Document this clearly.

## Error handling

```c
// After any call, check for error
const char* bee_correct_last_error(bee_correct_engine_t* engine);
// Returns NULL if no error, otherwise a string that's valid until next call
```

## Swift wrapper

```swift
class CorrectionEngine {
    private var handle: OpaquePointer

    func process(transcript: TranscriptJSON) -> CorrectionResult
    func teach(sessionId: String, edits: [EditResolution])
    func save()

    deinit { bee_correct_engine_free(handle) }
}
```

## Validation

- `cargo build -p bee-ffi` compiles with bee-correct dependency
- C header generated (cbindgen or manual)
- Simple test: load engine, process transcript, read results, teach
