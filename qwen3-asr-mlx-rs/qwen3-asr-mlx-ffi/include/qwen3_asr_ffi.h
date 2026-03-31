#ifndef QWEN3_ASR_FFI_H
#define QWEN3_ASR_FFI_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles */
typedef struct AsrEngine AsrEngine;
typedef struct AsrSession AsrSession;

/* Options for creating a streaming session */
typedef struct {
    float chunk_size_sec;           /* e.g. 0.5  */
    float session_duration_sec;     /* e.g. 10.0 — auto-rotates after this */
    const char *language;           /* e.g. "english", "french", or NULL for auto-detect */
    const char *prompt;             /* vocabulary hint text, or NULL for none */
    unsigned int unfixed_chunk_num;       /* 0 = use default (2) */
    unsigned int unfixed_token_num;       /* 0 = use default (12) */
    unsigned int max_new_tokens_streaming; /* 0 = use default (32) */
    unsigned int max_new_tokens_final;    /* 0 = use default (512) */
} AsrSessionOptions;

/*
 * Load a model from disk. Returns NULL on error; if out_err is non-NULL,
 * *out_err is set to a message string (free with asr_string_free).
 */
AsrEngine *asr_engine_load(const char *model_dir, char **out_err);

/*
 * Download a model from HuggingFace (if not cached) and load it.
 * model_id: e.g. "Qwen/Qwen3-ASR-0.6B"
 * cache_dir: local directory for caching model files.
 * Returns NULL on error (check *out_err). Free with asr_engine_free.
 */
AsrEngine *asr_engine_from_pretrained(const char *model_id,
                                      const char *cache_dir,
                                      char **out_err);

/*
 * Download a GGUF-quantized model from HuggingFace and load it.
 * base_repo_id: full-precision repo for config+tokenizer (e.g. "Qwen/Qwen3-ASR-1.7B")
 * gguf_repo_id: repo hosting GGUF files (e.g. "Alkd/qwen3-asr-gguf")
 * gguf_filename: specific file (e.g. "qwen3_asr_1.7b_q4_k.gguf")
 * cache_dir: local directory for caching model files.
 * Returns NULL on error (check *out_err). Free with asr_engine_free.
 */
AsrEngine *asr_engine_from_gguf(const char *base_repo_id,
                                 const char *gguf_repo_id,
                                 const char *gguf_filename,
                                 const char *cache_dir,
                                 char **out_err);

/*
 * Single-shot transcription from 16 kHz mono float32 samples.
 * Returns a freshly-allocated string (free with asr_string_free).
 * Returns NULL on error (check *out_err).
 */
char *asr_engine_transcribe_samples(const AsrEngine *engine,
                                     const float *samples,
                                     size_t num_samples,
                                     char **out_err);

/* Free an engine handle. NULL-safe. */
void asr_engine_free(AsrEngine *engine);

/*
 * Create a streaming session attached to an engine.
 * Free with asr_session_free when done.
 */
AsrSession *asr_session_create(const AsrEngine *engine, AsrSessionOptions opts);

/*
 * Feed 16 kHz mono float32 samples.
 *
 * Returns a freshly-allocated string with the current full transcript when a
 * chunk boundary is crossed. Returns NULL when still buffering.
 * On error, returns NULL and sets *out_err.
 * Caller must free the returned string with asr_string_free.
 */
char *asr_session_feed(AsrSession *session,
                       const float *samples,
                       size_t num_samples,
                       char **out_err);

/*
 * Feed finalization-time samples (stop path).
 *
 * Same semantics as asr_session_feed, but avoids dropping low-energy chunks
 * during stop/finalize so tail words are not lost.
 */
char *asr_session_feed_finalizing(AsrSession *session,
                                  const float *samples,
                                  size_t num_samples,
                                  char **out_err);

/*
 * Return committed transcript prefix length in UTF-16 code units.
 * This excludes the current pending/uncommitted tail.
 */
size_t asr_session_committed_utf16_len(const AsrSession *session);

/*
 * Return the most recently detected language for this session.
 * Returns a freshly-allocated string (free with asr_string_free), or NULL.
 */
char *asr_session_last_language(const AsrSession *session);

/*
 * Force or clear language for an active session.
 * Pass language=NULL (or empty) to restore auto-detection.
 * Returns true on success, false on error (check *out_err).
 */
bool asr_session_set_language(AsrSession *session, const char *language, char **out_err);

/*
 * Drain and return structured ASR debug events as a JSON array string.
 * The returned string is always valid JSON (typically "[]").
 * Caller must free with asr_string_free.
 */
char *asr_session_take_debug_events_json(AsrSession *session);

/*
 * Finalize the session and return the complete transcript.
 * Caller must free the returned string with asr_string_free.
 */
char *asr_session_finish(AsrSession *session, char **out_err);

/* Free a session handle. NULL-safe. */
void asr_session_free(AsrSession *session);

/* Free a string returned by any asr_* function. NULL-safe. */
void asr_string_free(char *s);

#ifdef __cplusplus
}
#endif

#endif /* QWEN3_ASR_FFI_H */
