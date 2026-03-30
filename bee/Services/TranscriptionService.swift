import Foundation

/// FFI wrapper around the qwen3-asr Rust engine.
/// Manages model loading and streaming session lifecycle.
final class TranscriptionService: @unchecked Sendable {
    private let lock = NSLock()
    private var engine: OpaquePointer? // AsrEngine*

    // MARK: - Model Loading

    func loadModel(repoID: String, cacheDir: String) async throws {
        // TODO: call asr_engine_from_pretrained or asr_engine_from_gguf
    }

    func unloadModel() {
        lock.withLock {
            if let engine {
                asr_engine_free(engine)
                self.engine = nil
            }
        }
    }

    // MARK: - Session API

    func createSession(
        chunkSizeSec: Float = 1.0,
        language: String? = nil
    ) -> OpaquePointer? {
        // TODO: asr_session_create
        nil
    }

    // h[impl asr.streaming]
    func feed(session: OpaquePointer, samples: [Float]) -> StreamingUpdate? {
        // TODO: asr_session_feed
        nil
    }

    // h[impl asr.finalize]
    func feedFinalizing(session: OpaquePointer, samples: [Float]) -> StreamingUpdate? {
        // TODO: asr_session_feed_finalizing
        nil
    }

    func finish(session: OpaquePointer) -> String? {
        // TODO: asr_session_finish
        nil
    }

    // h[impl lang.lock-during-streaming]
    func setLanguage(session: OpaquePointer, language: String?) -> Bool {
        // TODO: asr_session_set_language
        false
    }

    deinit {
        unloadModel()
    }
}
