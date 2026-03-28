import Foundation
import os

@_silgen_name("asr_session_committed_utf16_len")
private func asr_session_committed_utf16_len_ffi(_ session: OpaquePointer) -> UInt

/// Wraps the Rust qwen3-asr-ffi library for streaming transcription.
///
/// Thread-safe: the engine uses an internal mutex. However, a single session
/// must not be used from multiple threads concurrently.
final class TranscriptionService: @unchecked Sendable {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "TranscriptionService"
    )

    private var engine: OpaquePointer?  // *mut AsrEngine
    private let lock = NSLock()

    /// Whether a model is currently loaded and ready.
    var isLoaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return engine != nil
    }

    /// Load a model, downloading from HuggingFace if not cached.
    func loadModel(
        model: STTModelDefinition,
        cacheDir: String,
        updateHandler: (@MainActor @Sendable (ModelLoadUpdate) -> Void)? = nil
    ) async throws {
        // Unload previous model
        unloadModel()

        await updateHandler?(.downloading(progress: 0))

        let format = model.format

        // The Rust FFI handles download + cache. This is a blocking call,
        // so run it off the main thread.
        let loadedEngine: OpaquePointer = try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                var err: UnsafeMutablePointer<CChar>?
                let ptr: OpaquePointer?

                switch format {
                case .safetensors:
                    ptr = asr_engine_from_pretrained(model.repoID, cacheDir, &err)

                case .gguf(let ggufRepoID, let ggufFilename, let baseRepoID):
                    ptr = asr_engine_from_gguf(baseRepoID, ggufRepoID, ggufFilename, cacheDir, &err)
                }

                if let ptr {
                    continuation.resume(returning: ptr)
                } else {
                    let message = err.flatMap { String(cString: $0, encoding: .utf8) } ?? "unknown error"
                    err.flatMap { asr_string_free($0) }
                    continuation.resume(throwing: TranscriptionError.loadFailed(message))
                }
            }
        }

        await updateHandler?(.initializing)

        lock.withLock { engine = loadedEngine }

        Self.logger.info("Model loaded: \(model.displayName, privacy: .public)")
    }

    /// Load a model from a local directory (no download).
    func loadModelFromDirectory(_ path: String) throws {
        unloadModel()

        var err: UnsafeMutablePointer<CChar>?
        guard let ptr = asr_engine_load(path, &err) else {
            let message = err.flatMap { String(cString: $0, encoding: .utf8) } ?? "unknown error"
            err.flatMap { asr_string_free($0) }
            throw TranscriptionError.loadFailed(message)
        }

        lock.lock()
        engine = ptr
        lock.unlock()
    }

    func unloadModel() {
        lock.lock()
        let e = engine
        engine = nil
        lock.unlock()

        if let e {
            asr_engine_free(e)
        }
    }

    deinit {
        if let engine {
            asr_engine_free(engine)
        }
    }

    // MARK: - Single-shot Transcription

    /// Transcribe raw 16kHz f32 samples in one shot (non-streaming).
    /// Returns the transcript text, or nil on error.
    func transcribeSamples(_ samples: [Float]) -> String? {
        lock.lock()
        guard let engine else {
            lock.unlock()
            return nil
        }
        lock.unlock()

        var err: UnsafeMutablePointer<CChar>?
        let result = samples.withUnsafeBufferPointer { buf in
            asr_engine_transcribe_samples(engine, buf.baseAddress, buf.count, &err)
        }

        if let err {
            let msg = String(cString: err, encoding: .utf8) ?? "unknown"
            asr_string_free(err)
            Self.logger.error("transcribeSamples error: \(msg, privacy: .public)")
            return nil
        }

        guard let result else { return nil }
        let text = String(cString: result)
        asr_string_free(result)
        return text
    }

    // MARK: - Streaming Session

    /// Create a new streaming session.
    /// Returns nil if no model is loaded.
    /// `language`: Qwen3 language name (e.g. "english", "french") or nil for auto-detect.
    /// `prompt`: Vocabulary hint text for prefix conditioning, or nil.
    func createSession(chunkSizeSec: Float = 1.0, sessionDurationSec: Float = 120.0, language: String? = nil, prompt: String? = nil) -> StreamingSession? {
        lock.lock()
        guard let engine else {
            lock.unlock()
            return nil
        }
        lock.unlock()

        // Helper to create session with resolved C string pointers.
        func makeSession(langPtr: UnsafePointer<CChar>?, promptPtr: UnsafePointer<CChar>?) -> StreamingSession? {
            let opts = AsrSessionOptions(
                chunk_size_sec: chunkSizeSec,
                session_duration_sec: sessionDurationSec,
                language: langPtr,
                prompt: promptPtr
            )
            guard let session = asr_session_create(engine, opts) else { return nil }
            return StreamingSession(ptr: session)
        }

        // Nest withCString calls so pointers stay valid.
        switch (language, prompt) {
        case let (lang?, prompt?):
            return lang.withCString { lp in prompt.withCString { pp in makeSession(langPtr: lp, promptPtr: pp) } }
        case let (lang?, nil):
            return lang.withCString { lp in makeSession(langPtr: lp, promptPtr: nil) }
        case let (nil, prompt?):
            return prompt.withCString { pp in makeSession(langPtr: nil, promptPtr: pp) }
        case (nil, nil):
            return makeSession(langPtr: nil, promptPtr: nil)
        }
    }

    /// Feed 16kHz mono f32 audio into a streaming session.
    /// Returns transcript + committed prefix metadata when a chunk boundary is crossed.
    func feed(session: StreamingSession, samples: [Float]) -> StreamingTranscriptUpdate? {
        var err: UnsafeMutablePointer<CChar>?
        let result = samples.withUnsafeBufferPointer { buf in
            asr_session_feed(session.ptr, buf.baseAddress, buf.count, &err)
        }

        if let err {
            let msg = String(cString: err, encoding: .utf8) ?? "unknown"
            asr_string_free(err)
            Self.logger.error("feed error: \(msg, privacy: .public)")
            return nil
        }

        guard let result else { return nil }
        let text = String(cString: result)
        asr_string_free(result)

        let committedUTF16 = Int(asr_session_committed_utf16_len_ffi(session.ptr))
        let clampedCommittedUTF16 = min(max(0, committedUTF16), (text as NSString).length)

        return StreamingTranscriptUpdate(
            text: text,
            committedUTF16Count: clampedCommittedUTF16
        )
    }

    /// Finalize a streaming session and return the complete transcript.
    func finish(session: StreamingSession) -> String? {
        var err: UnsafeMutablePointer<CChar>?
        let result = asr_session_finish(session.ptr, &err)

        if let err {
            let msg = String(cString: err, encoding: .utf8) ?? "unknown"
            asr_string_free(err)
            Self.logger.error("finish error: \(msg, privacy: .public)")
            return nil
        }

        guard let result else { return nil }
        let text = String(cString: result)
        asr_string_free(result)
        return text
    }
}

/// Opaque streaming session handle wrapping the Rust AsrSession.
final class StreamingSession: @unchecked Sendable {
    let ptr: OpaquePointer  // *mut AsrSession

    init(ptr: OpaquePointer) {
        self.ptr = ptr
    }

    deinit {
        asr_session_free(ptr)
    }
}

enum TranscriptionError: LocalizedError {
    case modelNotLoaded
    case loadFailed(String)
    case emptyAudio
    case emptyResult

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "STT model not loaded"
        case .loadFailed(let message):
            return "Model load failed: \(message)"
        case .emptyAudio:
            return "No audio recorded"
        case .emptyResult:
            return "No speech detected"
        }
    }
}

enum ModelLoadUpdate: Sendable {
    case downloading(progress: Double)
    case initializing
}

struct StreamingTranscriptUpdate: Sendable {
    let text: String
    let committedUTF16Count: Int
}
