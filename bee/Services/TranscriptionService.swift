import Foundation
import os

@_silgen_name("asr_session_committed_utf16_len")
private func asr_session_committed_utf16_len_ffi(_ session: OpaquePointer) -> UInt

/// Wraps the Rust qwen3-asr-ffi library for streaming transcription.
final class TranscriptionService: @unchecked Sendable {
    private static let logger = Logger(
        subsystem: "fasterthanlime.bee",
        category: "TranscriptionService"
    )

    private var engine: OpaquePointer?
    private let lock = NSLock()

    var isLoaded: Bool {
        lock.withLock { engine != nil }
    }

    // MARK: - Model Loading

    func loadModel(model: STTModelDefinition, cacheDir: String) async throws {
        unloadModel()

        let format = model.format

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

        lock.withLock { engine = loadedEngine }
        Self.logger.info("Model loaded: \(model.displayName, privacy: .public)")
    }

    func unloadModel() {
        let e = lock.withLock { () -> OpaquePointer? in
            let e = engine
            engine = nil
            return e
        }
        if let e { asr_engine_free(e) }
    }

    // MARK: - Streaming Session

    func createSession(chunkSizeSec: Float = 1.0, language: String? = nil) -> StreamingSession? {
        lock.lock()
        guard let engine else {
            lock.unlock()
            return nil
        }
        lock.unlock()

        func makeSession(langPtr: UnsafePointer<CChar>?) -> StreamingSession? {
            let opts = AsrSessionOptions(
                chunk_size_sec: chunkSizeSec,
                session_duration_sec: 120.0,
                language: langPtr,
                prompt: nil
            )
            guard let session = asr_session_create(engine, opts) else { return nil }
            return StreamingSession(ptr: session)
        }

        if let language {
            return language.withCString { makeSession(langPtr: $0) }
        }
        return makeSession(langPtr: nil)
    }

    func feed(session: StreamingSession, samples: [Float]) -> StreamingUpdate? {
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
        let clamped = min(max(0, committedUTF16), (text as NSString).length)

        return StreamingUpdate(text: text, committedUTF16Count: clamped, detectedLanguage: nil)
    }

    func feedFinalizing(session: StreamingSession, samples: [Float]) -> StreamingUpdate? {
        var err: UnsafeMutablePointer<CChar>?
        let result = samples.withUnsafeBufferPointer { buf in
            asr_session_feed_finalizing(session.ptr, buf.baseAddress, buf.count, &err)
        }

        if let err {
            let msg = String(cString: err, encoding: .utf8) ?? "unknown"
            asr_string_free(err)
            Self.logger.error("feedFinalizing error: \(msg, privacy: .public)")
            return nil
        }

        guard let result else { return nil }
        let text = String(cString: result)
        asr_string_free(result)

        let committedUTF16 = Int(asr_session_committed_utf16_len_ffi(session.ptr))
        let clamped = min(max(0, committedUTF16), (text as NSString).length)

        return StreamingUpdate(text: text, committedUTF16Count: clamped, detectedLanguage: nil)
    }

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

    func setLanguage(session: StreamingSession, language: String?) -> Bool {
        var err: UnsafeMutablePointer<CChar>?
        let success: Bool = {
            guard let language else {
                return asr_session_set_language(session.ptr, nil, &err)
            }
            return language.withCString { asr_session_set_language(session.ptr, $0, &err) }
        }()

        if let err {
            let msg = String(cString: err, encoding: .utf8) ?? "unknown"
            asr_string_free(err)
            Self.logger.error("setLanguage error: \(msg, privacy: .public)")
            return false
        }
        return success
    }

    deinit {
        if let engine { asr_engine_free(engine) }
    }
}

final class StreamingSession: @unchecked Sendable {
    let ptr: OpaquePointer
    init(ptr: OpaquePointer) { self.ptr = ptr }
    deinit { asr_session_free(ptr) }
}

enum TranscriptionError: LocalizedError {
    case loadFailed(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let msg): "Model load failed: \(msg)"
        }
    }
}
