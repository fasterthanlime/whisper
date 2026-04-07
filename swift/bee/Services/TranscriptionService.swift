import Foundation
import os

/// Wraps the Rust bee-ffi service via vox-ffi for streaming transcription.
final class TranscriptionService: @unchecked Sendable {
    private static let logger = Logger(
        subsystem: "fasterthanlime.bee",
        category: "TranscriptionService"
    )

    private let engine: BeeEngine
    private let lock = NSLock()
    private var loaded = false

    init(engine: BeeEngine) {
        self.engine = engine
    }

    var isLoaded: Bool {
        lock.withLock { loaded }
    }

    private func client() async throws -> BeeClient {
        guard let client = await engine.client else {
            throw BeeError.engineNotLoaded
        }
        return client
    }

    // MARK: - Model Loading

    func loadModel(cacheDir: String) async throws {
        let client = try await client()
        let _ = try await client.loadEngine(cacheDir: cacheDir).get()
        lock.withLock { loaded = true }
        Self.logger.info("Model loaded via vox-ffi")
    }

    func unloadModel() {
        lock.withLock { loaded = false }
    }

    // MARK: - Streaming Session

    struct SessionConfig: Sendable {
        var language: String? = nil
    }

    func createSession(_ config: SessionConfig = SessionConfig()) async -> StreamingSession? {
        do {
            let client = try await client()
            let sessionId = try await client.createSession(language: config.language ?? "").get()
            return StreamingSession(id: sessionId)
        } catch {
            Self.logger.error("createSession failed: \(error)")
            return nil
        }
    }

    func feed(session: StreamingSession, samples: [Float]) async -> StreamingUpdate? {
        do {
            let client = try await client()
            let result = try await client.feed(sessionId: session.id, samples: samples).get()
            guard let result else { return nil }
            return StreamingUpdate(
                text: result.text,
                committedUTF16Count: Int(result.committedUtf16Len),
                detectedLanguage: nil,
                alignments: result.alignments,
                debugJSON: nil
            )
        } catch {
            Self.logger.error("feed error: \(error)")
            return nil
        }
    }

    func feedFinalizing(session: StreamingSession, samples: [Float]) async -> StreamingUpdate? {
        return await feed(session: session, samples: samples)
    }

    func finish(session: StreamingSession) async -> String? {
        do {
            let client = try await client()
            let text = try await client.finishSession(sessionId: session.id).get()
            return text.isEmpty ? nil : text
        } catch {
            Self.logger.error("finish error: \(error)")
            return nil
        }
    }

    func setLanguage(session: StreamingSession, language: String?) async -> Bool {
        guard let language, !language.isEmpty else { return true }
        do {
            let client = try await client()
            let _ = try await client.setLanguage(sessionId: session.id, language: language).get()
            return true
        } catch {
            Self.logger.error("setLanguage error: \(error)")
            return false
        }
    }

    func getStats() async -> EngineStats? {
        do {
            let client = try await client()
            return try await client.getStats()
        } catch {
            Self.logger.error("getStats error: \(error)")
            return nil
        }
    }

    func transcribeSamples(_ samples: [Float]) async -> String? {
        do {
            let client = try await client()
            let result = try await client.transcribeSamples(samples: samples).get()
            return result.isEmpty ? nil : result
        } catch {
            Self.logger.error("transcribeSamples error: \(error)")
            return nil
        }
    }
}

final class StreamingSession: @unchecked Sendable {
    let id: String
    init(id: String) { self.id = id }
}
