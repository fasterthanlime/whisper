import Foundation
import VoxRuntime

/// Manages the vox-ffi connection to the Rust bee-ffi dylib.
/// Provides typed access to the Bee service (model downloads, engine loading, transcription).
actor BeeEngine {
    private var rust: FfiDynamicLibrary?
    private var client: BeeClient?
    private var sessionHandle: SessionHandle?
    private var driverTask: Task<Void, Error>?

    /// Path to the libbee_ffi.dylib — checks app bundle first, then build output.
    private static func libraryURL() -> URL {
        // In the installed app: Contents/Frameworks/libbee_ffi.dylib
        if let frameworksPath = Bundle.main.privateFrameworksPath {
            let bundled = URL(fileURLWithPath: frameworksPath)
                .appendingPathComponent("libbee_ffi.dylib")
            if FileManager.default.fileExists(atPath: bundled.path) {
                return bundled
            }
        }

        // During development: target/release/libbee_ffi.dylib relative to project root
        let executableURL = Bundle.main.executableURL!
        let projectRoot = executableURL
            .deletingLastPathComponent()  // Contents/MacOS
            .deletingLastPathComponent()  // Contents
            .deletingLastPathComponent()  // *.app
            .deletingLastPathComponent()  // build dir
        let devPath = projectRoot.appendingPathComponent("target/release/libbee_ffi.dylib")
        if FileManager.default.fileExists(atPath: devPath.path) {
            return devPath
        }

        // Fallback
        return URL(fileURLWithPath: Bundle.main.privateFrameworksPath!)
            .appendingPathComponent("libbee_ffi.dylib")
    }

    /// Connect to the Rust bee-ffi service via vox-ffi.
    func connect() async throws {
        let libURL = Self.libraryURL()
        beeLog("BEE-ENGINE: loading dylib from \(libURL.path)")

        // Tell Rust where to log (must be set before dlopen triggers #[ctor])
        let logPath = FileManager.default.containerURL(
            forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee"
        )?.appendingPathComponent("bee.log").path ?? "/tmp/bee.log"
        setenv("BEE_FFI_LOG_PATH", logPath, 1)

        let rust = try FfiDynamicLibrary(path: libURL)
        self.rust = rust
        beeLog("BEE-ENGINE: dylib loaded, loading vtable")

        let endpoint = FfiEndpoint()
        let vtable = try rust.loadVtable(symbol: "bee_ffi_v1_vtable")
        beeLog("BEE-ENGINE: vtable loaded, connecting")

        let connector = try endpoint.connector(peer: vtable)
        beeLog("BEE-ENGINE: connector created, initiating session")

        let session = try await VoxRuntime.Session.initiator(
            connector,
            dispatcher: NoopBeeDispatcher(),
            resumable: false
        )
        beeLog("BEE-ENGINE: session established")

        self.sessionHandle = session.handle
        self.driverTask = Task {
            do {
                try await session.run()
                beeLog("BEE-ENGINE: driver task completed normally")
            } catch {
                beeLog("BEE-ENGINE: driver task failed: \(error)")
            }
        }

        self.client = BeeClient(connection: session.connection)
        beeLog("BEE-ENGINE: client ready")
    }

    /// Get the full manifest of required model repos from Rust.
    func requiredDownloads() async throws -> [RepoDownload] {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.requiredDownloads()
    }

    /// Load the ASR engine from the given cache directory.
    func loadEngine(cacheDir: String) async throws {
        guard let client else { throw BeeEngineError.notConnected }
        let result = try await client.loadEngine(cacheDir: cacheDir)
        if !result.isEmpty {
            throw BeeEngineError.loadFailed(result)
        }
    }

    /// Create a transcription session.
    func createSession(language: String = "") async throws -> String {
        guard let client else { throw BeeEngineError.notConnected }
        let sessionId = try await client.createSession(language: language)
        if sessionId.isEmpty {
            throw BeeEngineError.noEngine
        }
        return sessionId
    }

    /// Feed audio samples to a session.
    func feed(sessionId: String, samples: [Float]) async throws -> FeedResult {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.feed(sessionId: sessionId, samples: samples)
    }

    /// Finalize a session and get the final transcription.
    func finishSession(sessionId: String) async throws -> String {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.finishSession(sessionId: sessionId)
    }

    /// Set the language for a session.
    func setLanguage(sessionId: String, language: String) async throws -> Bool {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.setLanguage(sessionId: sessionId, language: language)
    }

    /// Single-shot transcription of raw 16kHz f32 samples.
    func transcribeSamples(samples: [Float]) async throws -> String {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.transcribeSamples(samples: samples)
    }

    /// Get engine resource usage stats.
    func getStats() async throws -> EngineStats {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.getStats()
    }

    /// Load the correction engine.
    func correctLoad(datasetDir: String, eventsPath: String, gateThreshold: Float, rankerThreshold: Float) async throws {
        guard let client else { throw BeeEngineError.notConnected }
        let result = try await client.correctLoad(datasetDir: datasetDir, eventsPath: eventsPath, gateThreshold: gateThreshold, rankerThreshold: rankerThreshold)
        if !result.isEmpty {
            throw BeeEngineError.loadFailed(result)
        }
    }

    /// Run correction on text.
    func correctProcess(text: String, appId: String) async throws -> CorrectionOutput {
        guard let client else { throw BeeEngineError.notConnected }
        return try await client.correctProcess(text: text, appId: appId)
    }

    /// Teach the correction engine from user resolutions.
    func correctTeach(sessionId: String, resolutions: [EditResolution]) async throws {
        guard let client else { throw BeeEngineError.notConnected }
        let _ = try await client.correctTeach(sessionId: sessionId, resolutions: resolutions)
    }

    /// Save correction engine state.
    func correctSave() async throws {
        guard let client else { throw BeeEngineError.notConnected }
        let _ = try await client.correctSave()
    }

    /// Shut down the vox session.
    func shutdown() {
        sessionHandle?.shutdown()
        driverTask?.cancel()
        client = nil
        sessionHandle = nil
        driverTask = nil
        rust = nil
    }

    deinit {
        sessionHandle?.shutdown()
        driverTask?.cancel()
    }
}

enum BeeEngineError: LocalizedError {
    case notConnected
    case loadFailed(String)
    case noEngine

    var errorDescription: String? {
        switch self {
        case .notConnected: "BeeEngine is not connected"
        case .loadFailed(let msg): "Engine load failed: \(msg)"
        case .noEngine: "No engine loaded"
        }
    }
}

/// Swift is purely a client — this dispatcher rejects all incoming calls.
private struct NoopBeeDispatcher: ServiceDispatcher {
    func dispatch(
        methodId: UInt64,
        payload: [UInt8],
        requestId: UInt64,
        registry: ChannelRegistry,
        schemaSendTracker: SchemaSendTracker,
        taskTx: @escaping @Sendable (TaskMessage) -> Void
    ) async {
        taskTx(.response(requestId: requestId, payload: encodeUnknownMethodError()))
    }

    func retryPolicy(methodId: UInt64) -> RetryPolicy {
        .volatile
    }

    func preregister(methodId: UInt64, payload: [UInt8], registry: ChannelRegistry) async {}
}
