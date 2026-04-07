import Foundation
import VoxRuntime

/// Manages the vox-ffi connection to the Rust bee-ffi dylib.
/// After connect(), exposes the BeeClient directly.
actor BeeEngine {
    private var rust: FfiDynamicLibrary?
    private(set) var client: BeeClient?
    private var sessionHandle: SessionHandle?
    private var driverTask: Task<Void, Error>?

    /// Path to the libbee_ffi.dylib — checks app bundle first, then build output.
    private static func libraryURL() -> URL {
        // In the installed app: Contents/Frameworks/libbee_ffi.dylib
        if let frameworksPath = Bundle.main.privateFrameworksPath {
            let bundled = URL(fileURLWithPath: frameworksPath)
                .appendingPathComponent("libbee_ffi.dylib")
            beeLog("BEE-ENGINE: checking bundled path \(bundled.path) exists=\(FileManager.default.fileExists(atPath: bundled.path))")
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
        beeLog("BEE-ENGINE: checking dev path \(devPath.path) exists=\(FileManager.default.fileExists(atPath: devPath.path))")
        if FileManager.default.fileExists(atPath: devPath.path) {
            return devPath
        }

        // Fallback: assume Frameworks dir next to executable
        let fallback = Bundle.main.executableURL!
            .deletingLastPathComponent()  // Contents/MacOS
            .deletingLastPathComponent()  // Contents
            .appendingPathComponent("Frameworks/libbee_ffi.dylib")
        beeLog("BEE-ENGINE: using fallback path \(fallback.path)")
        return fallback
    }

    /// Connect to the Rust bee-ffi service via vox-ffi.
    /// No-op if already connected (FFI link cannot be re-established).
    func connect() async throws {
        if client != nil { return }

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
