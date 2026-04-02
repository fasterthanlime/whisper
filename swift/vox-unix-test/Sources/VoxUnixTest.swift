import Foundation
import VoxRuntime

// MARK: - Server implementation

struct BeeIpcService: BeeIpcHandler, Sendable {
    func imeHello() async throws -> String {
        print("[server] imeHello called")
        return "server-instance-42"
    }

    func prepareSession(sessionId: String, targetPid: Int32) async throws -> Bool {
        print("[server] prepareSession: \(sessionId) pid=\(targetPid)")
        return true
    }

    func claimSession() async throws -> String {
        print("[server] claimSession")
        return "test-session-001"
    }

    func imeAttach(sessionId: String) async throws -> Bool {
        print("[server] imeAttach: \(sessionId)")
        return true
    }

    func setMarkedText(sessionId: String, text: String) async throws -> Bool {
        print("[server] setMarkedText: \(text) session=\(sessionId)")
        return true
    }

    func commitText(sessionId: String, text: String) async throws -> Bool {
        print("[server] commitText: \(text) session=\(sessionId)")
        return true
    }

    func stopDictating(sessionId: String) async throws -> Bool {
        print("[server] stopDictating: \(sessionId)")
        return true
    }

    func imeActivationRevoked() async throws -> Bool {
        print("[server] imeActivationRevoked")
        return true
    }

    func imeContextLost(hadMarkedText: Bool) async throws -> Bool {
        print("[server] imeContextLost hadMarkedText=\(hadMarkedText)")
        return true
    }

    func imeKeyEvent(sessionId: String, eventType: String, keyCode: UInt32, characters: String) async throws -> Bool {
        print("[server] imeKeyEvent: \(eventType) key=\(keyCode) chars=\(characters) session=\(sessionId)")
        return true
    }
}

// MARK: - Dispatcher adapter

final class BeeIpcDispatcherAdapter: ServiceDispatcher, @unchecked Sendable {
    private let handler: any BeeIpcHandler

    init(handler: any BeeIpcHandler) {
        self.handler = handler
    }

    func retryPolicy(methodId: UInt64) -> RetryPolicy {
        BeeIpcChannelingDispatcher.retryPolicy(methodId: methodId)
    }

    func preregister(methodId: UInt64, payload: [UInt8], registry: ChannelRegistry) async {
        await BeeIpcChannelingDispatcher.preregisterChannels(
            methodId: methodId, payload: Data(payload), registry: registry
        )
    }

    func dispatch(
        methodId: UInt64, payload: [UInt8], requestId: UInt64,
        registry: ChannelRegistry, schemaSendTracker: SchemaSendTracker,
        taskTx: @escaping @Sendable (TaskMessage) -> Void
    ) async {
        let dispatcher = BeeIpcChannelingDispatcher(
            handler: handler, registry: registry,
            taskSender: taskTx, schemaSendTracker: schemaSendTracker
        )
        await dispatcher.dispatch(methodId: methodId, requestId: requestId, payload: Data(payload))
    }
}

// MARK: - Entry point

@main
struct VoxUnixTest {
    static func main() async throws {
        setbuf(stdout, nil)
        let socketPath = "/tmp/bee-vox-test.sock"
        unlink(socketPath)
        print("[test] Starting...")

        // Start server
        let listener = try await UnixListener.bind(unixPath: socketPath)
        print("[test] Listening on \(socketPath)")

        Task {
            for await link in listener.acceptConnections() {
                print("[test] Accepted connection")
                let dispatcher = BeeIpcDispatcherAdapter(handler: BeeIpcService())
                do {
                    // Perform transport prologue (initiator sends hello, we accept)
                    _ = try await performAcceptorTransportPrologue(
                        transport: link, supportedConduit: .bare
                    )
                    let session = try await Session.acceptorOn(
                        link, transport: .bare,
                        dispatcher: dispatcher, resumable: false
                    )
                    Task { try await session.run() }
                } catch {
                    print("[test] Acceptor error: \(error)")
                }
            }
        }

        // Brief delay for listener
        try await Task.sleep(for: .milliseconds(50))

        // Connect client
        print("[test] Connecting client...")
        let connector = UnixConnector(path: socketPath)
        let emptyDispatcher = BeeIpcDispatcherAdapter(handler: BeeIpcService())
        let clientSession = try await Session.initiator(
            connector, dispatcher: emptyDispatcher, resumable: false
        )
        Task { try await clientSession.run() }

        let client = BeeIpcClient(connection: clientSession.connection)

        // Make calls
        print("[test] Calling imeHello...")
        let hello = try await client.imeHello()
        print("[test] imeHello → \(hello)")

        print("[test] Calling prepareSession...")
        let prepared = try await client.prepareSession(sessionId: "DEADBEEF", targetPid: 12345)
        print("[test] prepareSession → \(prepared)")

        print("[test] Calling claimSession...")
        let claimed = try await client.claimSession()
        print("[test] claimSession → \(claimed)")

        print("[test] Calling setMarkedText...")
        let marked = try await client.setMarkedText(sessionId: "DEADBEEF", text: "Hello 🐝")
        print("[test] setMarkedText → \(marked)")

        print("[test] Calling commitText...")
        let committed = try await client.commitText(sessionId: "DEADBEEF", text: "Hello world")
        print("[test] commitText → \(committed)")

        print("\n✅ All calls succeeded over Unix socket!")

        try await listener.close()
        unlink(socketPath)
    }
}
