import Foundation
import VoxRuntime

// MARK: - Socket path

func beeVoxSocketPath() -> String? {
    FileManager.default.containerURL(
        forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee"
    )?.appendingPathComponent("bee.sock").path
}

// MARK: - App-side handler (handles calls FROM IME, hops to MainActor)

private final class AppImpl: AppHandler, @unchecked Sendable {
    private let server: BeeIPCServer
    init(server: BeeIPCServer) { self.server = server }

    func imeHello() async throws -> String {
        await MainActor.run { [server] in server.onImeHello() }
    }
    func imeAttach() async throws -> Bool {
        await MainActor.run { [server] in server.onImeAttach() }
    }
    func imeActivationRevoked() async throws -> Bool {
        beeLog("VOXIPC: imeActivationRevoked"); return true
    }
    func imeContextLost(hadMarkedText: Bool) async throws -> Bool {
        await MainActor.run { [server, hadMarkedText] in
            server.onImeContextLost(hadMarkedText: hadMarkedText)
        }
    }
    func imeKeyEvent(eventType: String, keyCode: UInt32, characters: String) async throws -> Bool {
        await MainActor.run { [server, eventType, keyCode, characters] in
            server.onImeKeyEvent(eventType: eventType, keyCode: keyCode, characters: characters)
        }
    }
}

// MARK: - Server

@MainActor
final class BeeIPCServer {
    static let shared = BeeIPCServer()

    weak var appState: AppState?

    private var imeClient: ImeClient?
    private var imeReadyWaiters: [CheckedContinuation<Bool, Never>] = []

    private init() {}

    // MARK: - Lifecycle

    func start() async {
        guard let path = beeVoxSocketPath() else {
            beeLog("VOXIPC: no app group container URL")
            return
        }
        let acceptor = UnixAcceptor(path: path)
        let dispatcher = AppDispatcher(handler: AppImpl(server: self))
        beeLog("VOXIPC: accepting at \(path)")

        Task { [weak self] in
            while let self {
                do {
                    let session = try await VoxRuntime.Session.acceptor(
                        acceptor, dispatcher: dispatcher, resumable: false)
                    beeLog("VOXIPC: IME connected")
                    self.imeClient = ImeClient(connection: session.connection)
                    let waiters = imeReadyWaiters
                    imeReadyWaiters.removeAll()
                    for w in waiters { w.resume(returning: true) }
                    try await session.run()
                } catch {
                    beeLog("VOXIPC: session error: \(error)")
                }
                self.imeClient = nil
                beeLog("VOXIPC: re-accepting")
            }
        }
    }

    // MARK: - Inbound handlers (called from AppImpl via MainActor.run)

    func onImeHello() -> String {
        beeLog("VOXIPC: imeHello")
        return "bee-app-\(ProcessInfo.processInfo.processIdentifier)"
    }

    func onImeAttach() -> Bool {
        beeLog("VOXIPC: imeAttach")
        appState?.handleIMESessionStarted()
        return true
    }

    func onImeContextLost(hadMarkedText: Bool) -> Bool {
        beeLog("VOXIPC: imeContextLost hadMarkedText=\(hadMarkedText)")
        appState?.handleIMEContextLost(hadMarkedText: hadMarkedText)
        return true
    }

    func onImeKeyEvent(eventType: String, keyCode: UInt32, characters: String) -> Bool {
        beeLog("VOXIPC: imeKeyEvent type=\(eventType) key=\(keyCode)")
        switch eventType {
        case "submit":
            appState?.handleIMESubmit()
        case "cancel":
            appState?.handleIMECancel()
        default:
            appState?.handleIMEUserTyped()
        }
        return true
    }

    // MARK: - Outbound to IME

    var isIMEConnected: Bool { imeClient != nil }

    func waitForIMEReady() async -> Bool {
        if imeClient != nil {
            beeLog("VOXIPC: waitForIMEReady immediate=true")
            return true
        }
        beeLog("VOXIPC: waitForIMEReady waiting")
        return await withCheckedContinuation { continuation in
            imeReadyWaiters.append(continuation)
        }
    }

    func setMarkedText(text: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: setMarkedText — IME not connected, dropping")
            return
        }
        beeLog("VOXIPC: setMarkedText len=\(text.utf16.count) text=\(text.prefix(80).debugDescription)")
        do {
            _ = try await client.setMarkedText(text: text)
        } catch {
            beeLog("VOXIPC: setMarkedText failed: \(error)")
        }
    }

    func commitText(text: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: commitText — IME not connected, dropping")
            return
        }
        beeLog("VOXIPC: commitText len=\(text.utf16.count) text=\(text.prefix(80).debugDescription)")
        do {
            _ = try await client.commitText(text: text)
        } catch {
            beeLog("VOXIPC: commitText failed: \(error)")
        }
    }

    func stopDictating() async {
        guard let client = imeClient else {
            beeLog("VOXIPC: stopDictating — IME not connected, dropping")
            return
        }
        beeLog("VOXIPC: stopDictating")
        do {
            _ = try await client.stopDictating()
        } catch {
            beeLog("VOXIPC: stopDictating failed: \(error)")
        }
    }

    func replaceText(oldText: String, newText: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: replaceText — IME not connected, dropping")
            return
        }
        do {
            _ = try await client.replaceText(oldText: oldText, newText: newText)
        } catch {
            beeLog("VOXIPC: replaceText failed: \(error)")
        }
    }
}
