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
    func prepareSession(sessionId: String, targetPid: Int32) async throws -> Bool { false }
    func claimSession() async throws -> String {
        await MainActor.run { [server] in server.onClaimSession() }
    }
    func imeAttach(sessionId: String) async throws -> Bool {
        await MainActor.run { [server, sessionId] in server.onImeAttach(sessionId: sessionId) }
    }
    func imeActivationRevoked() async throws -> Bool {
        beeLog("VOXIPC: imeActivationRevoked"); return true
    }
    func imeContextLost(hadMarkedText: Bool) async throws -> Bool {
        await MainActor.run { [server, hadMarkedText] in
            server.onImeContextLost(hadMarkedText: hadMarkedText)
        }
    }
    func imeKeyEvent(sessionId: String, eventType: String, keyCode: UInt32, characters: String)
        async throws -> Bool
    {
        await MainActor.run { [server, sessionId, eventType, keyCode, characters] in
            server.onImeKeyEvent(
                sessionId: sessionId, eventType: eventType,
                keyCode: keyCode, characters: characters)
        }
    }
}

// MARK: - Server

@MainActor
final class BeeIPCServer {
    static let shared = BeeIPCServer()

    private static let imeSubmitName = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelName = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedName = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostName = NSNotification.Name("fasterthanlime.bee.imeContextLost")
    private static let imeSessionStartedName = NSNotification.Name(
        "fasterthanlime.bee.imeSessionStarted")

    private var imeClient: ImeClient?
    private var pendingSessionId: String?
    private(set) var activeSessionId: String?
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

    func onClaimSession() -> String {
        let id = pendingSessionId
        pendingSessionId = nil
        beeLog("VOXIPC: claimSession → \(id?.prefix(8) ?? "nil")")
        return id ?? ""
    }

    func onImeAttach(sessionId: String) -> Bool {
        beeLog("VOXIPC: imeAttach session=\(sessionId.prefix(8))")
        activeSessionId = sessionId
        NotificationCenter.default.post(
            name: Self.imeSessionStartedName, object: nil,
            userInfo: ["sessionID": sessionId])
        return true
    }

    func onImeContextLost(hadMarkedText: Bool) -> Bool {
        let sessionId = activeSessionId
        beeLog(
            "VOXIPC: imeContextLost hadMarkedText=\(hadMarkedText) session=\(sessionId?.prefix(8) ?? "nil")"
        )
        activeSessionId = nil
        if let id = sessionId {
            NotificationCenter.default.post(
                name: Self.imeContextLostName, object: nil,
                userInfo: ["sessionID": id, "hadMarkedText": hadMarkedText])
        }
        return true
    }

    func onImeKeyEvent(sessionId: String, eventType: String, keyCode: UInt32, characters: String)
        -> Bool
    {
        beeLog("VOXIPC: imeKeyEvent type=\(eventType) key=\(keyCode) session=\(sessionId.prefix(8))")
        switch eventType {
        case "submit":
            NotificationCenter.default.post(
                name: Self.imeSubmitName, object: nil, userInfo: ["sessionID": sessionId])
        case "cancel":
            NotificationCenter.default.post(
                name: Self.imeCancelName, object: nil, userInfo: ["sessionID": sessionId])
        default:
            NotificationCenter.default.post(
                name: Self.imeUserTypedName, object: nil,
                userInfo: [
                    "sessionID": sessionId,
                    "keyCode": Int(keyCode),
                    "characters": characters,
                ])
        }
        return true
    }

    // MARK: - Outbound to IME (called by BeeInputClient)

    var isIMEConnected: Bool { imeClient != nil }

    func waitForIMEReady() async -> Bool {
        if imeClient != nil { return true }
        return await withCheckedContinuation { continuation in
            imeReadyWaiters.append(continuation)
        }
    }

    func prepareDictationSession(sessionId: String, targetPid: Int32) async {
        pendingSessionId = sessionId
        if let client = imeClient {
            do {
                _ = try await client.prepareSession(sessionId: sessionId, targetPid: targetPid)
                beeLog("VOXIPC: prepareSession pushed to IME session=\(sessionId.prefix(8))")
                pendingSessionId = nil
            } catch {
                beeLog("VOXIPC: prepareSession to IME failed: \(error)")
            }
        } else {
            beeLog("VOXIPC: IME not connected, stored pending session=\(sessionId.prefix(8))")
        }
    }

    func setMarkedText(sessionId: String, text: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: setMarkedText — IME not connected, dropping")
            return
        }
        do {
            _ = try await client.setMarkedText(sessionId: sessionId, text: text)
        } catch {
            beeLog("VOXIPC: setMarkedText failed: \(error)")
        }
    }

    func commitText(sessionId: String, text: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: commitText — IME not connected, dropping")
            return
        }
        do {
            _ = try await client.commitText(sessionId: sessionId, text: text)
        } catch {
            beeLog("VOXIPC: commitText failed: \(error)")
        }
    }

    func stopDictating(sessionId: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: stopDictating — IME not connected, dropping")
            return
        }
        do {
            _ = try await client.stopDictating(sessionId: sessionId)
        } catch {
            beeLog("VOXIPC: stopDictating failed: \(error)")
        }
    }

    func replaceText(sessionId: String, oldText: String, newText: String) async {
        guard let client = imeClient else {
            beeLog("VOXIPC: replaceText — IME not connected, dropping")
            return
        }
        do {
            _ = try await client.replaceText(sessionId: sessionId, oldText: oldText, newText: newText)
        } catch {
            beeLog("VOXIPC: replaceText failed: \(error)")
        }
    }
}
