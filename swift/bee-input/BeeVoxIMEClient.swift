import Foundation
import VoxRuntime

// MARK: - Socket path

private func beeVoxSocketPath() -> String? {
    FileManager.default.containerURL(
        forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee"
    )?.appendingPathComponent("bee.sock").path
}

// MARK: - IME-side handler (handles calls FROM app)

private final class ImeImpl: ImeHandler, @unchecked Sendable {
    func setMarkedText(sessionId: String, text: String) async throws -> Bool {
        guard let id = UUID(uuidString: sessionId) else { return false }
        await MainActor.run { BeeIMEBridgeState.shared.setMarkedText(text, sessionID: id) }
        return true
    }

    func commitText(sessionId: String, text: String) async throws -> Bool {
        guard let id = UUID(uuidString: sessionId) else { return false }
        await MainActor.run { BeeIMEBridgeState.shared.commitText(text, submit: false, sessionID: id) }
        return true
    }

    func stopDictating(sessionId: String) async throws -> Bool {
        beeInputLog("VOXIPC: stopDictating session=\(sessionId.prefix(8))")
        guard let id = UUID(uuidString: sessionId) else { return false }
        await MainActor.run { BeeIMEBridgeState.shared.cancelInput(sessionID: id) }
        return true
    }

    func prepareSession(sessionId: String, targetPid: Int32) async throws -> Bool {
        beeInputLog("VOXIPC: prepareSession pushed from app session=\(sessionId.prefix(8))")
        await BeeVoxIMEClient.shared.handlePreparedSession(sessionId: sessionId, targetPid: targetPid)
        return true
    }

    func replaceText(sessionId: String, oldText: String, newText: String) async throws -> Bool {
        beeInputLog("VOXIPC: replaceText session=\(sessionId.prefix(8))")
        guard let id = UUID(uuidString: sessionId) else { return false }
        await MainActor.run { BeeIMEBridgeState.shared.replaceText(oldText: oldText, newText: newText, sessionID: id) }
        return true
    }
}

// MARK: - Client

final class BeeVoxIMEClient: Sendable {
    static let shared = BeeVoxIMEClient()

    private struct State: Sendable {
        var appClient: AppClient?
        var pendingSessionId: String?
        var expectedTargetPID: Int32 = 0
    }

    nonisolated(unsafe) private var state = State()
    private let lock = NSLock()

    private init() {}

    func start() {
        Task { await connect() }
    }

    private func connect() async {
        guard let path = beeVoxSocketPath() else {
            beeInputLog("VOXIPC: no app group container URL")
            return
        }
        beeInputLog("VOXIPC: connecting to \(path)")
        let connector = UnixConnector(path: path)
        let dispatcher = ImeDispatcher(handler: ImeImpl())
        do {
            let session = try await VoxRuntime.Session.initiator(
                connector, dispatcher: dispatcher, resumable: false)
            Task { try await session.run() }
            let client = AppClient(connection: session.connection)
            lock.withLock { state.appClient = client }
            let hello = try await client.imeHello()
            beeInputLog("VOXIPC: connected, app replied: \(hello)")
        } catch {
            beeInputLog("VOXIPC: connect failed: \(error)")
            try? await Task.sleep(for: .seconds(2))
            await connect()
        }
    }

    // MARK: - Outbound calls to app

    func claimSession() async -> String? {
        guard let client = lock.withLock({ state.appClient }) else {
            beeInputLog("VOXIPC: claimSession — not connected")
            return nil
        }
        do {
            let id = try await client.claimSession()
            return id.isEmpty ? nil : id
        } catch {
            beeInputLog("VOXIPC: claimSession failed: \(error)")
            return nil
        }
    }

    func imeAttach(sessionId: String) {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeAttach(sessionId: sessionId)
            } catch {
                beeInputLog("VOXIPC: imeAttach failed: \(error)")
            }
        }
    }

    func imeKeyEvent(sessionId: String, eventType: String, keyCode: UInt32, characters: String) {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeKeyEvent(
                    sessionId: sessionId, eventType: eventType,
                    keyCode: keyCode, characters: characters)
            } catch {
                beeInputLog("VOXIPC: imeKeyEvent failed: \(error)")
            }
        }
    }

    func imeContextLost(hadMarkedText: Bool) {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeContextLost(hadMarkedText: hadMarkedText)
            } catch {
                beeInputLog("VOXIPC: imeContextLost failed: \(error)")
            }
        }
    }

    func imeActivationRevoked() {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeActivationRevoked()
            } catch {
                beeInputLog("VOXIPC: imeActivationRevoked failed: \(error)")
            }
        }
    }

    // MARK: - State accessors

    var expectedTargetPID: Int32 {
        get { lock.withLock { state.expectedTargetPID } }
        set { lock.withLock { state.expectedTargetPID = newValue } }
    }

    // MARK: - Called by ImeImpl (inbound from app)

    func handlePreparedSession(sessionId: String, targetPid: Int32) async {
        lock.withLock {
            state.pendingSessionId = sessionId
            state.expectedTargetPID = targetPid
        }
        await MainActor.run {
            let bridge = BeeIMEBridgeState.shared
            guard bridge.activeController != nil else {
                beeInputLog("VOXIPC: prepareSession — no active controller, will claim on next activate")
                return
            }
            Task { await bridge.performAsyncClaim() }
        }
    }
}
