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
    func setMarkedText(text: String) async throws -> Bool {
        await MainActor.run { BeeIMEBridgeState.shared.setMarkedText(text) }
        return true
    }

    func commitText(text: String) async throws -> Bool {
        await MainActor.run { BeeIMEBridgeState.shared.commitText(text) }
        return true
    }

    func stopDictating() async throws -> Bool {
        beeInputLog("VOXIPC: stopDictating")
        await MainActor.run { BeeIMEBridgeState.shared.cancelInput() }
        return true
    }

    func replaceText(oldText: String, newText: String) async throws -> Bool {
        beeInputLog("VOXIPC: replaceText")
        await MainActor.run { BeeIMEBridgeState.shared.replaceText(oldText: oldText, newText: newText) }
        return true
    }
}

// MARK: - Client

final class BeeVoxIMEClient: Sendable {
    static let shared = BeeVoxIMEClient()

    private struct State: Sendable {
        var appClient: AppClient?
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
                connector, dispatcher: dispatcher, resumable: true)
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

    func imeAttach() {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeAttach()
            } catch {
                beeInputLog("VOXIPC: imeAttach failed: \(error)")
            }
        }
    }

    func imeKeyEvent(eventType: String, keyCode: UInt32, characters: String) {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeKeyEvent(
                    eventType: eventType, keyCode: keyCode, characters: characters)
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
}
