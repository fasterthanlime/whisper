import Foundation
import VoxRuntime

// MARK: - Socket path

private func beeVoxSocketPath() -> String? {
    FileManager.default.containerURL(
        forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee"
    )?.appendingPathComponent("bee.sock").path
}

// MARK: - Client

final class AppClientFactory: Sendable {
    static let shared = AppClientFactory()

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
            beeInputLog("APPCLIENT: no app group container URL")
            return
        }
        beeInputLog("APPCLIENT: connecting to \(path)")
        let connector = UnixConnector(path: path)
        let dispatcher = ImeDispatcher(handler: ImeImpl())
        do {
            let session = try await VoxRuntime.Session.initiator(
                connector, dispatcher: dispatcher, resumable: true)
            Task { try await session.run() }
            let client = AppClient(connection: session.connection)
            lock.withLock { state.appClient = client }
            let hello = try await client.imeHello()
            beeInputLog("APPCLIENT: connected, app replied: \(hello)")
        } catch {
            beeInputLog("APPCLIENT: connect failed: \(error)")
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
                beeInputLog("APPCLIENT: imeAttach failed: \(error)")
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
                beeInputLog("APPCLIENT: imeKeyEvent failed: \(error)")
            }
        }
    }

    func imeContextLost(hadMarkedText: Bool) {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeContextLost(hadMarkedText: hadMarkedText)
            } catch {
                beeInputLog("APPCLIENT: imeContextLost failed: \(error)")
            }
        }
    }

    func imeActivationRevoked() {
        guard let client = lock.withLock({ state.appClient }) else { return }
        Task {
            do {
                _ = try await client.imeActivationRevoked()
            } catch {
                beeInputLog("APPCLIENT: imeActivationRevoked failed: \(error)")
            }
        }
    }
}
