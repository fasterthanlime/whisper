import Foundation
import os

/// XPC protocol matching HarkInputProtocol in the IME.
@objc protocol HarkInputProtocol {
    func setMarkedText(_ text: String)
    func commitText(_ text: String)
    func cancelInput()
}

/// Client for communicating with the HarkInput input method via XPC.
@MainActor
final class HarkInputClient {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "HarkInputClient"
    )

    private static let machServiceName = "fasterthanlime.hark.input-method.xpc"

    private var connection: NSXPCConnection?

    /// Whether we have an active connection to the IME.
    var isConnected: Bool {
        connection != nil
    }

    /// Connect to the HarkInput IME.
    func connect() {
        disconnect()

        let conn = NSXPCConnection(machServiceName: Self.machServiceName)
        conn.remoteObjectInterface = NSXPCInterface(with: HarkInputProtocol.self)
        conn.invalidationHandler = { [weak self] in
            Task { @MainActor in
                Self.logger.warning("XPC connection invalidated")
                self?.connection = nil
            }
        }
        conn.interruptionHandler = { [weak self] in
            Task { @MainActor in
                Self.logger.warning("XPC connection interrupted")
                self?.connection = nil
            }
        }
        conn.resume()
        connection = conn
        Self.logger.info("Connected to HarkInput IME")
    }

    /// Disconnect from the IME.
    func disconnect() {
        connection?.invalidate()
        connection = nil
    }

    /// Send provisional (marked) text during streaming transcription.
    func setMarkedText(_ text: String) {
        guard let proxy = proxy() else { return }
        proxy.setMarkedText(text)
    }

    /// Commit the final text.
    func commitText(_ text: String) {
        guard let proxy = proxy() else { return }
        proxy.commitText(text)
    }

    /// Cancel — clear marked text.
    func cancelInput() {
        guard let proxy = proxy() else { return }
        proxy.cancelInput()
    }

    private func proxy() -> HarkInputProtocol? {
        guard let connection else {
            Self.logger.warning("No XPC connection — is HarkInput running?")
            return nil
        }
        return connection.remoteObjectProxy as? HarkInputProtocol
    }
}
