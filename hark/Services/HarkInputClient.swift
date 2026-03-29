import Carbon
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

    private static let imeBundleID = "fasterthanlime.inputmethod.hark"
    private static let imeBundleName = "harkInput.app"
    private static let machServiceName = "fasterthanlime.hark.input-method.xpc"

    private var connection: NSXPCConnection?

    /// Whether we have an active connection to the IME.
    var isConnected: Bool {
        connection != nil
    }

    /// Whether the IME is registered as an input source with macOS.
    static var isIMERegistered: Bool {
        let props = [kTISPropertyBundleID: imeBundleID as CFString] as CFDictionary
        guard let sources = TISCreateInputSourceList(props, true)?.takeRetainedValue() as? [TISInputSource] else {
            return false
        }
        return !sources.isEmpty
    }

    /// Register the IME with macOS if it's installed but not yet registered.
    /// Returns true if it was already registered or registration succeeded.
    @discardableResult
    static func ensureIMERegistered() -> Bool {
        if isIMERegistered {
            logger.info("IME already registered")
            return true
        }

        // Look for harkInput.app in ~/Library/Input Methods/
        let inputMethodsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Input Methods")
            .appendingPathComponent(imeBundleName)

        guard FileManager.default.fileExists(atPath: inputMethodsDir.path) else {
            logger.warning("IME not found at \(inputMethodsDir.path, privacy: .public)")
            return false
        }

        let status = TISRegisterInputSource(inputMethodsDir as CFURL)
        guard status == noErr else {
            logger.error("TISRegisterInputSource failed: \(status, privacy: .public)")
            return false
        }
        logger.info("IME registered, enabling...")

        // Enable it so it appears in System Settings
        let enableProps = [kTISPropertyBundleID: imeBundleID as CFString] as CFDictionary
        if let sources = TISCreateInputSourceList(enableProps, true)?.takeRetainedValue() as? [TISInputSource] {
            for source in sources {
                let enableStatus = TISEnableInputSource(source)
                logger.info("TISEnableInputSource status: \(enableStatus, privacy: .public)")
            }
        }
        return true
    }

    // MARK: - Distributed notification API (stateless, no connection needed)

    /// Send provisional (marked) text to the IME via distributed notification.
    static func sendSetMarkedText(_ text: String) {
        DistributedNotificationCenter.default().postNotificationName(
            NSNotification.Name("fasterthanlime.hark.setMarkedText"),
            object: nil,
            userInfo: ["text": text],
            deliverImmediately: true
        )
    }

    /// Commit final text via distributed notification.
    static func sendCommitText(_ text: String) {
        DistributedNotificationCenter.default().postNotificationName(
            NSNotification.Name("fasterthanlime.hark.commitText"),
            object: nil,
            userInfo: ["text": text],
            deliverImmediately: true
        )
    }

    /// Cancel input via distributed notification.
    static func sendCancelInput() {
        DistributedNotificationCenter.default().postNotificationName(
            NSNotification.Name("fasterthanlime.hark.cancelInput"),
            object: nil,
            userInfo: nil,
            deliverImmediately: true
        )
    }

    // MARK: - XPC API (unused for now, kept for future)

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
