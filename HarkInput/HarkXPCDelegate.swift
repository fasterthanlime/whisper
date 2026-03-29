import Foundation
import os

/// Manages XPC connections from the main Hark app.
class HarkXPCDelegate: NSObject, NSXPCListenerDelegate {
    static let shared = HarkXPCDelegate()

    private static let logger = Logger(
        subsystem: "fasterthanlime.hark.input-method",
        category: "XPC"
    )

    func listener(
        _ listener: NSXPCListener,
        shouldAcceptNewConnection newConnection: NSXPCConnection
    ) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: HarkInputProtocol.self)
        newConnection.exportedObject = HarkXPCService.shared
        newConnection.resume()
        Self.logger.info("Accepted XPC connection")
        return true
    }
}

/// The XPC service that receives commands from the main Hark app
/// and forwards them to the active IMKInputController.
class HarkXPCService: NSObject, HarkInputProtocol {
    static let shared = HarkXPCService()

    private static let logger = Logger(
        subsystem: "fasterthanlime.hark.input-method",
        category: "XPCService"
    )

    /// The currently active input controller (set by HarkInputController).
    weak var activeController: HarkInputController?

    func setMarkedText(_ text: String) {
        Self.logger.info("setMarkedText: \(text.prefix(40), privacy: .public)")
        DispatchQueue.main.async {
            self.activeController?.handleSetMarkedText(text)
        }
    }

    func commitText(_ text: String) {
        Self.logger.info("commitText: \(text.prefix(40), privacy: .public)")
        DispatchQueue.main.async {
            self.activeController?.handleCommitText(text)
        }
    }

    func cancelInput() {
        Self.logger.info("cancelInput")
        DispatchQueue.main.async {
            self.activeController?.handleCancelInput()
        }
    }
}
