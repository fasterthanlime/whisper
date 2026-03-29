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
        Self.logger.warning("Accepted XPC connection")
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
    /// The last controller that was active — kept alive (strong ref) as
    /// fallback when transient deactivations nil out activeController.
    var lastController: HarkInputController?

    /// Whether Hark is actively dictating (set by setMarkedText, cleared by commit/cancel).
    var isDictating = false

    /// Best available controller.
    var controller: HarkInputController? {
        activeController ?? lastController
    }

    func setMarkedText(_ text: String) {
        Self.logger.warning("setMarkedText: '\(text.prefix(40), privacy: .public)' activeController=\(self.activeController != nil)")
        isDictating = true
        DispatchQueue.main.async {
            self.controller?.handleSetMarkedText(text)
        }
    }

    func commitText(_ text: String) {
        commitText(text, submit: false)
    }

    func commitText(_ text: String, submit: Bool) {
        Self.logger.warning("commitText: \(text.prefix(40), privacy: .public) submit=\(submit)")
        isDictating = false
        DispatchQueue.main.async {
            self.controller?.handleCommitText(text, submit: submit)
        }
    }

    func cancelInput() {
        isDictating = false
        Self.logger.warning("cancelInput")
        DispatchQueue.main.async {
            self.controller?.handleCancelInput()
        }
    }
}
