import Foundation

class BeeXPCService: NSObject {
    static let shared = BeeXPCService()

    weak var activeController: BeeInputController?
    var lastController: BeeInputController?
    var isDictating = false
    var pendingText: String?

    var controller: BeeInputController? {
        activeController ?? lastController
    }

    /// Called from BeeInputController.activateServer to flush any pending text.
    func flushPending() {
        if let text = pendingText, let ctrl = controller {
            beeInputLog("flushPending: delivering \(text.prefix(40).debugDescription)")
            pendingText = nil
            ctrl.handleSetMarkedText(text)
        }
    }

    func setMarkedText(_ text: String) {
        isDictating = true
        DispatchQueue.main.async {
            if let ctrl = self.controller {
                ctrl.handleSetMarkedText(text)
            } else {
                beeInputLog("setMarkedText: no controller, queuing \(text.prefix(40).debugDescription)")
                self.pendingText = text
            }
        }
    }

    func commitText(_ text: String, submit: Bool) {
        isDictating = false
        DispatchQueue.main.async {
            self.controller?.handleCommitText(text, submit: submit)
        }
    }

    func cancelInput() {
        isDictating = false
        DispatchQueue.main.async {
            self.controller?.handleCancelInput()
        }
    }
}
