import Foundation

// h[impl ime.parking]
class BeeXPCService: NSObject {
    static let shared = BeeXPCService()

    weak var activeController: BeeInputController?
    var lastController: BeeInputController?
    var isDictating = false

    var controller: BeeInputController? {
        activeController ?? lastController
    }

    func setMarkedText(_ text: String) {
        isDictating = true
        DispatchQueue.main.async {
            self.controller?.handleSetMarkedText(text)
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
