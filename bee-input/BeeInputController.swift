import Cocoa
import InputMethodKit
import Carbon.HIToolbox.Events

@objc(BeeInputController)
class BeeInputController: IMKInputController {
    private var currentMarkedText: String = ""
    private var autoCommittedPrefix: String = ""

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        BeeXPCService.shared.activeController = self
        BeeXPCService.shared.lastController = self
    }

    override func deactivateServer(_ sender: Any!) {
        let hadMarkedText = !currentMarkedText.isEmpty

        if hadMarkedText && BeeXPCService.shared.isDictating {
            autoCommittedPrefix = currentMarkedText
        }
        currentMarkedText = ""

        if !hadMarkedText {
            if BeeXPCService.shared.activeController === self {
                BeeXPCService.shared.activeController = nil
            }
        }
        super.deactivateServer(sender)
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event, event.type == .keyDown, BeeXPCService.shared.isDictating else {
            return false
        }

        switch Int(event.keyCode) {
        case kVK_Return, kVK_ANSI_KeypadEnter:
            DistributedNotificationCenter.default().postNotificationName(
                NSNotification.Name("fasterthanlime.bee.imeSubmit"),
                object: nil, userInfo: nil, deliverImmediately: true
            )
            return true

        case kVK_Escape:
            DistributedNotificationCenter.default().postNotificationName(
                NSNotification.Name("fasterthanlime.bee.imeCancel"),
                object: nil, userInfo: nil, deliverImmediately: true
            )
            return true

        default:
            return false
        }
    }

    // MARK: - Text handling

    func handleSetMarkedText(_ text: String) {
        guard let client = self.client() else { return }

        var displayText = text
        if !autoCommittedPrefix.isEmpty {
            if text.hasPrefix(autoCommittedPrefix) {
                displayText = String(text.dropFirst(autoCommittedPrefix.count))
                displayText = String(displayText.drop(while: { $0 == " " }))
            }
            autoCommittedPrefix = ""
        }

        currentMarkedText = displayText

        let attributed = NSAttributedString(string: displayText, attributes: [
            .underlineStyle: 0,
            .backgroundColor: NSColor.systemBlue.withAlphaComponent(0.08),
        ])

        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: displayText.utf16.count, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleCommitText(_ text: String, submit: Bool = false) {
        guard let client = self.client() else { return }

        var finalText = text
        if !autoCommittedPrefix.isEmpty {
            if text.hasPrefix(autoCommittedPrefix) {
                finalText = String(text.dropFirst(autoCommittedPrefix.count))
                finalText = String(finalText.drop(while: { $0 == " " }))
            }
            autoCommittedPrefix = ""
        }

        currentMarkedText = ""
        client.insertText(
            finalText + " ",
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleCancelInput() {
        guard let client = self.client() else { return }
        currentMarkedText = ""
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }
}
