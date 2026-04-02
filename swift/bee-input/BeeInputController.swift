import Carbon.HIToolbox.Events
import Cocoa
import InputMethodKit

@objc(BeeInputController)
class BeeInputController: IMKInputController {
    private var currentMarkedText: String = ""
    private var autoCommittedPrefix: String = ""

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        let bridge = BeeIMEBridgeState.shared
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let clientIdentity = currentClientIdentity()
        bridge.activate(self, pid: frontmostPID, clientID: clientIdentity)

        // Synchronous XPC claim — blocks so deactivateServer can't race.
        guard let sessionID = BeeBrokerIMEClient.shared.claimPreparedSessionSync() else {
            beeInputLog("activateServer: no prepared session")
            return
        }

        beeInputLog("activateServer: claimed session=\(sessionID.uuidString.prefix(8))")
        bridge.attachSession(sessionID: sessionID)
        bridge.flushPending()
        BeeBrokerIMEClient.shared.imeAttach(sessionID: sessionID)
    }

    override func deactivateServer(_ sender: Any!) {
        let bridge = BeeIMEBridgeState.shared
        let hadMarkedText = !currentMarkedText.isEmpty
        let isDictating = bridge.isDictating
        let sessionID = bridge.activeSessionID

        beeInputLog("deactivateServer: hadMarkedText=\(hadMarkedText) isDictating=\(isDictating)")
        bridge.deactivate(self)

        if isDictating, let sessionID {
            autoCommittedPrefix = currentMarkedText
            BeeBrokerIMEClient.shared.imeContextLost(
                sessionID: sessionID,
                hadMarkedText: hadMarkedText
            )
        }
        currentMarkedText = ""
        super.deactivateServer(sender)
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        beeInputLog("handle!")
        guard let event, event.type == .keyDown,
            let sessionID = BeeIMEBridgeState.shared.activeSessionID
        else {
            return false
        }

        switch Int(event.keyCode) {
        case kVK_Return, kVK_ANSI_KeypadEnter:
            BeeBrokerIMEClient.shared.imeSubmit(sessionID: sessionID)
            return true

        case kVK_Escape:
            BeeBrokerIMEClient.shared.imeCancel(sessionID: sessionID)
            return true

        default:
            BeeBrokerIMEClient.shared.imeUserTyped(
                sessionID: sessionID,
                keyCode: event.keyCode,
                characters: event.characters ?? ""
            )
            return false
        }
    }

    // MARK: - Text handling

    func handleSetMarkedText(_ text: String) {
        beeInputLog("handleSetMarkedText!")
        guard let client = self.client() else {
            beeInputLog("handleSetMarkedText: NO CLIENT, dropping text")
            return
        }

        var displayText = text
        if !autoCommittedPrefix.isEmpty {
            if text.hasPrefix(autoCommittedPrefix) {
                displayText = String(text.dropFirst(autoCommittedPrefix.count))
                displayText = String(displayText.drop(while: { $0 == " " }))
            }
            autoCommittedPrefix = ""
        }

        currentMarkedText = displayText

        // Use markedClauseSegment to hint that this text shouldn't get the
        // default "thick underline" marked text treatment. Value 0 = single segment.
        // Also explicitly request no underline and a subtle background.
        let attributed = NSAttributedString(
            string: displayText,
            attributes: [
                .markedClauseSegment: 0,
                .underlineStyle: 0,
                .backgroundColor: NSColor.textColor.withAlphaComponent(0.06),
            ])

        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: displayText.utf16.count, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleCommitText(_ text: String, submit: Bool = false) {
        beeInputLog("handleCommitText!")
        guard let client = self.client() else { return }

        var finalText = text
        if !autoCommittedPrefix.isEmpty {
            if text.hasPrefix(autoCommittedPrefix) {
                finalText = String(text.dropFirst(autoCommittedPrefix.count))
                finalText = String(finalText.drop(while: { $0 == " " }))
            }
            autoCommittedPrefix = ""
        }
        finalText =
            finalText
            .replacingOccurrences(of: "🐝", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !finalText.isEmpty else {
            currentMarkedText = ""
            return
        }

        currentMarkedText = ""
        client.insertText(
            finalText + " ",
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleCancelInput() {
        beeInputLog("handleCancelInput!")
        guard let client = self.client() else { return }
        currentMarkedText = ""
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private func currentClientIdentity() -> String? {
        guard let client = self.client() else { return nil }
        let opaque = Unmanaged.passUnretained(client as AnyObject).toOpaque()
        return String(UInt(bitPattern: opaque), radix: 16, uppercase: true)
    }

}
