import Cocoa
import InputMethodKit
import Carbon.HIToolbox.Events

@objc(BeeInputController)
class BeeInputController: IMKInputController {
    private static let imeSessionStartedNotification = NSNotification.Name("fasterthanlime.bee.imeSessionStarted")
    private static let imeSubmitNotification = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelNotification = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedNotification = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostNotification = NSNotification.Name("fasterthanlime.bee.imeContextLost")

    private var currentMarkedText: String = ""
    private var autoCommittedPrefix: String = ""

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let clientIdentity = currentClientIdentity()
        BeeXPCService.shared.registerActiveController(
            self,
            clientPID: frontmostPID,
            clientIdentity: clientIdentity
        )
        let clientName = (self.client() as? NSObject)?.description.prefix(80) ?? "nil"
        beeInputLog(
            "activateServer: client=\(clientName) clientID=\(clientIdentity ?? "nil") frontmostPID=\(frontmostPID.map(String.init) ?? "nil")"
        )

        guard BeeXPCService.shared.hasActiveSession else {
            beeInputLog("activateServer: no app handshake/session, switching away")
            BeeXPCService.shared.switchAwayFromBeeInput()
            return
        }

        BeeXPCService.shared.flushPending()
        postSessionStartedIfReady()
    }

    override func deactivateServer(_ sender: Any!) {
        beeInputLog("deactivateServer: hadMarkedText=\(!currentMarkedText.isEmpty) isDictating=\(BeeXPCService.shared.isDictating)")
        let hadMarkedText = !currentMarkedText.isEmpty

        BeeXPCService.shared.unregisterActiveController(self)

        if hadMarkedText,
           let sessionID = BeeXPCService.shared.activeSessionID {
            autoCommittedPrefix = currentMarkedText
            Self.postNotification(
                Self.imeContextLostNotification,
                userInfo: ["sessionID": sessionID.uuidString]
            )
        }
        currentMarkedText = ""
        super.deactivateServer(sender)
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event, event.type == .keyDown,
              let sessionID = BeeXPCService.shared.activeSessionID else {
            return false
        }

        switch Int(event.keyCode) {
        case kVK_Return, kVK_ANSI_KeypadEnter:
            Self.postNotification(
                Self.imeSubmitNotification,
                userInfo: ["sessionID": sessionID.uuidString]
            )
            return true

        case kVK_Escape:
            Self.postNotification(
                Self.imeCancelNotification,
                userInfo: ["sessionID": sessionID.uuidString]
            )
            return true

        default:
            Self.postNotification(
                Self.imeUserTypedNotification,
                userInfo: [
                    "sessionID": sessionID.uuidString,
                    "keyCode": event.keyCode,
                    "characters": event.characters ?? "",
                ]
            )
            return false
        }
    }

    // MARK: - Text handling

    func handleSetMarkedText(_ text: String) {
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
        let attributed = NSAttributedString(string: displayText, attributes: [
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
        guard let client = self.client() else { return }

        var finalText = text
        if !autoCommittedPrefix.isEmpty {
            if text.hasPrefix(autoCommittedPrefix) {
                finalText = String(text.dropFirst(autoCommittedPrefix.count))
                finalText = String(finalText.drop(while: { $0 == " " }))
            }
            autoCommittedPrefix = ""
        }
        finalText = finalText
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
        guard let client = self.client() else { return }
        currentMarkedText = ""
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private static func postNotification(_ name: NSNotification.Name, userInfo: [AnyHashable: Any]? = nil) {
        DistributedNotificationCenter.default().postNotificationName(
            name,
            object: nil,
            userInfo: userInfo,
            deliverImmediately: true
        )
    }

    private func currentClientIdentity() -> String? {
        guard let client = self.client() else { return nil }
        let opaque = Unmanaged.passUnretained(client as AnyObject).toOpaque()
        return String(UInt(bitPattern: opaque), radix: 16, uppercase: true)
    }

    private func postSessionStartedIfReady() {
        guard let ack = BeeXPCService.shared.consumeSessionStartAcknowledgementIfReady() else {
            return
        }
        var userInfo: [AnyHashable: Any] = ["sessionID": ack.sessionID.uuidString]
        if let clientID = ack.clientIdentity {
            userInfo["clientID"] = clientID
        }
        Self.postNotification(
            Self.imeSessionStartedNotification,
            userInfo: userInfo
        )
        beeInputLog(
            "imeSessionStarted: session=\(ack.sessionID.uuidString.prefix(8)) clientID=\(ack.clientIdentity ?? "nil")"
        )
    }
}
