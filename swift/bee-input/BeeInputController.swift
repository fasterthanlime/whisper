import Carbon
import Cocoa
import InputMethodKit

@objc(BeeInputController)
class BeeInputController: IMKInputController {
    private var currentMarkedText: String = ""
    private var autoCommittedPrefix: String = ""
    private var didRequestSwitchAway: Bool = false
    private var pendingClaimToken: UUID?

    override func activateServer(_ sender: Any!) {
        beeInputLog("activateServer: entry")
        super.activateServer(sender)
        didRequestSwitchAway = false
        let bridge = BeeIMEBridgeState.shared
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        let clientIdentity = currentClientIdentity()
        bridge.activate(self, pid: frontmostPID, clientID: clientIdentity)

        // Defer the claim by 60ms. If deactivateServer fires before then,
        // the claim is cancelled and the session stays prepared for retry.
        pendingClaim?.cancel()
        pendingClaim = nil
        let token = UUID()
        pendingClaimToken = token
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) { [weak self] in
            guard let self, self.pendingClaimToken == token else { return }
            self.pendingClaimToken = nil
            self.performClaim()
        }
    }

    private func performClaim() {
        // Check frontmost PID before claiming — during a focus cycle,
        // activateServer may fire on the wrong app. Don't burn the session.
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
        let expectedPID = BeeBrokerIMEClient.shared.expectedTargetPID

        if expectedPID != 0 && frontmostPID != pid_t(expectedPID) {
            beeInputLog("activateServer: PID mismatch (frontmost=\(frontmostPID) expected=\(expectedPID)), skipping claim")
            return
        }

        let claim = BeeBrokerIMEClient.shared.claimPreparedSessionSync()
        guard let sessionID = claim.sessionID else {
            if !claim.shouldStayActive {
                beeInputLog("activateServer: no session, switching to next input source")
                switchToNextInputSource()
            } else {
                beeInputLog("activateServer: no session (staying active, recent session)")
            }
            return
        }

        beeInputLog("activateServer: claimed session=\(sessionID.uuidString.prefix(8)) pid=\(frontmostPID)")
        let bridge = BeeIMEBridgeState.shared
        bridge.attachSession(sessionID: sessionID)
        bridge.flushPending()
        BeeBrokerIMEClient.shared.imeAttach(sessionID: sessionID)
    }

    override func deactivateServer(_ sender: Any!) {
        // Cancel any pending claim — the activation was spurious
        if pendingClaimToken != nil {
            pendingClaimToken = nil
            beeInputLog("deactivateServer: cancelled pending claim")
        }

        let bridge = BeeIMEBridgeState.shared
        let hadMarkedText = !currentMarkedText.isEmpty
        let isDictating = bridge.isDictating
        let sessionID = bridge.activeSessionID

        beeInputLog(
            "deactivateServer: session=\(sessionID?.uuidString.prefix(8) ?? "none") hadMarkedText=\(hadMarkedText)"
        )
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
        guard let event, event.type == .keyDown else {
            return false
        }
        guard let sessionID = BeeIMEBridgeState.shared.activeSessionID else {
            // User is typing but no session — switch away so keystrokes
            // go to the real input source. Only try once per activation.
            if !didRequestSwitchAway {
                didRequestSwitchAway = true
                beeInputLog("handle: no session, switching to next input source")
                switchToNextInputSource()
            }
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
        beeInputLog("handleSetMarkedText: \(text.prefix(40).debugDescription) hasClient=\(self.client() != nil)")
        guard let client = self.client() else {
            beeInputLog("handleSetMarkedText: no client, dropping")
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

        beeInputLog("commitText: \(finalText.prefix(60).debugDescription) hasClient=\(true)")
        currentMarkedText = ""
        client.insertText(
            finalText + " ",
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleCancelInput() {
        beeInputLog("cancelInput")
        guard let client = self.client() else { return }
        currentMarkedText = ""
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private static let beeBundleID = "fasterthanlime.inputmethod.bee"

    private func switchToNextInputSource() {
        let properties: [CFString: Any] = [kTISPropertyInputSourceIsSelectCapable: true]
        guard let sources = TISCreateInputSourceList(properties as CFDictionary, false)?
            .takeRetainedValue() as? [TISInputSource] else { return }

        // Find first non-bee source
        let candidate = sources.first { source in
            guard let bundleID = TISGetInputSourceProperty(source, kTISPropertyBundleID) else {
                return true  // not an IME, probably a keyboard layout — fine
            }
            let id = Unmanaged<CFString>.fromOpaque(bundleID).takeUnretainedValue() as String
            return id != Self.beeBundleID
        }

        guard let next = candidate else {
            beeInputLog("switchToNextInputSource: no alternative found")
            return
        }

        let result = TISSelectInputSource(next)
        beeInputLog("switchToNextInputSource: result=\(result)")
    }

    private func currentClientIdentity() -> String? {
        guard let client = self.client() else { return nil }
        let opaque = Unmanaged.passUnretained(client as AnyObject).toOpaque()
        return String(UInt(bitPattern: opaque), radix: 16, uppercase: true)
    }

}
