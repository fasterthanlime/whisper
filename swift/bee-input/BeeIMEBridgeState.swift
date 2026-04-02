import AppKit
import Foundation

/// Holds all per-activation state. Lives on the bridge, not on the controller.
/// IMKInputController instances are transient — the OS can create new ones
/// before deactivating the old, and both share the same client().
final class BeeIMESession {
    weak var controller: BeeInputController?
    let pid: pid_t?
    let clientID: String?

    // State that used to live on the controller
    var currentMarkedText: String = ""
    var autoCommittedPrefix: String = ""
    var didRequestSwitchAway: Bool = false
    var pendingClaimToken: UUID?

    init(controller: BeeInputController, pid: pid_t?, clientID: String?) {
        self.controller = controller
        self.pid = pid
        self.clientID = clientID
    }

    // MARK: - Deferred claim

    func startDeferredClaim() {
        let token = UUID()
        pendingClaimToken = token
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.06) { [weak self] in
            guard let self, self.pendingClaimToken == token else { return }
            self.pendingClaimToken = nil
            self.performClaim()
        }
    }

    func cancelPendingClaim() {
        guard pendingClaimToken != nil else { return }
        pendingClaimToken = nil
        beeInputLog("deactivateServer: cancelled pending claim")
    }

    private func performClaim() {
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
        let expectedPID = BeeBrokerIMEClient.shared.expectedTargetPID

        if expectedPID != 0 && frontmostPID != pid_t(expectedPID) {
            beeInputLog(
                "activateServer: PID mismatch (frontmost=\(frontmostPID) expected=\(expectedPID)), skipping claim"
            )
            return
        }

        let claim = BeeBrokerIMEClient.shared.claimPreparedSessionSync()
        guard let sessionID = claim.sessionID else {
            if !claim.shouldStayActive {
                beeInputLog("activateServer: no session, switching to next input source")
                controller?.switchToNextInputSource()
            } else {
                beeInputLog("activateServer: no session (staying active, recent session)")
            }
            return
        }

        beeInputLog(
            "activateServer: claimed session=\(sessionID.uuidString.prefix(8)) pid=\(frontmostPID)")
        let bridge = BeeIMEBridgeState.shared
        bridge.attachSession(sessionID: sessionID)
        bridge.flushPending()
        BeeBrokerIMEClient.shared.imeAttach(sessionID: sessionID)
    }

    // MARK: - Text handling

    func handleSetMarkedText(_ text: String) {
        let hasClient = controller?.client() != nil
        beeInputLog(
            "handleSetMarkedText: \(text.prefix(40).debugDescription) hasClient=\(hasClient)")
        guard let client = controller?.client() else {
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
        guard let client = controller?.client() else { return }

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

        beeInputLog("commitText: \(finalText.prefix(60).debugDescription) hasClient=true")
        currentMarkedText = ""
        client.insertText(
            finalText + " ",
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleCancelInput() {
        beeInputLog("cancelInput")
        guard let client = controller?.client() else { return }
        currentMarkedText = ""
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    /// Clear any marked text still showing. Called from deactivateServer
    /// as a last-gasp cleanup before the controller goes away.
    func clearOrphanedMarkedText() {
        guard !currentMarkedText.isEmpty, let client = controller?.client() else { return }
        beeInputLog("deactivateServer: clearing orphaned marked text")
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        currentMarkedText = ""
    }
}

// MARK: - Bridge State

/// Tracks which session is active and routes XPC callbacks to it.
final class BeeIMEBridgeState: NSObject {
    static let shared = BeeIMEBridgeState()

    enum State {
        case idle
        case activated(BeeIMESession)
        case serving(BeeIMESession, sessionID: UUID, pendingText: String?)
    }

    private(set) var state: State = .idle

    // MARK: - Queries

    var isDictating: Bool {
        if case .serving = state { return true }
        return false
    }

    var activeSessionID: UUID? {
        if case .serving(_, let sessionID, _) = state { return sessionID }
        return nil
    }

    var activeController: BeeInputController? {
        currentSession?.controller
    }

    var currentSession: BeeIMESession? {
        switch state {
        case .idle: return nil
        case .activated(let session): return session
        case .serving(let session, _, _): return session
        }
    }

    // MARK: - State transitions

    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?) {
        let session = BeeIMESession(controller: controller, pid: pid, clientID: clientID)
        state = .activated(session)
        beeInputLog("state → activated pid=\(pid.map(String.init) ?? "nil")")
        session.startDeferredClaim()
    }

    func deactivate(_ controller: BeeInputController) {
        guard activeController === controller else { return }
        state = .idle
        beeInputLog("state → idle")
    }

    func attachSession(sessionID: UUID) {
        guard let session = currentSession else {
            beeInputLog("attachSession: ignored (idle, no controller)")
            return
        }
        state = .serving(session, sessionID: sessionID, pendingText: nil)
        beeInputLog("state → serving session=\(sessionID.uuidString.prefix(8))")
    }

    func clearSessionIfMatching(sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            return
        }
        state = .activated(session)
        beeInputLog("state → activated (session \(sessionID.uuidString.prefix(8)) cleared)")
    }

    // MARK: - Text routing

    func flushPending() {
        guard case .serving(let session, let sessionID, let text?) = state else { return }
        beeInputLog("flushPending: delivering \(text.prefix(40).debugDescription)")
        state = .serving(session, sessionID: sessionID, pendingText: nil)
        session.handleSetMarkedText(text)
    }

    func setMarkedText(_ text: String, sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            beeInputLog("setMarkedText: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        if session.controller != nil {
            session.handleSetMarkedText(text)
        } else {
            beeInputLog("setMarkedText: controller lost, queuing")
            state = .serving(session, sessionID: sessionID, pendingText: text)
        }
    }

    func commitText(_ text: String, submit: Bool, sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            beeInputLog("commitText: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        state = .activated(session)
        session.handleCommitText(text, submit: submit)
    }

    func cancelInput(sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            beeInputLog("cancelInput: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        state = .activated(session)
        session.handleCancelInput()
    }

    func stopDictating(sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            beeInputLog("stopDictating: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        state = .activated(session)
    }
}
