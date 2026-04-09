import AppKit
import Carbon
import Foundation
import InputMethodKit

/// Holds all per-activation state. Lives on the bridge, not on the controller.
/// IMKInputController instances are transient — the OS can create new ones
/// before deactivating the old, and both share the same client().
final class BeeIMESession {
    weak var controller: BeeInputController?
    let pid: pid_t?
    let clientID: String?

    var currentMarkedText: String = ""
    var lastCommittedText: String = ""

    init(controller: BeeInputController, pid: pid_t?, clientID: String?) {
        self.controller = controller
        self.pid = pid
        self.clientID = clientID
    }

    // MARK: - Text handling

    func handleSetMarkedText(_ text: String) {
        guard let client = controller?.client() else {
            beeInputLog("handleSetMarkedText: no client, dropping")
            return
        }

        currentMarkedText = text
        beeInputLog(
            "handleSetMarkedText: pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") len=\(text.utf16.count) markedRange(before)=\(NSStringFromRange(client.markedRange())) selectedRange(before)=\(NSStringFromRange(client.selectedRange())) text=\(text.prefix(80).debugDescription)"
        )

        let attributed = NSAttributedString(
            string: text,
            attributes: [.markedClauseSegment: 0])

        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: text.utf16.count, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        beeInputLog(
            "handleSetMarkedText: clientID=\(clientID ?? "nil") markedRange(after)=\(NSStringFromRange(client.markedRange())) selectedRange(after)=\(NSStringFromRange(client.selectedRange()))"
        )
    }

    func handleCommitText(_ text: String, submit: Bool = false) {
        guard let client = controller?.client() else {
            beeInputLog("handleCommitText: no client, dropping")
            return
        }

        let finalText =
            text
            .replacingOccurrences(of: "🐝", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !finalText.isEmpty else {
            currentMarkedText = ""
            return
        }

        beeInputLog(
            "commitText: clientID=\(clientID ?? "nil") submit=\(submit) markedRange(before)=\(NSStringFromRange(client.markedRange())) selectedRange(before)=\(NSStringFromRange(client.selectedRange())) text=\(finalText.prefix(80).debugDescription)"
        )
        currentMarkedText = ""
        let textWithSpace = finalText + " "
        lastCommittedText = textWithSpace
        client.insertText(
            textWithSpace,
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        beeInputLog(
            "commitText: clientID=\(clientID ?? "nil") markedRange(after)=\(NSStringFromRange(client.markedRange())) selectedRange(after)=\(NSStringFromRange(client.selectedRange())) inserted=\(textWithSpace.prefix(80).debugDescription)"
        )
    }

    func handleReplaceText(oldText: String, newText: String) {
        guard let client = controller?.client() else {
            beeInputLog("handleReplaceText: no client, dropping")
            return
        }
        // The old text (with trailing space) was just committed.
        // Cursor should be right after it. Use selectedRange to find where we are,
        // then compute the replacement range by looking back.
        let sel = client.selectedRange()
        let oldWithSpace = oldText + " "
        let oldLen = oldWithSpace.utf16.count
        let replaceStart = sel.location >= oldLen ? sel.location - oldLen : 0
        let replaceRange = NSRange(location: replaceStart, length: oldLen)
        let newWithSpace = newText + " "
        beeInputLog(
            "handleReplaceText: clientID=\(clientID ?? "nil") selectedRange(before)=\(NSStringFromRange(sel)) replaceRange=\(NSStringFromRange(replaceRange)) old=\(oldWithSpace.prefix(60).debugDescription) new=\(newWithSpace.prefix(60).debugDescription)"
        )
        client.insertText(
            newWithSpace,
            replacementRange: replaceRange
        )
        lastCommittedText = newWithSpace
    }

}

// MARK: - Bridge State

/// Tracks which session is active and routes Vox callbacks to it.
@MainActor
final class BeeIMEBridgeState: NSObject {
    static let shared = BeeIMEBridgeState()

    enum State {
        case idle
        case activated(BeeIMESession)
        case serving(BeeIMESession, sessionID: UUID, pendingText: String?)
    }

    private(set) var state: State = .idle

    /// Retained after deactivation so post-session replaceText can still reach the client.
    private var lastSession: BeeIMESession?
    private var lastSessionID: UUID?

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

    func discardMarkedText(on client: AnyObject?, reason: String) {
        guard let client else {
            beeInputLog("discardMarkedText[\(reason)]: no client object")
            return
        }
        guard let client = client as? (any IMKTextInput & NSObjectProtocol) else {
            beeInputLog(
                "discardMarkedText[\(reason)]: client does not conform to IMKTextInput type=\(String(describing: type(of: client)))"
            )
            return
        }

        let markedRange = client.markedRange()
        let hasMarked = markedRange.location != NSNotFound && markedRange.length > 0

        if hasMarked {
            beeInputLog(
                "discardMarkedText[\(reason)]: via insertText client=\(String(describing: type(of: client))) markedRange(before)=\(NSStringFromRange(markedRange)) selectedRange(before)=\(NSStringFromRange(client.selectedRange()))"
            )
            // Force-delete any pending composition text so clients can't commit it on focus change.
            client.insertText(
                "",
                replacementRange: markedRange
            )
            // As a safety net, also ensure composition state is cleared.
            client.setMarkedText(
                "",
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: 0)
            )
            beeInputLog(
                "discardMarkedText[\(reason)]: markedRange(after)=\(NSStringFromRange(client.markedRange())) selectedRange(after)=\(NSStringFromRange(client.selectedRange()))"
            )
        } else {
            // No marked range; ensure the composition state is cleared anyway at the caret.
            beeInputLog(
                "discardMarkedText[\(reason)]: no marked range; ensuring clear via empty setMarkedText at caret"
            )
            client.setMarkedText(
                "",
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: 0)
            )
        }
    }

    // MARK: - State transitions

    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?) {
        if case .serving(let session, let sessionID, let pending) = state {
            beeInputLog(
                "activate: already serving session=\(sessionID.uuidString.prefix(8)), updating controller pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") pendingLen=\(pending?.utf16.count ?? 0)"
            )
            session.controller = controller
            state = .serving(session, sessionID: sessionID, pendingText: pending)
            return
        }
        lastSession = nil
        lastSessionID = nil
        let session = BeeIMESession(controller: controller, pid: pid, clientID: clientID)
        state = .activated(session)
        beeInputLog(
            "state → activated pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil")")
        Task { await self.performAsyncClaim() }
    }

    func deactivate(_ controller: BeeInputController) {
        guard activeController === controller else { return }
        switch state {
        case .idle:
            lastSession = nil
            lastSessionID = nil
        case .activated(let session):
            lastSession = session
            lastSessionID = nil
        case .serving(let session, let sessionID, _):
            lastSession = session
            lastSessionID = sessionID
        }
        state = .idle
        beeInputLog(
            "state → idle lastSessionID=\(lastSessionID?.uuidString.prefix(8) ?? "nil") lastClientID=\(lastSession?.clientID ?? "nil")"
        )
    }

    func attachSession(sessionID: UUID) {
        guard let session = currentSession else {
            beeInputLog("attachSession: ignored (idle, no controller)")
            return
        }
        state = .serving(session, sessionID: sessionID, pendingText: nil)
        beeInputLog(
            "state → serving session=\(sessionID.uuidString.prefix(8)) pid=\(session.pid.map(String.init) ?? "nil") clientID=\(session.clientID ?? "nil")"
        )
    }

    func clearSessionIfMatching(sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            return
        }
        state = .activated(session)
        beeInputLog("state → activated (session \(sessionID.uuidString.prefix(8)) cleared)")
    }

    // MARK: - Session claim (async — called via Task from activate, or pushed by BeeVoxIMEClient)

    func performAsyncClaim() async {
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier ?? 0
        let expectedPID = BeeVoxIMEClient.shared.expectedTargetPID

        if expectedPID != 0 && frontmostPID != pid_t(expectedPID) {
            beeInputLog(
                "performAsyncClaim: PID mismatch (frontmost=\(frontmostPID) expected=\(expectedPID)), skipping"
            )
            return
        }

        beeInputLog(
            "performAsyncClaim: claimSession start frontmostPID=\(frontmostPID) expectedPID=\(expectedPID)"
        )
        guard let sessionIDString = await BeeVoxIMEClient.shared.claimSession() else {
            beeInputLog("performAsyncClaim: no session (palette mode or not ready)")
            return
        }
        guard let sessionID = UUID(uuidString: sessionIDString) else {
            beeInputLog("performAsyncClaim: invalid session ID: \(sessionIDString)")
            return
        }

        // Check we're still in activated state (could have been deactivated while waiting)
        guard currentSession != nil else {
            beeInputLog(
                "performAsyncClaim: deactivated while waiting for claim, dropping session=\(sessionID.uuidString.prefix(8))"
            )
            return
        }

        beeInputLog(
            "performAsyncClaim: claimed session=\(sessionID.uuidString.prefix(8)) pid=\(frontmostPID)"
        )
        attachSession(sessionID: sessionID)
        flushPending()
        BeeVoxIMEClient.shared.imeAttach(sessionId: sessionID.uuidString)
    }

    // MARK: - Text routing

    func flushPending() {
        guard case .serving(let session, let sessionID, let text?) = state else { return }
        beeInputLog(
            "flushPending: session=\(sessionID.uuidString.prefix(8)) delivering len=\(text.utf16.count) text=\(text.prefix(60).debugDescription)"
        )
        state = .serving(session, sessionID: sessionID, pendingText: nil)
        session.handleSetMarkedText(text)
    }

    func setMarkedText(_ text: String, sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            beeInputLog("setMarkedText: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        if session.controller != nil {
            beeInputLog(
                "setMarkedText: delivering session=\(sessionID.uuidString.prefix(8)) clientID=\(session.clientID ?? "nil") len=\(text.utf16.count)"
            )
            session.handleSetMarkedText(text)
        } else {
            beeInputLog(
                "setMarkedText: controller lost, queuing session=\(sessionID.uuidString.prefix(8)) clientID=\(session.clientID ?? "nil") len=\(text.utf16.count)"
            )
            state = .serving(session, sessionID: sessionID, pendingText: text)
        }
    }

    func commitText(_ text: String, submit: Bool, sessionID: UUID) {
        guard case .serving(let session, let currentID, _) = state, currentID == sessionID else {
            beeInputLog("commitText: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        beeInputLog(
            "commitText: session=\(sessionID.uuidString.prefix(8)) clientID=\(session.clientID ?? "nil") submit=\(submit) len=\(text.utf16.count)"
        )
        state = .activated(session)
        session.handleCommitText(text, submit: submit)
    }

    func cancelInput(sessionID: UUID) {
        if case .serving(let session, let currentID, _) = state, currentID == sessionID {
            if let client = session.controller?.client() {
                beeInputLog(
                    "cancelInput: discardMarkedText session=\(sessionID.uuidString.prefix(8)) clientID=\(session.clientID ?? "nil")"
                )
                discardMarkedText(on: client as AnyObject, reason: "cancel")
            } else {
                session.currentMarkedText = ""
            }
            session.currentMarkedText = ""
            state = .activated(session)
            beeInputLog("cancelInput: state → activated session=\(sessionID.uuidString.prefix(8))")
            return
        }

        guard let session = lastSession, lastSessionID == sessionID else {
            beeInputLog("cancelInput: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        session.currentMarkedText = ""
        lastSessionID = nil
    }

    func stopDictating(sessionID: UUID) {
        if case .serving(let session, let currentID, _) = state, currentID == sessionID {
            beeInputLog(
                "stopDictating: state serving → activated session=\(sessionID.uuidString.prefix(8)) clientID=\(session.clientID ?? "nil")"
            )
            state = .activated(session)
            return
        }

        guard lastSessionID == sessionID else {
            beeInputLog("stopDictating: stale session=\(sessionID.uuidString.prefix(8)), dropping")
            return
        }
        lastSessionID = nil
    }

    func replaceText(oldText: String, newText: String, sessionID: UUID) {
        // Try current session first, fall back to lastSession for post-dictation corrections
        guard let session = currentSession ?? lastSession else {
            beeInputLog("replaceText: no session (current or last), dropping")
            return
        }
        session.handleReplaceText(oldText: oldText, newText: newText)
    }

    func didCancelComposition(on controller: BeeInputController) {
        guard let session = currentSession, session.controller === controller else { return }
        beeInputLog(
            "didCancelComposition: clientID=\(session.clientID ?? "nil") clearing currentMarkedText len=\(session.currentMarkedText.utf16.count)"
        )
        session.currentMarkedText = ""
    }
}

