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
    let bundleID: String?

    var currentMarkedText: String = ""
    var lastCommittedText: String = ""

    init(controller: BeeInputController, pid: pid_t?, clientID: String?, bundleID: String?) {
        self.controller = controller
        self.pid = pid
        self.clientID = clientID
        self.bundleID = bundleID
    }

    // MARK: - Text handling

    func handleSetMarkedText(_ text: String) {
        guard let client = controller?.client() else {
            beeInputLog("handleSetMarkedText: no client, dropping")
            return
        }

        currentMarkedText = text
        beeInputLog(
            "handleSetMarkedText: pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") len=\(text.utf16.count) text=\(text.prefix(80).debugDescription)"
        )

        let attributed = NSAttributedString(
            string: text,
            attributes: [.markedClauseSegment: 0])

        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: text.utf16.count, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
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
            "commitText: clientID=\(clientID ?? "nil") submit=\(submit) text=\(finalText.prefix(80).debugDescription)"
        )
        currentMarkedText = ""
        let textWithSpace = finalText + " "
        lastCommittedText = textWithSpace
        client.insertText(
            textWithSpace,
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    func handleReplaceText(oldText: String, newText: String) {
        guard let client = controller?.client() else {
            beeInputLog("handleReplaceText: no client, dropping")
            return
        }
        let sel = client.selectedRange()
        let oldWithSpace = oldText + " "
        let oldLen = oldWithSpace.utf16.count
        let replaceStart = sel.location >= oldLen ? sel.location - oldLen : 0
        let replaceRange = NSRange(location: replaceStart, length: oldLen)
        let newWithSpace = newText + " "
        beeInputLog(
            "handleReplaceText: clientID=\(clientID ?? "nil") old=\(oldWithSpace.prefix(60).debugDescription) new=\(newWithSpace.prefix(60).debugDescription)"
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
        case active(BeeIMESession, pendingText: String?)
    }

    private(set) var state: State = .idle

    /// The `uniqueClientIdentifierString` of the client where dictation started.
    /// We only deliver `setMarkedText` to this client to prevent leakage when switching apps.
    private var dictationOriginClientID: String?

    /// Weak reference to the actual `IMKTextInput` client where dictation started.
    /// Used to clear marked text when canceling from a different app.
    private weak var dictationOriginClient: (any IMKTextInput)?

    // MARK: - Queries

    var isDictating: Bool {
        if case .active = state { return true }
        return false
    }

    var activeController: BeeInputController? {
        currentSession?.controller
    }

    var currentSession: BeeIMESession? {
        if case .active(let session, _) = state { return session }
        return nil
    }

    // MARK: - State transitions

    /// Returns true if this is a fresh activation (not just a controller update).
    @discardableResult
    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?) -> Bool {
        if case .active(let session, let pending) = state {
            beeInputLog(
                "activate: already active, updating controller pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") pendingLen=\(pending?.utf16.count ?? 0)"
            )
            session.controller = controller
            return false
        }
        let bundleID = controller.client()?.bundleIdentifier()
        let session = BeeIMESession(
            controller: controller, pid: pid, clientID: clientID, bundleID: bundleID)
        state = .active(session, pendingText: nil)
        beeInputLog(
            "state → active pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") bundle=\(bundleID ?? "nil")"
        )
        return true
    }

    func deactivate(_ controller: BeeInputController) {
        guard activeController === controller else { return }
        state = .idle
        beeInputLog("state → idle")
    }

    // MARK: - Text routing

    func flushPending() {
        guard case .active(let session, let text?) = state else { return }
        // Only deliver to the original client to prevent text leakage
        // into apps we switched into during dictation.
        guard session.clientID == dictationOriginClientID else {
            beeInputLog(
                "flushPending: client mismatch, dropping (origin=\(dictationOriginClientID ?? "nil") current=\(session.clientID ?? "nil"))"
            )
            return
        }
        beeInputLog(
            "flushPending: delivering len=\(text.utf16.count) text=\(text.prefix(60).debugDescription)"
        )
        state = .active(session, pendingText: nil)
        session.handleSetMarkedText(text)
    }

    func setMarkedText(_ text: String) {
        guard case .active(let session, _) = state else {
            beeInputLog("setMarkedText: not active, dropping")
            return
        }

        // Capture origin on first setMarkedText of a dictation session
        if dictationOriginClientID == nil {
            dictationOriginClientID = session.clientID
            dictationOriginClient = session.controller?.client()
            beeInputLog("setMarkedText: captured origin clientID=\(session.clientID ?? "nil")")
        }

        // Only deliver to the original client
        if session.clientID != dictationOriginClientID {
            beeInputLog(
                "setMarkedText: client mismatch, dropping (origin=\(dictationOriginClientID ?? "nil") current=\(session.clientID ?? "nil"))"
            )
            return
        }

        if session.controller != nil {
            session.handleSetMarkedText(text)
        } else {
            beeInputLog(
                "setMarkedText: controller lost, queuing clientID=\(session.clientID ?? "nil") len=\(text.utf16.count)"
            )
            state = .active(session, pendingText: text)
        }
    }

    func commitText(_ text: String, submit: Bool = false) {
        guard case .active(let session, _) = state else {
            beeInputLog("commitText: not active, dropping")
            dictationOriginClientID = nil
            return
        }
        // Only commit text to the original client to prevent
        // injecting text into apps we switched into during dictation.
        guard session.clientID == dictationOriginClientID else {
            beeInputLog(
                "commitText: client mismatch, dropping (origin=\(dictationOriginClientID ?? "nil") current=\(session.clientID ?? "nil"))"
            )
            dictationOriginClientID = nil
            return
        }
        session.handleCommitText(text, submit: submit)
        dictationOriginClientID = nil
        dictationOriginClient = nil
    }

    func cancelInput() {
        guard case .active(let session, _) = state else {
            beeInputLog("cancelInput: not active, dropping")
            dictationOriginClientID = nil
            dictationOriginClient = nil
            return
        }
        // Always clear marked text in the original client so stale
        // text doesn't linger when we switch back after canceling.
        if let originClient = dictationOriginClient {
            let empty = NSAttributedString(string: "", attributes: [.markedClauseSegment: 0])
            originClient.setMarkedText(
                empty,
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: 0)
            )
        }
        if session.clientID == dictationOriginClientID {
            session.currentMarkedText = ""
        }
        dictationOriginClientID = nil
        dictationOriginClient = nil
        beeInputLog("cancelInput: done, cleared origin")
    }

    func replaceText(oldText: String, newText: String) {
        guard let session = currentSession else {
            beeInputLog("replaceText: no active session, dropping")
            return
        }
        // Only deliver to the original client to prevent text replacement
        // in apps we switched into during dictation.
        guard session.clientID == dictationOriginClientID else {
            beeInputLog(
                "replaceText: client mismatch, dropping (origin=\(dictationOriginClientID ?? "nil") current=\(session.clientID ?? "nil"))"
            )
            return
        }
        session.handleReplaceText(oldText: oldText, newText: newText)
    }
}
