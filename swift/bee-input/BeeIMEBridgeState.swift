import AppKit
import Carbon
import Foundation
import InputMethodKit

// MARK: - Session

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
            beeInputLog("⚠️ DROP  handleSetMarkedText — no client")
            return
        }

        currentMarkedText = text
        beeInputLog(
            "📝 MARKED len=\(text.utf16.count) client=\(clientID ?? "?") \"\(text.prefix(80))\""
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
            beeInputLog("⚠️ DROP  handleCommitText — no client")
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
            "✅ COMMIT len=\(finalText.utf16.count) submit=\(submit) client=\(clientID ?? "?") \"\(finalText.prefix(80))\""
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
            beeInputLog("⚠️ DROP  handleReplaceText — no client")
            return
        }
        let sel = client.selectedRange()
        let oldWithSpace = oldText + " "
        let oldLen = oldWithSpace.utf16.count
        let replaceStart = sel.location >= oldLen ? sel.location - oldLen : 0
        let replaceRange = NSRange(location: replaceStart, length: oldLen)
        let newWithSpace = newText + " "
        beeInputLog(
            "🔄 REPLACE client=\(clientID ?? "?") \"\(oldWithSpace.prefix(60))\" → \"\(newWithSpace.prefix(60))\""
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

    /// Client IDs whose dictation ended while we couldn't reach them directly
    /// (e.g. weak ref released). On next `activate` for a matching client,
    /// clear any lingering marked text.
    private var staleOriginClientIDs: Set<String> = []

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
                "🟢 REACTIVATE updating controller pid=\(pid.map(String.init) ?? "-") client=\(clientID ?? "-") pendingLen=\(pending?.utf16.count ?? 0)"
            )
            session.controller = controller
            return false
        }
        let bundleID = controller.client()?.bundleIdentifier()
        let session = BeeIMESession(
            controller: controller, pid: pid, clientID: clientID, bundleID: bundleID)
        state = .active(session, pendingText: nil)
        beeInputLog(
            "🟢 ACTIVATE pid=\(pid.map(String.init) ?? "-") client=\(clientID ?? "-") bundle=\(bundleID ?? "-")"
        )
        return true
    }

    func deactivate(_ controller: BeeInputController) {
        guard activeController === controller else { return }
        state = .idle
        beeInputLog("🔴 DEACTIVATE → idle")
    }

    /// Returns `true` if this client had a dictation session that ended while
    /// unreachable. The caller should clear the client's marked text and
    /// this method removes the client from the stale set.
    func isStaleClient(_ clientID: String) -> Bool {
        let wasStale = staleOriginClientIDs.remove(clientID) != nil
        if wasStale {
            beeInputLog("🧹 STALE client=\(clientID)")
        }
        return wasStale
    }

    // MARK: - Text routing

    func flushPending() {
        guard case .active(let session, let text?) = state else { return }
        guard session.clientID == dictationOriginClientID else {
            beeInputLog(
                "🚫 BLOCKED flushPending — origin=\(dictationOriginClientID ?? "-") current=\(session.clientID ?? "-")"
            )
            return
        }
        beeInputLog(
            "📦 FLUSH len=\(text.utf16.count) → \"\(text.prefix(60))\""
        )
        state = .active(session, pendingText: nil)
        session.handleSetMarkedText(text)
    }

    func setMarkedText(_ text: String) {
        guard case .active(let session, _) = state else {
            beeInputLog("⚠️ DROP  setMarkedText — not active")
            return
        }

        // Capture origin on first setMarkedText of a dictation session
        if dictationOriginClientID == nil {
            dictationOriginClientID = session.clientID
            dictationOriginClient = session.controller?.client()
            beeInputLog("🎯 ORIGIN captured client=\(session.clientID ?? "-")")
        }

        // Only deliver to the original client
        if session.clientID != dictationOriginClientID {
            beeInputLog(
                "🚫 BLOCKED setMarkedText — origin=\(dictationOriginClientID ?? "-") current=\(session.clientID ?? "-")"
            )
            return
        }

        if session.controller != nil {
            session.handleSetMarkedText(text)
        } else {
            beeInputLog(
                "⏳ QUEUED setMarkedText len=\(text.utf16.count) (controller lost, client=\(session.clientID ?? "-"))"
            )
            state = .active(session, pendingText: text)
        }
    }

    func commitText(_ text: String, submit: Bool = false) {
        guard case .active(let session, _) = state else {
            beeInputLog("⚠️ DROP  commitText — not active")
            dictationOriginClientID = nil
            dictationOriginClient = nil
            return
        }
        guard session.clientID == dictationOriginClientID else {
            beeInputLog(
                "🚫 BLOCKED commitText — origin=\(dictationOriginClientID ?? "-") current=\(session.clientID ?? "-")"
            )
            dictationOriginClientID = nil
            dictationOriginClient = nil
            return
        }
        session.handleCommitText(text, submit: submit)
        // If weak ref was already gone, the origin client couldn't be cleared directly.
        // Register it for cleanup on next activate.
        if dictationOriginClient == nil, let cid = dictationOriginClientID {
            staleOriginClientIDs.insert(cid)
        }
        dictationOriginClientID = nil
        dictationOriginClient = nil
    }

    func cancelInput() {
        guard case .active(let session, _) = state else {
            beeInputLog("⚠️ DROP  cancelInput — not active")
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
            beeInputLog("🧹 CLEAR  marked text in origin client=\(dictationOriginClientID ?? "-")")
        } else if let cid = dictationOriginClientID {
            // Weak ref is gone — queue for cleanup on next activate.
            staleOriginClientIDs.insert(cid)
            beeInputLog("🧹 QUEUED stale clear for client=\(cid)")
        }
        if session.clientID == dictationOriginClientID {
            session.currentMarkedText = ""
        }
        dictationOriginClientID = nil
        dictationOriginClient = nil
        beeInputLog("🗑️ CANCEL  done, origin cleared")
    }

    func replaceText(oldText: String, newText: String) {
        guard let session = currentSession else {
            beeInputLog("⚠️ DROP  replaceText — no active session")
            return
        }
        guard session.clientID == dictationOriginClientID else {
            beeInputLog(
                "🚫 BLOCKED replaceText — origin=\(dictationOriginClientID ?? "-") current=\(session.clientID ?? "-")"
            )
            return
        }
        session.handleReplaceText(oldText: oldText, newText: newText)
    }
}
