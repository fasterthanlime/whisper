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

    /// The last client we successfully set marked text on.
    /// Stashed because by the time deactivateServer runs, controller.client() is already swapped.
    var lastUsedClient: (any IMKTextInput & NSObjectProtocol)?

    /// The input context captured at activate time, for discardMarkedText() on context switch.
    weak var inputContext: NSTextInputContext?

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

        lastUsedClient = client
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
        case active(BeeIMESession, pendingText: String?)
    }

    private(set) var state: State = .idle

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

        // Clear composition by setting an empty marked string and then unmarking.
        if hasMarked {
            beeInputLog(
                "discardMarkedText[\(reason)]: cancel-style clear client=\(String(describing: type(of: client))) markedRange(before)=\(NSStringFromRange(markedRange)) selectedRange(before)=\(NSStringFromRange(client.selectedRange()))"
            )
        } else {
            beeInputLog(
                "discardMarkedText[\(reason)]: no marked range; clear via empty setMarkedText at caret"
            )
        }

        // IMPORTANT: Do not call insertText here — many hosts treat it as a commit.
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        // Some hosts require an explicit unmark to drop composition state without committing.
        if client.responds(to: Selector(("unmarkText"))) {
            (client as AnyObject).perform(Selector(("unmarkText")))
        }
        beeInputLog(
            "discardMarkedText[\(reason)]: markedRange(after)=\(NSStringFromRange(client.markedRange())) selectedRange(after)=\(NSStringFromRange(client.selectedRange()))"
        )
    }

    // MARK: - State transitions

    /// Returns true if this is a fresh activation (not just a controller update).
    @discardableResult
    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?, inputContext: NSTextInputContext?) -> Bool {
        if case .active(let session, let pending) = state {
            beeInputLog(
                "activate: already active, updating controller pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") pendingLen=\(pending?.utf16.count ?? 0)"
            )
            // Clear marked text via the stashed input context before it becomes unreachable
            if !session.currentMarkedText.isEmpty {
                if let ctx = session.inputContext {
                    beeInputLog("activate: discardMarkedText via stashed inputContext")
                    ctx.discardMarkedText()
                } else if let oldClient = session.lastUsedClient {
                    beeInputLog("activate: clearing stale composition on old client (no inputContext)")
                    oldClient.setMarkedText(
                        "",
                        selectionRange: NSRange(location: 0, length: 0),
                        replacementRange: NSRange(location: NSNotFound, length: 0))
                }
                session.currentMarkedText = ""
                session.lastUsedClient = nil
                session.inputContext = nil
            }
            session.controller = controller
            session.inputContext = inputContext
            return false
        }
        let session = BeeIMESession(controller: controller, pid: pid, clientID: clientID)
        session.inputContext = inputContext
        state = .active(session, pendingText: nil)
        beeInputLog(
            "state → active pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil")")
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
        if session.controller != nil {
            beeInputLog(
                "setMarkedText: delivering clientID=\(session.clientID ?? "nil") len=\(text.utf16.count)"
            )
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
            return
        }
        beeInputLog(
            "commitText: clientID=\(session.clientID ?? "nil") submit=\(submit) len=\(text.utf16.count)"
        )
        session.handleCommitText(text, submit: submit)
    }

    func cancelInput() {
        guard case .active(let session, _) = state else {
            beeInputLog("cancelInput: not active, dropping")
            return
        }
        if let client = session.controller?.client() {
            beeInputLog("cancelInput: discardMarkedText clientID=\(session.clientID ?? "nil")")
            discardMarkedText(on: client as AnyObject, reason: "cancel")
        }
        session.currentMarkedText = ""
        beeInputLog("cancelInput: done")
    }

    func replaceText(oldText: String, newText: String) {
        guard let session = currentSession else {
            beeInputLog("replaceText: no active session, dropping")
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
