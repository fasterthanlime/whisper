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

    private static let heartbeatEmojis = ["🍎", "🍊", "🍋", "🍇", "🍓", "🫐", "🍑", "🍒", "🥝", "🍍"]
    private var heartbeatTimer: Timer?
    private var heartbeatIndex: Int = 0

    init(controller: BeeInputController, pid: pid_t?, clientID: String?, bundleID: String?) {
        self.controller = controller
        self.pid = pid
        self.clientID = clientID
        self.bundleID = bundleID
    }

    func startHeartbeat() {
        stopHeartbeat()
        heartbeatIndex = 0
        heartbeatTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.heartbeatTick()
            }
        }
        heartbeatTick() // fire immediately
    }

    func stopHeartbeat() {
        heartbeatTimer?.invalidate()
        heartbeatTimer = nil
    }

    private func heartbeatTick() {
        guard let client = controller?.client() else {
            beeInputLog("heartbeat: no client")
            return
        }
        let emoji = Self.heartbeatEmojis[heartbeatIndex % Self.heartbeatEmojis.count]
        heartbeatIndex += 1

        let attributed = NSAttributedString(
            string: emoji,
            attributes: [.markedClauseSegment: 0])
        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: emoji.utf16.count, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
        beeInputLog("heartbeat: \(emoji) (#\(heartbeatIndex)) clientID=\(clientID ?? "nil") bundle=\(client.bundleIdentifier() ?? "?") markedRange(after)=\(NSStringFromRange(client.markedRange()))")
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

    /// When we deactivate with marked text still in the client, record it
    /// so the next activate for the same bundle can attempt cleanup.
    struct PendingCleanup {
        let bundleID: String
        let markedTextUTF16Length: Int
    }
    private(set) var pendingCleanup: PendingCleanup? = nil

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
        // Check if we need to clean up marked text left behind in a previous session
        if let cleanup = pendingCleanup,
           let client = controller.client() as? (any IMKTextInput & NSObjectProtocol),
           client.bundleIdentifier() == cleanup.bundleID {
            let markedBefore = client.markedRange()
            let selectedBefore = client.selectedRange()

            beeInputLog(
                "activate: attempting cleanup for \(cleanup.bundleID) len=\(cleanup.markedTextUTF16Length) markedRange=\(NSStringFromRange(markedBefore)) selectedRange=\(NSStringFromRange(selectedBefore))"
            )

            // Try to clear via setMarkedText("")
            let empty = NSAttributedString(string: "", attributes: [.markedClauseSegment: 0])
            client.setMarkedText(
                empty,
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: NSNotFound, length: 0)
            )

            let markedAfter = client.markedRange()
            let selectedAfter = client.selectedRange()
            beeInputLog(
                "activate: cleanup result markedRange=\(NSStringFromRange(markedAfter)) selectedRange=\(NSStringFromRange(selectedAfter))"
            )

            pendingCleanup = nil
        }

        if case .active(let session, let pending) = state {
            beeInputLog(
                "activate: already active, updating controller pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") pendingLen=\(pending?.utf16.count ?? 0)"
            )
            session.controller = controller
            return false
        }
        let bundleID = (controller.client() as? (any IMKTextInput & NSObjectProtocol))?.bundleIdentifier()
        let session = BeeIMESession(controller: controller, pid: pid, clientID: clientID, bundleID: bundleID)
        state = .active(session, pendingText: nil)
        // E-004: heartbeat disabled to isolate cleanup experiment
        // session.startHeartbeat()
        beeInputLog(
            "state → active pid=\(pid.map(String.init) ?? "nil") clientID=\(clientID ?? "nil") bundle=\(bundleID ?? "nil")")
        return true
    }

    func deactivate(_ controller: BeeInputController) {
        guard activeController === controller else { return }
        let session = currentSession
        session?.stopHeartbeat()

        // If we're leaving marked text behind, record it for cleanup on reactivation
        // Use the session's stored bundleID (captured at activation time), NOT
        // controller.client() which may already point at the new app (F-013).
        let markedLen = session?.currentMarkedText.utf16.count ?? 0
        if markedLen > 0, let bundle = session?.bundleID {
            pendingCleanup = PendingCleanup(bundleID: bundle, markedTextUTF16Length: markedLen)
            beeInputLog("state → idle (pendingCleanup: bundle=\(bundle) len=\(markedLen))")
        } else {
            pendingCleanup = nil
            beeInputLog("state → idle")
        }

        state = .idle
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
        session.handleCommitText(text, submit: submit)
    }

    func cancelInput() {
        guard case .active(let session, _) = state else {
            beeInputLog("cancelInput: not active, dropping")
            return
        }
        session.currentMarkedText = ""
        session.handleSetMarkedText("")
        beeInputLog("cancelInput: done")
    }

    func replaceText(oldText: String, newText: String) {
        guard let session = currentSession else {
            beeInputLog("replaceText: no active session, dropping")
            return
        }
        session.handleReplaceText(oldText: oldText, newText: newText)
    }
}
