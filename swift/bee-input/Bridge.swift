import AppKit
import Carbon
import Foundation
import InputMethodKit

// MARK: - Bridge State

/// Tracks which client owns the current dictation route and replays/flushes
/// pending text when that same client becomes reachable again.
@MainActor
final class Bridge: NSObject {
    static let shared = Bridge()

    enum ActivationEvent {
        case none
        case stickyRouteRestored
        case delayedTerminalFlushed
    }

    enum DeactivationEvent {
        case none
        case stickyRouteUnavailable(hadMarkedText: Bool)
    }

    private enum PendingTerminalAction {
        case commit(String)
        case clear
    }

    private enum DictationRouteState {
        case idle
        case live(stickyClientID: String, markedText: String)
        case pendingTerminal(stickyClientID: String, action: PendingTerminalAction)
    }

    static let noClientID = "-"

    private var state: DictationRouteState = .idle
    private weak var controller: BeeInputController?

    // MARK: - Derived state

    private var stickyClientID: String? {
        switch state {
        case .idle:
            return nil
        case .live(let stickyClientID, _):
            return stickyClientID
        case .pendingTerminal(let stickyClientID, _):
            return stickyClientID
        }
    }

    private var currentMarkedText: String? {
        guard case .live(_, let markedText) = state else { return nil }
        return markedText
    }

    // MARK: - State transitions

    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?)
        -> ActivationEvent
    {
        self.controller = controller

        let normalizedClientID = Self.normalizeClientID(clientID)
        let bundleID = Self.normalizeClientID(controller.client().bundleIdentifier())

        switch state {
        case .idle:
            beeInputLog(
                "🟢 ACTIVATE idle pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) waiting for sticky owner"
            )
            return .none

        case .live(let stickyClientID, let markedText):
            guard normalizedClientID == stickyClientID else {
                beeInputLog(
                    "🚫 ACTIVATE ignored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) sticky=\(stickyClientID)"
                )
                return .none
            }

            beeInputLog(
                "🟡 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) replaying markedText"
            )
            replayMarkedText(markedText)
            return .stickyRouteRestored

        case .pendingTerminal(let stickyClientID, let action):
            guard normalizedClientID == stickyClientID else {
                beeInputLog(
                    "🚫 ACTIVATE ignored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) sticky=\(stickyClientID) pending terminal action"
                )
                return .none
            }

            switch action {
            case .commit(let text):
                beeInputLog(
                    "🟢 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) flushing delayed commit"
                )
                deliverCommitText(text)
                state = .idle
            case .clear:
                beeInputLog(
                    "🟢 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) flushing delayed clear"
                )
                deliverClearMarkedText()
                state = .idle
            }
            return .delayedTerminalFlushed
        }
    }

    func deactivate(_ controller: BeeInputController, clientID: String?) -> DeactivationEvent {
        let normalizedClientID = Self.normalizeClientID(clientID)

        switch state {
        case .idle:
            beeInputLog(
                "⏭️ DEACTIVATE idle client=\(normalizedClientID)"
            )
            return .none

        case .live(let stickyClientID, let markedText):
            if normalizedClientID == stickyClientID {
                beeInputLog(
                    "🟡 DEACTIVATE sticky client=\(normalizedClientID) markedTextLen=\((markedText as NSString).length) keeping route sticky"
                )
                return .stickyRouteUnavailable(hadMarkedText: true)
            } else {
                beeInputLog(
                    "🚫 DEACTIVATE ignored client=\(normalizedClientID) sticky=\(stickyClientID)"
                )
                return .none
            }

        case .pendingTerminal(let stickyClientID, let action):
            if normalizedClientID == stickyClientID {
                beeInputLog(
                    "🟡 DEACTIVATE sticky client=\(normalizedClientID) pending=\(describe(action)) waiting to flush"
                )
                return .stickyRouteUnavailable(hadMarkedText: false)
            } else {
                beeInputLog(
                    "🚫 DEACTIVATE ignored client=\(normalizedClientID) sticky=\(stickyClientID) pending=\(describe(action))"
                )
                return .none
            }
        }
    }

    // MARK: - Text routing

    func setMarkedText(_ text: String) {
        let normalizedCurrentClientID = activeClientID()

        if text.isEmpty {
            switch state {
            case .idle:
                beeInputLog(
                    "⏭️ setMarkedText empty ignored client=\(normalizedCurrentClientID) state=idle"
                )

            case .live(let stickyClientID, _):
                state = .pendingTerminal(stickyClientID: stickyClientID, action: .clear)
                beeInputLog(
                    "🟡 setMarkedText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending clear"
                )
                flushPendingTerminalIfPossible()

            case .pendingTerminal(let stickyClientID, _):
                state = .pendingTerminal(stickyClientID: stickyClientID, action: .clear)
                beeInputLog(
                    "🟡 setMarkedText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending action with clear"
                )
                flushPendingTerminalIfPossible()
            }
            return
        }

        switch state {
        case .idle:
            guard normalizedCurrentClientID != Self.noClientID else {
                beeInputLog(
                    "🚫 BLOCKED setMarkedText text=\(text) client=\(normalizedCurrentClientID) state=idle no active client"
                )
                return
            }

            state = .live(stickyClientID: normalizedCurrentClientID, markedText: text)
            beeInputLog(
                "🟢 setMarkedText text=\(text) sticky claimed client=\(normalizedCurrentClientID)"
            )
            replayMarkedText(text)

        case .live(let stickyClientID, _):
            if normalizedCurrentClientID == stickyClientID {
                state = .live(stickyClientID: stickyClientID, markedText: text)
                beeInputLog(
                    "🟡 setMarkedText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID)"
                )
                replayMarkedText(text)
            } else {
                state = .live(stickyClientID: stickyClientID, markedText: text)
                beeInputLog(
                    "🟡 setMarkedText text=\(text) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) route unavailable, storing for replay"
                )
            }

        case .pendingTerminal(let stickyClientID, _):
            if normalizedCurrentClientID == stickyClientID {
                state = .live(stickyClientID: stickyClientID, markedText: text)
                beeInputLog(
                    "🟡 setMarkedText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) resuming live dictation"
                )
                replayMarkedText(text)
            } else {
                state = .live(stickyClientID: stickyClientID, markedText: text)
                beeInputLog(
                    "🟡 setMarkedText text=\(text) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) revived live dictation, waiting for sticky route"
                )
            }
        }
    }

    func commitText(_ text: String) {
        let normalizedCurrentClientID = activeClientID()

        switch state {
        case .idle:
            beeInputLog(
                "⏭️ commitText ignored text=\(text) client=\(normalizedCurrentClientID) state=idle"
            )

        case .live(let stickyClientID, _):
            let action: PendingTerminalAction = text.isEmpty ? .clear : .commit(text)
            state = .pendingTerminal(stickyClientID: stickyClientID, action: action)
            beeInputLog(
                text.isEmpty
                    ? "🟢 commitText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending clear"
                    : "🟢 commitText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending commit"
            )
            flushPendingTerminalIfPossible()

        case .pendingTerminal(let stickyClientID, _):
            let action: PendingTerminalAction = text.isEmpty ? .clear : .commit(text)
            state = .pendingTerminal(stickyClientID: stickyClientID, action: action)
            beeInputLog(
                text.isEmpty
                    ? "🟢 commitText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending terminal action with clear"
                    : "🟢 commitText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending terminal action with commit"
            )
            flushPendingTerminalIfPossible()
        }
    }

    // MARK: - Helpers

    private func activeClientID() -> String {
        Self.normalizeClientID(controller?.client().bundleIdentifier())
    }

    private static func normalizeClientID(_ clientID: String?) -> String {
        guard let clientID, !clientID.isEmpty else { return noClientID }
        return clientID
    }

    private func replayMarkedText(_ text: String) {
        guard activeClientID() == stickyClientID else {
            beeInputLog(
                "🚫 BLOCKED replayMarkedText client=\(activeClientID()) sticky=\(stickyClientID ?? Self.noClientID)"
            )
            return
        }

        beeInputLog("🟡 replayMarkedText text=\(text)")
        controller?.client().setMarkedText(
            text,
            selectionRange: NSRange(location: 0, length: (text as NSString).length),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private func deliverCommitText(_ text: String) {
        guard activeClientID() == stickyClientID else {
            beeInputLog(
                "🚫 BLOCKED commitText delivery client=\(activeClientID()) sticky=\(stickyClientID ?? Self.noClientID)"
            )
            return
        }

        beeInputLog("🟢 deliverCommitText text=\(text)")
        controller?.client().insertText(
            text,
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private func deliverClearMarkedText() {
        guard activeClientID() == stickyClientID else {
            beeInputLog(
                "🚫 BLOCKED clear delivery client=\(activeClientID()) sticky=\(stickyClientID ?? Self.noClientID)"
            )
            return
        }

        beeInputLog("🟢 deliverClearMarkedText")
        controller?.client().setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private func flushPendingTerminalIfPossible() {
        guard activeClientID() == stickyClientID else {
            beeInputLog(
                "🟡 pending terminal retained client=\(activeClientID()) sticky=\(stickyClientID ?? Self.noClientID)"
            )
            return
        }

        switch state {
        case .pendingTerminal(_, let action):
            switch action {
            case .commit(let text):
                deliverCommitText(text)
            case .clear:
                deliverClearMarkedText()
            }
            state = .idle

        case .idle, .live:
            break
        }
    }

    private func describe(_ action: PendingTerminalAction) -> String {
        switch action {
        case .commit(let text):
            return "commit(len=\((text as NSString).length))"
        case .clear:
            return "clear"
        }
    }
}
