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
        case live(
            stickyClientID: String,
            markedText: String,
            phase: ImePhase
        )
        case pendingTerminal(stickyClientID: String, action: PendingTerminalAction)
    }

    static let noClientID = "-"
    private static let finalizingSpinnerFrames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    private var state: DictationRouteState = .idle
    private weak var controller: BeeInputController?
    private var finalizingSpinnerFrameIndex = 0
    private var spinnerTask: Task<Void, Never>?

    // MARK: - Derived state

    private var stickyClientID: String? {
        switch state {
        case .idle:
            return nil
        case .live(let stickyClientID, _, _):
            return stickyClientID
        case .pendingTerminal(let stickyClientID, _):
            return stickyClientID
        }
    }

    private var currentMarkedText: String? {
        guard case .live(_, let markedText, _) = state else { return nil }
        return markedText
    }

    private var currentPhase: ImePhase? {
        guard case .live(_, _, let phase) = state else { return nil }
        return phase
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

        case .live(let stickyClientID, let markedText, let phase):
            guard normalizedClientID == stickyClientID else {
                beeInputLog(
                    "🚫 ACTIVATE ignored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) sticky=\(stickyClientID)"
                )
                return .none
            }

            beeInputLog(
                "🟡 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) replaying markedText phase=\(phase)"
            )
            replayMarkedText(markedText)
            startSpinnerIfNeeded()
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
                stopSpinner()
                deliverCommitText(text)
                state = .idle
            case .clear:
                beeInputLog(
                    "🟢 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) flushing delayed clear"
                )
                stopSpinner()
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

        case .live(let stickyClientID, let markedText, _):
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

    func setPhase(_ phase: ImePhase) {
        let normalizedCurrentClientID = activeClientID()

        switch state {
        case .idle:
            guard normalizedCurrentClientID != Self.noClientID else {
                beeInputLog(
                    "🚫 BLOCKED setPhase phase=\(phase) client=\(normalizedCurrentClientID) state=idle no active client"
                )
                return
            }

            if phase == .finalizing {
                finalizingSpinnerFrameIndex = 0
            } else {
                stopSpinner()
            }

            state = .live(
                stickyClientID: normalizedCurrentClientID,
                markedText: "",
                phase: phase
            )
            beeInputLog(
                "🟢 setPhase phase=\(phase) sticky claimed client=\(normalizedCurrentClientID)"
            )
            replayMarkedText("")
            startSpinnerIfNeeded()

        case .live(let stickyClientID, let markedText, let previousPhase):
            if phase == .finalizing && previousPhase != .finalizing {
                finalizingSpinnerFrameIndex = 0
            } else if phase == .dictating {
                stopSpinner()
            }

            state = .live(
                stickyClientID: stickyClientID,
                markedText: markedText,
                phase: phase
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) client=\(normalizedCurrentClientID) sticky=\(stickyClientID)"
                )
                replayMarkedText(markedText)
                startSpinnerIfNeeded()
            } else {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) route unavailable, storing for replay"
                )
                startSpinnerIfNeeded()
            }

        case .pendingTerminal(let stickyClientID, _):
            if phase == .finalizing {
                finalizingSpinnerFrameIndex = 0
            } else {
                stopSpinner()
            }

            state = .live(
                stickyClientID: stickyClientID,
                markedText: "",
                phase: phase
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) resuming live dictation"
                )
                replayMarkedText("")
                startSpinnerIfNeeded()
            } else {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) revived live dictation, waiting for sticky route"
                )
                startSpinnerIfNeeded()
            }
        }
    }

    func setMarkedText(_ text: String) {
        let normalizedCurrentClientID = activeClientID()

        if text.isEmpty {
            switch state {
            case .idle:
                beeInputLog(
                    "⏭️ setMarkedText empty ignored client=\(normalizedCurrentClientID) state=idle"
                )

            case .live(let stickyClientID, _, let phase):
                state = .live(
                    stickyClientID: stickyClientID,
                    markedText: "",
                    phase: phase
                )
                beeInputLog(
                    "🟡 setMarkedText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) phase=\(phase)"
                )
                if normalizedCurrentClientID == stickyClientID {
                    replayMarkedText("")
                }

            case .pendingTerminal(let stickyClientID, _):
                beeInputLog(
                    "⏭️ setMarkedText empty ignored client=\(normalizedCurrentClientID) sticky=\(stickyClientID) state=pendingTerminal"
                )
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

            state = .live(
                stickyClientID: normalizedCurrentClientID,
                markedText: text,
                phase: .dictating
            )
            beeInputLog(
                "🟢 setMarkedText text=\(text) sticky claimed client=\(normalizedCurrentClientID) phase=dictating"
            )
            replayMarkedText(text)
            startSpinnerIfNeeded()

        case .live(let stickyClientID, _, let phase):
            state = .live(
                stickyClientID: stickyClientID,
                markedText: text,
                phase: phase
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) phase=\(phase)"
                )
                replayMarkedText(text)
                startSpinnerIfNeeded()
            } else {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) phase=\(phase) route unavailable, storing for replay"
                )
            }

        case .pendingTerminal(let stickyClientID, _):
            state = .live(
                stickyClientID: stickyClientID,
                markedText: text,
                phase: .dictating
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) phase=dictating resuming live dictation"
                )
                replayMarkedText(text)
                startSpinnerIfNeeded()
            } else {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) phase=dictating revived live dictation, waiting for sticky route"
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

        case .live(let stickyClientID, _, _):
            let action: PendingTerminalAction = text.isEmpty ? .clear : .commit(text)
            state = .pendingTerminal(stickyClientID: stickyClientID, action: action)
            beeInputLog(
                text.isEmpty
                    ? "🟢 commitText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending clear"
                    : "🟢 commitText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending commit"
            )
            stopSpinner()
            flushPendingTerminalIfPossible()

        case .pendingTerminal(let stickyClientID, _):
            let action: PendingTerminalAction = text.isEmpty ? .clear : .commit(text)
            state = .pendingTerminal(stickyClientID: stickyClientID, action: action)
            beeInputLog(
                text.isEmpty
                    ? "🟢 commitText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending terminal action with clear"
                    : "🟢 commitText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending terminal action with commit"
            )
            stopSpinner()
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

        let renderedText = adorn(text)
        beeInputLog(
            "🟡 replayMarkedText phase=\(currentPhase.map(String.init(describing:)) ?? "nil") text=\(renderedText)"
        )
        controller?.client().setMarkedText(
            renderedText,
            selectionRange: NSRange(location: 0, length: (renderedText as NSString).length),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private func startSpinnerIfNeeded() {
        guard case .live(_, _, let phase) = state else {
            stopSpinner()
            return
        }
        guard phase == .finalizing else {
            stopSpinner()
            return
        }
        guard spinnerTask == nil else { return }

        spinnerTask = Task { @MainActor [weak self] in
            while let self, !Task.isCancelled {
                do {
                    try await Task.sleep(for: .milliseconds(80))
                } catch {
                    break
                }

                guard case .live(_, let currentMarkedText, let currentPhase) = self.state else {
                    break
                }
                guard currentPhase == .finalizing else { break }
                guard self.activeClientID() == self.stickyClientID else { continue }

                self.replayMarkedText(currentMarkedText)
            }

            self?.spinnerTask = nil
        }
    }

    private func stopSpinner() {
        spinnerTask?.cancel()
        spinnerTask = nil
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
            stopSpinner()

        case .idle, .live:
            break
        }
    }

    private func adorn(_ text: String) -> String {
        switch currentPhase {
        case .dictating:
            return text.isEmpty ? "🐝" : "\(text) 🐝"
        case .finalizing:
            let frame = Self.finalizingSpinnerFrames[
                finalizingSpinnerFrameIndex % Self.finalizingSpinnerFrames.count
            ]
            finalizingSpinnerFrameIndex += 1
            return text.isEmpty ? frame : "\(text) \(frame)"
        case .none:
            return text
        @unknown default:
            return text
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
