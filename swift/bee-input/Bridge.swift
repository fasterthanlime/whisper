import AppKit
import Carbon
import Foundation
import InputMethodKit

// MARK: - Bridge State

/// Tracks which client owns the current dictation route and renders marked text
/// locally inside the IME, including dictation/finalization adornments and text
/// animation.
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
            targetText: String,
            displayedText: String,
            phase: ImePhase
        )
        case pendingTerminal(stickyClientID: String, action: PendingTerminalAction)
    }

    static let noClientID = "-"

    private static let finalizingSpinnerFrames = [
        "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
    ]

    private let animationMorphSpeed: Float = 1.0
    private let animationAppendSpeed: Float = 1.0

    private var state: DictationRouteState = .idle
    private weak var controller: BeeInputController?

    private var finalizingSpinnerFrameIndex = 0
    private var spinnerTask: Task<Void, Never>?
    private var textAnimationTask: Task<Void, Never>?

    // MARK: - Derived state

    private var stickyClientID: String? {
        switch state {
        case .idle:
            return nil
        case .live(let stickyClientID, _, _, _):
            return stickyClientID
        case .pendingTerminal(let stickyClientID, _):
            return stickyClientID
        }
    }

    private var currentTargetText: String? {
        guard case .live(_, let targetText, _, _) = state else { return nil }
        return targetText
    }

    private var currentDisplayedText: String? {
        guard case .live(_, _, let displayedText, _) = state else { return nil }
        return displayedText
    }

    private var currentPhase: ImePhase? {
        guard case .live(_, _, _, let phase) = state else { return nil }
        return phase
    }

    private var isStickyRouteAvailable: Bool {
        activeClientID() == stickyClientID
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

        case .live(let stickyClientID, let targetText, _, let phase):
            guard normalizedClientID == stickyClientID else {
                beeInputLog(
                    "🚫 ACTIVATE ignored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) sticky=\(stickyClientID)"
                )
                return .none
            }

            beeInputLog(
                "🟡 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) phase=\(phase) replaying display"
            )
            renderCurrentDisplay()
            restartTextAnimationIfNeeded()
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
                stopAnimations()
                deliverCommitText(text)
                state = .idle
            case .clear:
                beeInputLog(
                    "🟢 ACTIVATE sticky restored pid=\(pid.map(String.init) ?? "nil") client=\(normalizedClientID) bundle=\(bundleID) flushing delayed clear"
                )
                stopAnimations()
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
            beeInputLog("⏭️ DEACTIVATE idle client=\(normalizedClientID)")
            return .none

        case .live(let stickyClientID, let targetText, _, _):
            if normalizedClientID == stickyClientID {
                beeInputLog(
                    "🟡 DEACTIVATE sticky client=\(normalizedClientID) targetTextLen=\((targetText as NSString).length) keeping route sticky"
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

            prepareForPhaseChange(from: nil, to: phase)
            state = .live(
                stickyClientID: normalizedCurrentClientID,
                targetText: "",
                displayedText: "",
                phase: phase
            )
            beeInputLog(
                "🟢 setPhase phase=\(phase) sticky claimed client=\(normalizedCurrentClientID)"
            )
            renderCurrentDisplay()
            restartTextAnimationIfNeeded()
            startSpinnerIfNeeded()

        case .live(let stickyClientID, let targetText, let displayedText, let previousPhase):
            prepareForPhaseChange(from: previousPhase, to: phase)
            state = .live(
                stickyClientID: stickyClientID,
                targetText: targetText,
                displayedText: displayedText,
                phase: phase
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) client=\(normalizedCurrentClientID) sticky=\(stickyClientID)"
                )
                renderCurrentDisplay()
                restartTextAnimationIfNeeded()
                startSpinnerIfNeeded()
            } else {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) route unavailable, storing for replay"
                )
                startSpinnerIfNeeded()
            }

        case .pendingTerminal(let stickyClientID, _):
            prepareForPhaseChange(from: nil, to: phase)
            state = .live(
                stickyClientID: stickyClientID,
                targetText: "",
                displayedText: "",
                phase: phase
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setPhase phase=\(phase) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) resuming live dictation"
                )
                renderCurrentDisplay()
                restartTextAnimationIfNeeded()
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
                targetText: text,
                displayedText: "",
                phase: .dictating
            )
            beeInputLog(
                "🟢 setMarkedText text=\(text) sticky claimed client=\(normalizedCurrentClientID) phase=dictating"
            )
            restartTextAnimationIfNeeded()
            startSpinnerIfNeeded()

        case .live(let stickyClientID, _, let displayedText, let phase):
            state = .live(
                stickyClientID: stickyClientID,
                targetText: text,
                displayedText: displayedText,
                phase: phase
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) phase=\(phase)"
                )
                restartTextAnimationIfNeeded()
                startSpinnerIfNeeded()
            } else {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) sticky=\(stickyClientID) currentClient=\(normalizedCurrentClientID) phase=\(phase) route unavailable, storing for replay"
                )
            }

        case .pendingTerminal(let stickyClientID, _):
            state = .live(
                stickyClientID: stickyClientID,
                targetText: text,
                displayedText: "",
                phase: .dictating
            )

            if normalizedCurrentClientID == stickyClientID {
                beeInputLog(
                    "🟡 setMarkedText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) phase=dictating resuming live dictation"
                )
                restartTextAnimationIfNeeded()
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

        case .live(let stickyClientID, _, _, _):
            let action: PendingTerminalAction = text.isEmpty ? .clear : .commit(text)
            state = .pendingTerminal(stickyClientID: stickyClientID, action: action)
            beeInputLog(
                text.isEmpty
                    ? "🟢 commitText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending clear"
                    : "🟢 commitText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) pending commit"
            )
            stopAnimations()
            flushPendingTerminalIfPossible()

        case .pendingTerminal(let stickyClientID, _):
            let action: PendingTerminalAction = text.isEmpty ? .clear : .commit(text)
            state = .pendingTerminal(stickyClientID: stickyClientID, action: action)
            beeInputLog(
                text.isEmpty
                    ? "🟢 commitText empty client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending terminal action with clear"
                    : "🟢 commitText text=\(text) client=\(normalizedCurrentClientID) sticky=\(stickyClientID) replacing pending terminal action with commit"
            )
            stopAnimations()
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

    private func prepareForPhaseChange(from oldPhase: ImePhase?, to newPhase: ImePhase) {
        if newPhase == .finalizing && oldPhase != .finalizing {
            finalizingSpinnerFrameIndex = 0
        }
        if newPhase == .dictating {
            stopSpinner()
        }
    }

    private func stopAnimations() {
        stopSpinner()
        stopTextAnimation()
    }

    private func stopTextAnimation() {
        textAnimationTask?.cancel()
        textAnimationTask = nil
    }

    private func restartTextAnimationIfNeeded() {
        guard case .live(_, let targetText, let displayedText, _) = state else {
            stopTextAnimation()
            return
        }

        if displayedText == targetText {
            if isStickyRouteAvailable {
                renderCurrentDisplay()
            }
            stopTextAnimation()
            return
        }

        guard isStickyRouteAvailable else {
            return
        }

        stopTextAnimation()
        textAnimationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            await self.runTextAnimation()
        }
    }

    private func runTextAnimation() async {
        defer { textAnimationTask = nil }

        while !Task.isCancelled {
            guard
                case .live(let stickyClientID, let targetText, let displayedText, let phase) = state
            else { return }
            guard activeClientID() == stickyClientID else { return }
            guard displayedText != targetText else {
                renderCurrentDisplay()
                return
            }

            let nextText = nextAnimatedText(from: displayedText, to: targetText)
            state = .live(
                stickyClientID: stickyClientID,
                targetText: targetText,
                displayedText: nextText,
                phase: phase
            )
            renderCurrentDisplay()

            let didAppend = nextText.count > displayedText.count
            let speed = didAppend ? animationAppendSpeed : animationMorphSpeed
            if speed <= 0 {
                state = .live(
                    stickyClientID: stickyClientID,
                    targetText: targetText,
                    displayedText: targetText,
                    phase: phase
                )
                renderCurrentDisplay()
                return
            }

            let baseMs = nextText.count > displayedText.count ? 18 : 25
            let delayMs = max(1, Int(Float(baseMs) / speed))

            do {
                try await Task.sleep(for: .milliseconds(delayMs))
            } catch {
                return
            }
        }
    }

    private func nextAnimatedText(from old: String, to target: String) -> String {
        if old == target { return target }

        var chars = Array(old)
        let targetChars = Array(target)

        let canAppend = chars.count < targetChars.count
        let canTrim = chars.count > targetChars.count
        let wrongIndices = (0..<min(chars.count, targetChars.count)).filter {
            chars[$0] != targetChars[$0]
        }

        if wrongIndices.isEmpty && !canAppend && !canTrim {
            return target
        }

        let morphWeight = wrongIndices.count
        let appendWeight = canAppend ? max(1, wrongIndices.count / 2) : 0
        let trimWeight = canTrim ? max(1, wrongIndices.count / 2) : 0
        let total = morphWeight + appendWeight + trimWeight
        let roll = Int.random(in: 0..<max(total, 1))

        if roll < morphWeight && !wrongIndices.isEmpty {
            let idx = wrongIndices.randomElement()!
            chars[idx] = targetChars[idx]
            return String(chars)
        }

        if roll < morphWeight + appendWeight && canAppend {
            chars.append(targetChars[chars.count])
            return String(chars)
        }

        if canTrim {
            chars.removeLast()
            return String(chars)
        }

        if canAppend {
            chars.append(targetChars[chars.count])
            return String(chars)
        }

        if !wrongIndices.isEmpty {
            let idx = wrongIndices.randomElement()!
            chars[idx] = targetChars[idx]
            return String(chars)
        }

        return target
    }

    private func renderCurrentDisplay() {
        guard let renderedText = renderedMarkedText() else { return }
        guard isStickyRouteAvailable else { return }

        beeInputLog(
            "🟡 renderCurrentDisplay phase=\(currentPhase.map(String.init(describing:)) ?? "nil") text=\(renderedText)"
        )
        controller?.client().setMarkedText(
            renderedText,
            selectionRange: NSRange(location: 0, length: (renderedText as NSString).length),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    private func renderedMarkedText() -> String? {
        guard let displayedText = currentDisplayedText else { return nil }
        switch currentPhase {
        case .dictating:
            return displayedText.isEmpty ? "🐝" : "\(displayedText) 🐝"
        case .finalizing:
            let frame = Self.finalizingSpinnerFrames[
                finalizingSpinnerFrameIndex % Self.finalizingSpinnerFrames.count
            ]
            return displayedText.isEmpty ? frame : "\(displayedText) \(frame)"
        case .none:
            return nil
        @unknown default:
            return displayedText
        }
    }

    private func startSpinnerIfNeeded() {
        guard case .live(_, _, _, let phase) = state else {
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

                guard case .live(let stickyClientID, _, _, let phase) = self.state else {
                    break
                }
                guard phase == .finalizing else { break }
                guard self.activeClientID() == stickyClientID else { continue }

                self.finalizingSpinnerFrameIndex =
                    (self.finalizingSpinnerFrameIndex + 1) % Self.finalizingSpinnerFrames.count
                self.renderCurrentDisplay()
            }

            self?.spinnerTask = nil
        }
    }

    private func stopSpinner() {
        spinnerTask?.cancel()
        spinnerTask = nil
    }

    private func deliverCommitText(_ text: String) {
        guard isStickyRouteAvailable else {
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
        guard isStickyRouteAvailable else {
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
        guard isStickyRouteAvailable else {
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
            stopAnimations()

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
