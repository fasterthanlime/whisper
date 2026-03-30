import Foundation
import SwiftUI

// MARK: - UI Layer State Machine

// h[impl ui.swallow-policy]
/// The UI layer handles keyboard events and manages the status indicator.
/// It creates sessions and tells them how to end.
///
/// Event handlers return `true` if the event should be swallowed (the app
/// never sees it), `false` if it should pass through.
@Observable
@MainActor
final class AppState {
    private(set) var uiState: UIState = .idle
    private var pendingTimer: Task<Void, Never>?
    private var consumeNextROptUp = false

    // Shared infrastructure
    let audioEngine: AudioEngine
    let transcriptionService: TranscriptionService
    let inputClient: BeeInputClient

    // Menu bar state
    var modelStatus: ModelStatus = .notLoaded
    var selectedModelID: String = ""

    init(
        audioEngine: AudioEngine,
        transcriptionService: TranscriptionService,
        inputClient: BeeInputClient
    ) {
        self.audioEngine = audioEngine
        self.transcriptionService = transcriptionService
        self.inputClient = inputClient
    }

    // MARK: - State

    enum UIState {
        case idle
        case pending(Session)
        case pushToTalk(Session)
        case locked(Session)
        case lockedOptionHeld(Session)

        var session: Session? {
            switch self {
            case .idle: nil
            case .pending(let s): s
            case .pushToTalk(let s): s
            case .locked(let s): s
            case .lockedOptionHeld(let s): s
            }
        }

        var isRecording: Bool {
            switch self {
            case .idle: false
            default: true
            }
        }
    }

    // MARK: - Event Handlers

    // h[impl ui.idle-to-pending]
    func handleROptDown() -> Bool {
        switch uiState {
        case .idle:
            let session = createSession()
            uiState = .pending(session)
            startPendingTimer(session: session)
            Task { await session.start(language: detectLanguage()) }
            return false // not swallowed

        // h[impl ui.locked-option-down]
        case .locked(let session):
            uiState = .lockedOptionHeld(session)
            return true // swallowed

        default:
            return false
        }
    }

    func handleROptUp() -> Bool {
        if consumeNextROptUp {
            consumeNextROptUp = false
            return true // swallowed (after RCmd → Locked transition)
        }

        switch uiState {
        // h[impl ui.pending-to-locked]
        case .pending(let session):
            pendingTimer?.cancel()
            uiState = .locked(session)
            // h[impl sounds.recording-started]
            playRecordingStartedSound()
            return false

        // h[impl ui.ptt-commit]
        case .pushToTalk(let session):
            uiState = .idle
            Task { await session.commit(submit: false) }
            return true // swallowed

        // h[impl ui.locked-option-held-commit]
        case .lockedOptionHeld(let session):
            uiState = .idle
            Task { await session.commit(submit: false) }
            return true // swallowed

        default:
            return false
        }
    }

    // h[impl ui.ptt-to-locked]
    func handleRCmdDown() -> Bool {
        switch uiState {
        case .pushToTalk(let session):
            uiState = .locked(session)
            consumeNextROptUp = true
            return true // swallowed

        default:
            return false
        }
    }

    func handleEscape() -> Bool {
        switch uiState {
        // h[impl ui.ptt-cancel]
        case .pushToTalk(let session):
            uiState = .idle
            Task { await session.cancel() }
            return true // swallowed

        // h[impl ui.locked-esc-passthrough]
        case .locked:
            return false // passthrough

        // h[impl ui.locked-option-held-cancel]
        case .lockedOptionHeld(let session):
            uiState = .idle
            Task { await session.cancel() }
            return true // swallowed

        default:
            return false
        }
    }

    // h[impl ui.locked-enter]
    func handleEnter() -> Bool {
        switch uiState {
        case .locked(let session):
            uiState = .idle
            Task { await session.commit(submit: true) }
            return true // swallowed

        default:
            return false
        }
    }

    // h[impl ui.pending-abort]
    // h[impl history.paste-last]
    func handleOtherKey(keyCode: UInt16) -> Bool {
        switch uiState {
        case .pending(let session):
            pendingTimer?.cancel()
            uiState = .idle

            // ROpt+P = paste last history entry
            if keyCode == 0x23 /* kVK_ANSI_P */ {
                pasteLastHistoryEntry()
                Task { await session.abort() }
                return true // swallowed
            }

            // Spurious activation — abort silently, let the key through
            // h[impl ui.spurious-passthrough]
            Task { await session.abort() }
            return false

        default:
            return false
        }
    }

    // MARK: - Pending Timer

    // h[impl ui.pending-to-ptt]
    private func startPendingTimer(session: Session) {
        pendingTimer = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            if case .pending(let s) = uiState, s.id == session.id {
                uiState = .pushToTalk(s)
                // h[impl sounds.recording-started]
                playRecordingStartedSound()
            }
        }
    }

    // MARK: - Max Duration

    // h[impl ui.max-duration]
    private func startMaxDurationTimer(session: Session) {
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(300))
            guard !Task.isCancelled else { return }
            if uiState.session?.id == session.id {
                uiState = .idle
                await session.commit(submit: false)
            }
        }
    }

    // MARK: - Session Factory

    private func createSession() -> Session {
        let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        return Session(
            audioEngine: audioEngine,
            transcriptionService: transcriptionService,
            inputClient: inputClient,
            targetBundleID: bundleID
        )
    }

    // MARK: - Stubs

    private func detectLanguage() -> String? {
        // h[impl lang.detect-from-ax]
        // TODO: walk AX tree, run NLLanguageRecognizer
        nil
    }

    private func playRecordingStartedSound() {
        // TODO: play tink sound
    }

    private func pasteLastHistoryEntry() {
        // TODO: look up most recent history entry, paste via IME
    }
}

// MARK: - Model Status

enum ModelStatus: Equatable {
    case notLoaded
    case downloading(progress: Double)
    case loading
    case loaded
    case error(String)
}
