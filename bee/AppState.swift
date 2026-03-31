import Foundation
import SwiftUI

// MARK: - UI Layer State Machine

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

    // Model
    var modelStatus: ModelStatus = .notLoaded

    // Input devices
    var availableInputDevices: [InputDeviceInfo] = []
    var activeInputDeviceUID: String?
    var activeInputDeviceName: String?
    var activeInputDeviceKeepWarm: Bool = false

    // History
    var transcriptionHistory: [TranscriptionHistoryItem] = []

    // ASR settings
    var chunkSizeSec: Float = 0.5
    var maxNewTokensStreaming: UInt32 = 0  // 0 = Rust default (32)
    var maxNewTokensFinal: UInt32 = 0      // 0 = Rust default (512)

    // Debug
    var debugEnabled = false
    var lastSessionDiag: SessionDiag.Snapshot?

    struct InputDeviceInfo: Sendable {
        let uid: String
        let name: String
        let isBuiltIn: Bool
        let isDefault: Bool
    }

    func selectInputDevice(uid: String) {
        activeInputDeviceUID = uid
        if let device = availableInputDevices.first(where: { $0.uid == uid }) {
            activeInputDeviceName = device.name
        }
        audioEngine.selectDevice(uid: uid)
    }

    func toggleActiveInputDeviceKeepWarm() {
        activeInputDeviceKeepWarm.toggle()
        if let uid = activeInputDeviceUID {
            audioEngine.deviceWarmPolicy[uid] = activeInputDeviceKeepWarm
        }
    }

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

    func handleROptDown() -> Bool {
        switch uiState {
        case .idle:
            let session = createSession()
            uiState = .pending(session)
            startPendingTimer(session: session)
            let config = TranscriptionService.SessionConfig(
                chunkSizeSec: chunkSizeSec,
                maxNewTokensStreaming: maxNewTokensStreaming,
                maxNewTokensFinal: maxNewTokensFinal
            )
            Task { await session.start(language: detectLanguage(), asrConfig: config) }
            return false // not swallowed

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
        case .pending(let session):
            pendingTimer?.cancel()
            uiState = .locked(session)
            playRecordingStartedSound()
            return false

        case .pushToTalk(let session):
            uiState = .idle
            Task { await session.commit(submit: false) }
            return true // swallowed

        case .lockedOptionHeld(let session):
            uiState = .idle
            Task { await session.commit(submit: false) }
            return true // swallowed

        default:
            return false
        }
    }

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
        case .pushToTalk(let session):
            uiState = .idle
            Task { await session.cancel() }
            return true // swallowed

        case .locked:
            return false // passthrough

        case .lockedOptionHeld(let session):
            uiState = .idle
            Task { await session.cancel() }
            return true // swallowed

        default:
            return false
        }
    }

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
            Task { await session.abort() }
            return false

        default:
            return false
        }
    }

    // MARK: - Pending Timer

    private func startPendingTimer(session: Session) {
        pendingTimer = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            if case .pending(let s) = uiState, s.id == session.id {
                uiState = .pushToTalk(s)
                playRecordingStartedSound()
            }
        }
    }

    // MARK: - Max Duration

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
        let session = Session(
            audioEngine: audioEngine,
            transcriptionService: transcriptionService,
            inputClient: inputClient,
            targetBundleID: bundleID
        )

        Task {
            await session.setOnComplete { [weak self] result in
                Task { @MainActor in
                    guard let self else { return }
                    self.lastSessionDiag = session.diag.snapshot
                    self.handleSessionResult(result)
                }
            }
        }

        return session
    }

    private func handleSessionResult(_ result: SessionResult) {
        switch result {
        case .aborted:
            break // no trace
        case .cancelled(_, let text):
            SoundEffects.shared.playCancel()
            if !text.isEmpty {
                addHistoryEntry(text: text)
            }
        case .committed(_, let text, _):
            SoundEffects.shared.playCommit()
            if !text.isEmpty {
                addHistoryEntry(text: text)
            }
        }
    }

    private func addHistoryEntry(text: String) {
        let item = TranscriptionHistoryItem(text: text)
        transcriptionHistory.insert(item, at: 0)
        if transcriptionHistory.count > 20 {
            transcriptionHistory = Array(transcriptionHistory.prefix(20))
        }
    }

    // MARK: - Model Loading

    static let defaultModel = STTModelDefinition.allModels.first(where: { $0.id == "qwen3-1.7b-mlx-4bit" })
        ?? STTModelDefinition.default

    func loadModelAtStartup() {
        let model = Self.defaultModel
        modelStatus = .loading
        SoundEffects.shared.warmUp()

        Task {
            // Request mic permission
            let micGranted = await AudioEngine.requestPermission()
            if !micGranted {
                await MainActor.run {
                    self.modelStatus = .error("Microphone permission denied")
                }
                return
            }

            do {
                try await transcriptionService.loadModel(
                    model: model,
                    cacheDir: STTModelDefinition.cacheDirectory
                )
                // Warm up audio engine after model loads
                try audioEngine.warmUp()

                await MainActor.run {
                    self.modelStatus = .loaded
                }
            } catch {
                await MainActor.run {
                    self.modelStatus = .error(error.localizedDescription)
                }
            }
        }
    }

    // MARK: - Stubs

    private func detectLanguage() -> String? {
        // TODO: walk AX tree, run NLLanguageRecognizer
        nil
    }

    private func playRecordingStartedSound() {
        SoundEffects.shared.playRecordingStarted()
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
