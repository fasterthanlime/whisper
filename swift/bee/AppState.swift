import AppKit
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
    private static let imeSubmitName = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelName = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedName = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostName = NSNotification.Name("fasterthanlime.bee.imeContextLost")

    private(set) var uiState: UIState = .idle
    private var pendingTimer: Task<Void, Never>?
    private var consumeNextROptUp = false
    private var activeSessionID: UUID?
    private var activeSessionTargetPID: pid_t?
    private var distributedObservers: [NSObjectProtocol] = []
    private var workspaceObservers: [NSObjectProtocol] = []

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
        installExternalObservers()
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
            let targetPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let session = createSession(targetProcessID: targetPID)
            activeSessionID = session.id
            activeSessionTargetPID = targetPID
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
            transitionToIdle()
            Task { await session.commit(submit: false) }
            return true // swallowed

        case .lockedOptionHeld(let session):
            transitionToIdle()
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
            transitionToIdle()
            Task { await session.cancel() }
            return true // swallowed

        case .locked:
            return false // passthrough

        case .lockedOptionHeld(let session):
            transitionToIdle()
            Task { await session.cancel() }
            return true // swallowed

        default:
            return false
        }
    }

    func handleEnter() -> Bool {
        switch uiState {
        case .locked(let session):
            transitionToIdle()
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
            transitionToIdle()

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
                transitionToIdle()
                await session.commit(submit: false)
            }
        }
    }

    // MARK: - Session Factory

    private func createSession(targetProcessID: pid_t?) -> Session {
        let session = Session(
            audioEngine: audioEngine,
            transcriptionService: transcriptionService,
            inputClient: inputClient,
            targetProcessID: targetProcessID
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
        let resultID: UUID
        switch result {
        case .aborted(let id):
            resultID = id
            break // no trace
        case .cancelled(let id, let text):
            resultID = id
            SoundEffects.shared.playCancel()
            if !text.isEmpty {
                addHistoryEntry(text: text)
            }
        case .committed(let id, let text, _):
            resultID = id
            SoundEffects.shared.playCommit()
            if !text.isEmpty {
                addHistoryEntry(text: text)
            }
        }

        if activeSessionID == resultID {
            activeSessionID = nil
            activeSessionTargetPID = nil
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

    // MARK: - External Events

    private func installExternalObservers() {
        let dnc = DistributedNotificationCenter.default()
        let nc = NSWorkspace.shared.notificationCenter

        distributedObservers.append(
            dnc.addObserver(forName: Self.imeSubmitName, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.handleIMESubmit()
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeCancelName, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.handleIMECancel()
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeUserTypedName, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.handleIMEUserTyped()
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeContextLostName, object: nil, queue: .main) { [weak self] _ in
                Task { @MainActor in
                    self?.handleIMEContextLost()
                }
            }
        )
        workspaceObservers.append(
            nc.addObserver(
                forName: NSWorkspace.didActivateApplicationNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let activatedPID = (
                    notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
                )?.processIdentifier
                Task { @MainActor in
                    self?.handleDidActivateApplication(processIdentifier: activatedPID)
                }
            }
        )
    }

    private func handleIMESubmit() {
        switch uiState {
        case .pending(let session):
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session), .locked(let session), .lockedOptionHeld(let session):
            transitionToIdle()
            Task { await session.commit(submit: true) }
        case .idle:
            break
        }
    }

    private func handleIMECancel() {
        switch uiState {
        case .pending(let session):
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session), .locked(let session), .lockedOptionHeld(let session):
            transitionToIdle()
            Task { await session.cancel() }
        case .idle:
            break
        }
    }

    private func handleIMEUserTyped() {
        switch uiState {
        case .pending(let session):
            inputClient.stopDictating()
            inputClient.clearMarkedText()
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session), .locked(let session), .lockedOptionHeld(let session):
            inputClient.stopDictating()
            inputClient.clearMarkedText()
            transitionToIdle()
            Task { await session.abort() }
        case .idle:
            break
        }
    }

    private func handleIMEContextLost() {
        switch uiState {
        case .pending(let session):
            inputClient.stopDictating()
            inputClient.clearMarkedText()
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session), .locked(let session), .lockedOptionHeld(let session):
            inputClient.stopDictating()
            inputClient.clearMarkedText()
            transitionToIdle()
            Task { await session.abort() }
        case .idle:
            break
        }
    }

    private func handleDidActivateApplication(processIdentifier: pid_t?) {
        guard let session = uiState.session else { return }
        guard activeSessionID == session.id else { return }
        guard let targetPID = activeSessionTargetPID else { return }
        guard let processIdentifier else { return }
        guard processIdentifier != targetPID else { return }

        beeLog("SESSION: active app changed targetPID=\(targetPID) newPID=\(processIdentifier), cancelling")
        inputClient.stopDictating()
        inputClient.clearMarkedText()
        transitionToIdle()
        Task { await session.abort() }
    }

    private func transitionToIdle() {
        uiState = .idle
        pendingTimer?.cancel()
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
