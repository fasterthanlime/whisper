import AppKit
import AVFoundation
import Foundation
import SwiftUI
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "AppState")

// MARK: - UI Layer State Machine

/// The UI layer handles keyboard events and manages the status indicator.
/// It creates sessions and tells them how to end.
///
/// Event handlers return `true` if the event should be swallowed (the app
/// never sees it), `false` if it should pass through.
@Observable
@MainActor
final class AppState {
    private enum DefaultsKey {
        static let selectedInputDeviceUID = "audio.selectedInputDeviceUID"
        static let deviceWarmPolicy = "audio.deviceWarmPolicy"
    }

    private static let imeSubmitName = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelName = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedName = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostName = NSNotification.Name("fasterthanlime.bee.imeContextLost")
    private static let imeSessionStartedName = NSNotification.Name("fasterthanlime.bee.imeSessionStarted")

    private(set) var uiState: UIState = .idle
    private var pendingTimer: Task<Void, Never>?
    private var consumeNextROptUp = false
    private var activeSessionID: UUID?
    private var activeSessionTargetPID: pid_t?
    fileprivate var activeSessionTargetAppName: String?
    fileprivate var activeSessionTargetAppIcon: NSImage?
    private var activeSessionIMEConfirmed = false
    private var pendingLockRequest = false
    private var loggedWaitingForIME = false
    private var pendingIMEAckTimeoutTask: Task<Void, Never>?
    private var isSessionParked = false
    private var distributedObservers: [NSObjectProtocol] = []
    private var workspaceObservers: [NSObjectProtocol] = []
    private var captureDeviceObservers: [NSObjectProtocol] = []
    private var lastKnownInputDeviceUIDs: Set<String> = []
    private var pendingAudioReconfigureAfterSession = false
    private var parkedOverlayPanel: NSPanel?
    private var parkedOverlayPollTask: Task<Void, Never>?

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
    var parkedOverlayText = ""

    struct InputDeviceInfo: Sendable {
        let uid: String
        let name: String
        let isBuiltIn: Bool
        let isDefault: Bool
    }

    func selectInputDevice(uid: String) {
        guard let device = availableInputDevices.first(where: { $0.uid == uid }) else { return }
        activeInputDeviceUID = uid
        activeInputDeviceName = device.name
        activeInputDeviceKeepWarm = audioEngine.deviceWarmPolicy[uid] ?? false
        audioEngine.selectDevice(uid: uid)
        persistAudioPreferences()
        reconfigureAudioEngineIfNeeded(forceRestart: true)
    }

    func toggleActiveInputDeviceKeepWarm() {
        activeInputDeviceKeepWarm.toggle()
        if let uid = activeInputDeviceUID {
            audioEngine.deviceWarmPolicy[uid] = activeInputDeviceKeepWarm
            persistAudioPreferences()
        }
        reconfigureAudioEngineIfNeeded(forceRestart: pendingAudioReconfigureAfterSession)
        pendingAudioReconfigureAfterSession = false
    }

    init(
        audioEngine: AudioEngine,
        transcriptionService: TranscriptionService,
        inputClient: BeeInputClient
    ) {
        self.audioEngine = audioEngine
        self.transcriptionService = transcriptionService
        self.inputClient = inputClient
        restoreAudioPreferences()
        installExternalObservers()
        installCaptureDeviceObservers()
        refreshInputDevices(reason: "startup")
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
            let targetApp = NSWorkspace.shared.frontmostApplication
            let targetPID = targetApp?.processIdentifier
            beeLog("APP: hotkey down targetPID=\(targetPID.map(String.init) ?? "nil") targetApp=\(targetApp?.localizedName ?? "nil")")
            let session = createSession(targetProcessID: targetPID)
            activeSessionID = session.id
            activeSessionTargetPID = targetPID
            activeSessionTargetAppName = targetApp?.localizedName
            activeSessionTargetAppIcon = targetApp?.icon
            activeSessionIMEConfirmed = false
            pendingLockRequest = false
            loggedWaitingForIME = false
            pendingIMEAckTimeoutTask?.cancel()
            pendingIMEAckTimeoutTask = nil
            isSessionParked = false
            parkedOverlayText = ""
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
            guard activeSessionIMEConfirmed else {
                pendingLockRequest = true
                if !loggedWaitingForIME {
                    beeLog("SESSION: ROpt up while IME unconfirmed, waiting")
                    loggedWaitingForIME = true
                }
                startIMEAckTimeoutIfNeeded(session: session)
                return false
            }
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
            while !Task.isCancelled {
                guard case .pending(let s) = uiState, s.id == session.id else { return }
                guard activeSessionIMEConfirmed else {
                    try? await Task.sleep(for: .milliseconds(100))
                    continue
                }
                guard !pendingLockRequest else { return }

                uiState = .pushToTalk(s)
                playRecordingStartedSound()
                return
            }
        }
    }

    private func startIMEAckTimeoutIfNeeded(session: Session) {
        guard pendingIMEAckTimeoutTask == nil else { return }
        pendingIMEAckTimeoutTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(1200))
            guard let self else { return }
            defer { self.pendingIMEAckTimeoutTask = nil }

            guard case .pending(let s) = self.uiState, s.id == session.id else { return }
            guard self.pendingLockRequest, !self.activeSessionIMEConfirmed else { return }

            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            beeLog(
                "SESSION: IME confirm timeout id=\(session.id.uuidString.prefix(8)) targetPID=\(self.activeSessionTargetPID.map(String.init) ?? "nil") frontmostPID=\(frontmostPID.map(String.init) ?? "nil"), continuing to capture"
            )
            self.startIMEAckTimeoutIfNeeded(session: session)
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
        case .committed(let id, let text, let submitted):
            resultID = id
            if submitted {
                SoundEffects.shared.playCommitSubmit()
            } else {
                SoundEffects.shared.playCommit()
            }
            if !text.isEmpty {
                addHistoryEntry(text: text)
            }
        }

        if uiState.session?.id == resultID {
            transitionToIdle()
        }

        if activeSessionID == resultID {
            activeSessionID = nil
            activeSessionTargetPID = nil
            activeSessionTargetAppName = nil
            activeSessionTargetAppIcon = nil
            activeSessionIMEConfirmed = false
            pendingLockRequest = false
            isSessionParked = false
            hideParkedOverlay()
        }

        applyWarmPolicyForCurrentState()
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
                    self.availableInputDevices = []
                    self.activeInputDeviceUID = nil
                    self.activeInputDeviceName = nil
                    self.activeInputDeviceKeepWarm = false
                    self.modelStatus = .error("Microphone permission denied")
                }
                return
            }

            do {
                await MainActor.run {
                    self.refreshInputDevices(reason: "permission-granted")
                }
                try await transcriptionService.loadModel(
                    model: model,
                    cacheDir: STTModelDefinition.cacheDirectory
                )
                await MainActor.run {
                    self.modelStatus = .loaded
                    self.applyWarmPolicyForCurrentState()
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
            dnc.addObserver(forName: Self.imeSubmitName, object: nil, queue: .main) { [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMESubmit(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeCancelName, object: nil, queue: .main) { [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMECancel(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeUserTypedName, object: nil, queue: .main) { [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMEUserTyped(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeContextLostName, object: nil, queue: .main) { [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMEContextLost(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            dnc.addObserver(forName: Self.imeSessionStartedName, object: nil, queue: .main) { [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMESessionStarted(sessionID: sessionID)
                }
            }
        )
        workspaceObservers.append(
            nc.addObserver(
                forName: NSWorkspace.didActivateApplicationNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
                let activatedPID = app?.processIdentifier
                let activatedBundleID = app?.bundleIdentifier
                Task { @MainActor in
                    self?.handleDidActivateApplication(
                        processIdentifier: activatedPID,
                        bundleIdentifier: activatedBundleID
                    )
                }
            }
        )
        workspaceObservers.append(
            nc.addObserver(
                forName: NSWorkspace.didTerminateApplicationNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let app = notification.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
                let terminatedPID = app?.processIdentifier
                Task { @MainActor in
                    self?.handleDidTerminateApplication(processIdentifier: terminatedPID)
                }
            }
        )
    }

    private func handleIMESubmit(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }
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

    private func handleIMECancel(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }
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

    private func handleIMEUserTyped(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }
        switch uiState {
        case .pending(let session):
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session), .locked(let session), .lockedOptionHeld(let session):
            transitionToIdle()
            Task { await session.immediateCommitFromTyping() }
        case .idle:
            break
        }
    }

    private func handleIMEContextLost(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }
        if isSessionParked {
            return
        }
        if let targetPID = activeSessionTargetPID,
           let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier,
           frontmostPID == targetPID {
            beeLog("SESSION: ignoring imeContextLost while still on targetPID=\(targetPID)")
            return
        }

        switch uiState {
        case .pending(let session):
            pendingTimer?.cancel()
            parkSession(session, reason: "imeContextLost")
        case .pushToTalk(let session), .locked(let session), .lockedOptionHeld(let session):
            parkSession(session, reason: "imeContextLost")
        case .idle:
            break
        }
    }

    private func handleIMESessionStarted(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }
        guard !activeSessionIMEConfirmed else { return }

        activeSessionIMEConfirmed = true
        guard case .pending(let session) = uiState else { return }

        beeLog("SESSION: IME confirmed id=\(session.id.uuidString.prefix(8))")
        if pendingLockRequest {
            pendingLockRequest = false
            pendingTimer?.cancel()
            uiState = .locked(session)
            playRecordingStartedSound()
        }
    }

    private func handleDidActivateApplication(processIdentifier: pid_t?, bundleIdentifier: String?) {
        guard let session = uiState.session else { return }
        guard activeSessionID == session.id else { return }
        guard let targetPID = activeSessionTargetPID else { return }
        guard let processIdentifier else { return }

        // Ignore Bee + beeInput activations; they're implementation detail
        // churn and should not affect dictation lifecycle.
        if bundleIdentifier == "fasterthanlime.bee" || bundleIdentifier == "fasterthanlime.inputmethod.bee" {
            return
        }

        if processIdentifier == targetPID {
            if isSessionParked {
                beeLog("SESSION: resumed targetPID=\(targetPID)")
                isSessionParked = false
                hideParkedOverlay()
                Task { await session.resume() }
            }
            return
        }

        parkSession(session, reason: "appActivated:\(processIdentifier)")
    }

    private func handleDidTerminateApplication(processIdentifier: pid_t?) {
        guard let processIdentifier else { return }
        guard let session = uiState.session else { return }
        guard activeSessionID == session.id else { return }
        guard processIdentifier == activeSessionTargetPID else { return }

        beeLog("SESSION: target terminated pid=\(processIdentifier)")
        transitionToIdle()
        Task { await session.cancel() }
    }

    private func parkSession(_ session: Session, reason: String) {
        guard !isSessionParked else { return }
        isSessionParked = true
        beeLog("SESSION: parked id=\(session.id.uuidString.prefix(8)) reason=\(reason)")
        showParkedOverlay(for: session)
        Task { await session.park() }
    }

    private func isNotificationForActiveSession(_ sessionID: UUID?) -> Bool {
        guard let activeSessionID, let sessionID else { return false }
        return sessionID == activeSessionID
    }

    nonisolated private static func extractSessionID(_ userInfo: [AnyHashable: Any]?) -> UUID? {
        guard let raw = userInfo?["sessionID"] as? String else {
            return nil
        }
        return UUID(uuidString: raw)
    }

    private func transitionToIdle() {
        uiState = .idle
        pendingTimer?.cancel()
        activeSessionIMEConfirmed = false
        pendingLockRequest = false
        isSessionParked = false
        hideParkedOverlay()
    }

    private func showParkedOverlay(for session: Session) {
        if activeSessionTargetAppName == nil,
           let targetPID = activeSessionTargetPID,
           let app = NSRunningApplication(processIdentifier: targetPID) {
            activeSessionTargetAppName = app.localizedName
            activeSessionTargetAppIcon = app.icon
        }

        if parkedOverlayPanel == nil {
            let panel = NSPanel(
                contentRect: NSRect(x: 0, y: 0, width: 420, height: 118),
                styleMask: [.borderless, .nonactivatingPanel],
                backing: .buffered,
                defer: false
            )
            panel.isFloatingPanel = true
            panel.level = .statusBar
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .ignoresCycle]
            panel.isOpaque = false
            panel.backgroundColor = .clear
            panel.hasShadow = true
            panel.ignoresMouseEvents = true
            panel.hidesOnDeactivate = false
            panel.contentView = NSHostingView(rootView: ParkedOverlayView(appState: self))
            positionParkedOverlay(panel)
            panel.orderFrontRegardless()
            parkedOverlayPanel = panel
        } else {
            parkedOverlayPanel?.orderFrontRegardless()
        }

        parkedOverlayPollTask?.cancel()
        parkedOverlayPollTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                guard self.isSessionParked,
                      self.activeSessionID == session.id else { return }
                self.parkedOverlayText = await session.liveText()
                try? await Task.sleep(for: .milliseconds(80))
            }
        }
    }

    private func hideParkedOverlay() {
        parkedOverlayPollTask?.cancel()
        parkedOverlayPollTask = nil
        parkedOverlayPanel?.close()
        parkedOverlayPanel = nil
        parkedOverlayText = ""
    }

    private func positionParkedOverlay(_ panel: NSPanel) {
        guard let screen = NSScreen.main else { return }
        let frame = screen.visibleFrame
        let x = frame.midX - 210
        let y = frame.maxY - 170
        panel.setFrameOrigin(NSPoint(x: x, y: y))
    }

    // MARK: - Audio Preferences

    private func restoreAudioPreferences() {
        let defaults = UserDefaults.standard

        if let rawPolicy = defaults.dictionary(forKey: DefaultsKey.deviceWarmPolicy) {
            var policy: [String: Bool] = [:]
            for (uid, rawValue) in rawPolicy {
                if let boolValue = rawValue as? Bool {
                    policy[uid] = boolValue
                } else if let numberValue = rawValue as? NSNumber {
                    policy[uid] = numberValue.boolValue
                }
            }
            audioEngine.deviceWarmPolicy = policy
        }

        if let selectedUID = defaults.string(forKey: DefaultsKey.selectedInputDeviceUID) {
            activeInputDeviceUID = selectedUID
            activeInputDeviceKeepWarm = audioEngine.deviceWarmPolicy[selectedUID] ?? false
            audioEngine.selectDevice(uid: selectedUID)
        }
    }

    private func persistAudioPreferences() {
        let defaults = UserDefaults.standard
        defaults.set(activeInputDeviceUID, forKey: DefaultsKey.selectedInputDeviceUID)
        defaults.set(audioEngine.deviceWarmPolicy, forKey: DefaultsKey.deviceWarmPolicy)
    }

    private func applyWarmPolicyForCurrentState() {
        guard modelStatus == .loaded else { return }
        guard activeSessionID == nil else { return }
        if uiState.isRecording { return }

        if activeInputDeviceKeepWarm {
            if !audioEngine.isWarm {
                do {
                    try audioEngine.warmUp()
                } catch {
                    logger.error("Failed to warm audio engine for active device: \(error.localizedDescription, privacy: .public)")
                }
            }
        } else if audioEngine.isWarm {
            audioEngine.coolDown()
        }
    }

    private func installCaptureDeviceObservers() {
        let center = NotificationCenter.default

        captureDeviceObservers.append(
            center.addObserver(forName: AVCaptureDevice.wasConnectedNotification, object: nil, queue: .main) {
                [weak self] _ in
                Task { @MainActor in
                    self?.refreshInputDevices(reason: "capture-device-connected")
                }
            }
        )
        captureDeviceObservers.append(
            center.addObserver(forName: AVCaptureDevice.wasDisconnectedNotification, object: nil, queue: .main) {
                [weak self] _ in
                Task { @MainActor in
                    self?.refreshInputDevices(reason: "capture-device-disconnected")
                }
            }
        )
    }

    private func refreshInputDevices(reason: String) {
        let previousUID = activeInputDeviceUID
        let defaultUID = AVCaptureDevice.default(for: .audio)?.uniqueID

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .external],
            mediaType: .audio,
            position: .unspecified
        )
        let captureDevices = discovery.devices

        let info = captureDevices
            .map { device in
                InputDeviceInfo(
                    uid: device.uniqueID,
                    name: device.localizedName,
                    isBuiltIn: device.deviceType == .microphone,
                    isDefault: device.uniqueID == defaultUID
                )
            }
            .sorted { lhs, rhs in
                if lhs.isDefault != rhs.isDefault { return lhs.isDefault && !rhs.isDefault }
                if lhs.isBuiltIn != rhs.isBuiltIn { return lhs.isBuiltIn && !rhs.isBuiltIn }
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }

        availableInputDevices = info

        let availableUIDs = Set(info.map(\.uid))
        let topologyChanged = availableUIDs != lastKnownInputDeviceUIDs
        lastKnownInputDeviceUIDs = availableUIDs

        let selectedUID: String?
        if let current = previousUID, availableUIDs.contains(current) {
            selectedUID = current
        } else if let preferred = activeInputDeviceUID, availableUIDs.contains(preferred) {
            selectedUID = preferred
        } else if let defaultUID, availableUIDs.contains(defaultUID) {
            selectedUID = defaultUID
        } else {
            selectedUID = info.first?.uid
        }

        if let uid = selectedUID, let selected = info.first(where: { $0.uid == uid }) {
            activeInputDeviceUID = uid
            activeInputDeviceName = selected.name
            activeInputDeviceKeepWarm = audioEngine.deviceWarmPolicy[uid] ?? false
            audioEngine.selectDevice(uid: uid)
        } else {
            activeInputDeviceUID = nil
            activeInputDeviceName = nil
            activeInputDeviceKeepWarm = false
        }

        persistAudioPreferences()
        reconfigureAudioEngineIfNeeded(forceRestart: topologyChanged || previousUID != activeInputDeviceUID)

        logger.info(
            "Refreshed input devices (\(reason, privacy: .public)): count=\(info.count), selected=\(self.activeInputDeviceUID ?? "none", privacy: .public)"
        )
    }

    private func reconfigureAudioEngineIfNeeded(forceRestart: Bool) {
        guard modelStatus == .loaded else { return }
        if activeSessionID != nil || uiState.isRecording {
            if forceRestart {
                pendingAudioReconfigureAfterSession = true
            }
            return
        }

        if forceRestart && audioEngine.isWarm {
            audioEngine.coolDown()
        }

        applyWarmPolicyForCurrentState()
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

private struct ParkedOverlayView: View {
    let appState: AppState

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if let icon = appState.activeSessionTargetAppIcon {
                Image(nsImage: icon)
                    .resizable()
                    .frame(width: 28, height: 28)
                    .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            } else {
                Image(systemName: "app")
                    .font(.system(size: 22))
                    .frame(width: 28, height: 28)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Dictating in \(appState.activeSessionTargetAppName ?? "Target App")")
                    .font(.system(.headline, weight: .semibold))
                    .lineLimit(1)
                Text(appState.parkedOverlayText.isEmpty ? "Listening..." : appState.parkedOverlayText)
                    .font(.system(.body, design: .rounded))
                    .lineLimit(2)
                    .foregroundStyle(.primary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .frame(width: 420, height: 118, alignment: .leading)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(.white.opacity(0.2), lineWidth: 0.5)
        }
    }
}
