import AVFoundation
import AppKit
import ApplicationServices
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
        static let debugOverlayEnabled = "ui.debugOverlayEnabled"
    }

    private static let imeSubmitName = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelName = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedName = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostName = NSNotification.Name("fasterthanlime.bee.imeContextLost")
    private static let imeSessionStartedName = NSNotification.Name(
        "fasterthanlime.bee.imeSessionStarted")
    private static let imeActivationRevokedName = NSNotification.Name(
        "fasterthanlime.bee.imeActivationRevoked")

    private(set) var hotkeyState: HotkeyState = .idle
    private(set) var imeSessionState: IMESessionState = .inactive
    private var pendingTimer: Task<Void, Never>?
    private var consumeNextROptUp = false
    private var activeSessionTargetPID: pid_t?
    fileprivate var activeSessionTargetAppName: String?
    fileprivate var activeSessionTargetAppIcon: NSImage?
    // pendingIMEAckWorkItem declared near startIMEAckTimeoutIfNeeded
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

    // IME readiness
    var imeReady: Bool = false

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
    var maxNewTokensFinal: UInt32 = 0  // 0 = Rust default (512)

    // Debug
    var debugEnabled = false {
        didSet {
            UserDefaults.standard.set(debugEnabled, forKey: DefaultsKey.debugOverlayEnabled)
        }
    }
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
        self.debugEnabled = UserDefaults.standard.bool(forKey: DefaultsKey.debugOverlayEnabled)
        restoreAudioPreferences()
        installExternalObservers()
        installCaptureDeviceObservers()
        refreshInputDevices(reason: "startup")
    }

    // MARK: - State

    enum HotkeyState {
        case idle
        case held(Session)  // ROpt held, session starting
        case released(Session)  // ROpt released (tap), wants locked when IME ready
        case pushToTalk(Session)  // ROpt held past 300ms, recording
        case locked(Session)  // hands-free recording
        case lockedOptionHeld(Session)  // locked, ROpt pressed again

        var session: Session? {
            switch self {
            case .idle: nil
            case .held(let s): s
            case .released(let s): s
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

    enum IMESessionState {
        case inactive
        case activating  // prepareSession done, waiting for IME confirmation
        case active  // IME confirmed, text can be routed
        case parked  // target app lost focus, session paused
    }

    // MARK: - Event Handlers

    func handleROptDown() -> Bool {
        switch hotkeyState {
        case .idle:
            let targetApp = NSWorkspace.shared.frontmostApplication
            let targetPID = targetApp?.processIdentifier
            beeLog(
                "APP: hotkey down targetPID=\(targetPID.map(String.init) ?? "nil") targetApp=\(targetApp?.localizedName ?? "nil")"
            )
            let session = createSession(targetProcessID: targetPID)
            beeLog("APP: session created id=\(session.id.uuidString.prefix(8))")
            activeSessionTargetPID = targetPID
            activeSessionTargetAppName = targetApp?.localizedName
            activeSessionTargetAppIcon = targetApp?.icon
            pendingIMEAckWorkItem?.cancel()
            pendingIMEAckWorkItem = nil
            imeSessionState = .activating
            parkedOverlayText = ""
            hotkeyState = .held(session)
            startPendingTimer(session: session)
            startIMEAckTimeoutIfNeeded(session: session)
            let config = TranscriptionService.SessionConfig(
                chunkSizeSec: chunkSizeSec,
                maxNewTokensStreaming: maxNewTokensStreaming,
                maxNewTokensFinal: maxNewTokensFinal
            )
            // Kick off IME activation on MainActor immediately, then
            // start the audio/ASR pipeline on the Session actor.
            let language = detectLanguage()
            beeLog("APP: handleROptDown done, dispatching Task")
            Task {
                beeLog("APP: Task started")
                await session.start(language: language, asrConfig: config)
            }
            return false  // not swallowed

        case .locked(let session):
            hotkeyState = .lockedOptionHeld(session)
            return true  // swallowed

        default:
            return false
        }
    }

    func handleROptUp() -> Bool {
        if consumeNextROptUp {
            consumeNextROptUp = false
            return true  // swallowed (after RCmd → Locked transition)
        }

        switch hotkeyState {
        case .held(let session):
            if imeSessionState == .active {
                pendingTimer?.cancel()
                hotkeyState = .locked(session)
                playRecordingStartedSound()
            } else {
                beeLog("SESSION: ROpt up while IME unconfirmed, waiting")
                hotkeyState = .released(session)
                startIMEAckTimeoutIfNeeded(session: session)
            }
            return false

        case .pushToTalk(let session):
            transitionToIdle()
            Task { await session.commit(submit: false) }
            return true  // swallowed

        case .lockedOptionHeld(let session):
            transitionToIdle()
            Task { await session.commit(submit: false) }
            return true  // swallowed

        default:
            return false
        }
    }

    func handleRCmdDown() -> Bool {
        switch hotkeyState {
        case .pushToTalk(let session):
            hotkeyState = .locked(session)
            consumeNextROptUp = true
            return true  // swallowed

        default:
            return false
        }
    }

    func handleEscape() -> Bool {
        switch hotkeyState {
        case .held(let session), .released(let session):
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
            return true  // swallowed

        case .pushToTalk(let session):
            transitionToIdle()
            Task { await session.cancel() }
            return true  // swallowed

        case .locked:
            return false  // passthrough

        case .lockedOptionHeld(let session):
            transitionToIdle()
            Task { await session.cancel() }
            return true  // swallowed

        case .idle:
            return false
        }
    }

    func handleEnter() -> Bool {
        switch hotkeyState {
        case .locked(let session):
            transitionToIdle()
            Task { await session.commit(submit: true) }
            return true  // swallowed

        default:
            return false
        }
    }

    func handleOtherKey(keyCode: UInt16) -> Bool {
        switch hotkeyState {
        case .held(let session), .released(let session):
            pendingTimer?.cancel()
            transitionToIdle()

            // ROpt+P = paste last history entry
            if keyCode == 0x23 /* kVK_ANSI_P */ {
                pasteLastHistoryEntry()
                Task { await session.abort() }
                return true  // swallowed
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
            do { try await Task.sleep(for: .milliseconds(300)) } catch { return }
            while !Task.isCancelled {
                guard case .held(let s) = hotkeyState, s.id == session.id else { return }
                guard imeSessionState == .active else {
                    do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
                    continue
                }
                hotkeyState = .pushToTalk(s)
                playRecordingStartedSound()
                return
            }
        }
    }

    private var pendingIMEAckWorkItem: DispatchWorkItem?

    private func startIMEAckTimeoutIfNeeded(session: Session) {
        guard pendingIMEAckWorkItem == nil else { return }

        let sessionID = session.id
        // Fallback focus cycle for when activateServer never fires at all.
        // (The XPC revoked path handles spurious activate/deactivate pairs.)
        let focusCycleWork = DispatchWorkItem { [weak self] in
            guard let self else { return }
            guard self.imeSessionState != .active, self.imeSessionState != .parked else {
                self.pendingIMEAckWorkItem = nil
                return
            }
            guard self.hotkeyState.session?.id == sessionID else {
                self.pendingIMEAckWorkItem = nil
                return
            }
            beeLog("SESSION: IME not confirmed after 500ms, fallback focus cycle id=\(sessionID.uuidString.prefix(8))")
            if let targetPID = self.activeSessionTargetPID {
                BeeInputClient.stealthFocusCycle(targetPID: targetPID)
            }

            // Schedule abort after another 2s
            let abortWork = DispatchWorkItem { [weak self] in
                guard let self else { return }
                guard self.imeSessionState != .active, self.imeSessionState != .parked else {
                    self.pendingIMEAckWorkItem = nil
                    return
                }
                guard self.hotkeyState.session?.id == sessionID else {
                    self.pendingIMEAckWorkItem = nil
                    return
                }
                beeLog("SESSION: IME confirm timeout id=\(sessionID.uuidString.prefix(8)) imeState=\(self.imeSessionState), aborting")
                self.playStartFailureSound()
                self.pendingTimer?.cancel()
                self.transitionToIdle()
                Task { await session.abort() }
                self.pendingIMEAckWorkItem = nil
            }
            self.pendingIMEAckWorkItem = abortWork
            DispatchQueue.main.asyncAfter(deadline: .now() + 2.0, execute: abortWork)
        }
        pendingIMEAckWorkItem = focusCycleWork
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2, execute: focusCycleWork)
    }

    private func logFocusDiagnostics(reason: String) {
        let frontmost = NSWorkspace.shared.frontmostApplication
        let frontmostPID = frontmost?.processIdentifier
        let frontmostName = frontmost?.localizedName ?? "nil"
        let frontmostBundleID = frontmost?.bundleIdentifier ?? "nil"

        guard AXIsProcessTrusted() else {
            beeLog(
                "FOCUS DIAG [\(reason)]: frontmostPID=\(frontmostPID.map(String.init) ?? "nil") app=\(frontmostName) bundleID=\(frontmostBundleID) axTrusted=false"
            )
            return
        }

        let systemWide = AXUIElementCreateSystemWide()
        let focusedApp = axElement(systemWide, kAXFocusedApplicationAttribute as CFString)
        let focusedAppPID = focusedApp.flatMap { axPid($0) }

        let focusedElement = axElement(systemWide, kAXFocusedUIElementAttribute as CFString)
        let role =
            focusedElement.flatMap { axAttribute($0, kAXRoleAttribute as CFString) as? String }
            ?? "nil"
        let subrole =
            focusedElement.flatMap { axAttribute($0, kAXSubroleAttribute as CFString) as? String }
            ?? "nil"
        let title =
            focusedElement.flatMap { axAttribute($0, kAXTitleAttribute as CFString) as? String }
            ?? "nil"
        let valueClass =
            focusedElement.flatMap { axAttribute($0, kAXValueAttribute as CFString) }.map {
                String(describing: type(of: $0))
            } ?? "nil"
        let valueSettable = focusedElement.map {
            axAttributeSettable($0, kAXValueAttribute as CFString)
        }

        beeLog(
            "FOCUS DIAG [\(reason)]: frontmostPID=\(frontmostPID.map(String.init) ?? "nil") app=\(frontmostName) bundleID=\(frontmostBundleID) focusedAppPID=\(focusedAppPID.map(String.init) ?? "nil") role=\(role) subrole=\(subrole) valueSettable=\(valueSettable.map(String.init) ?? "nil") title=\(title.debugDescription) valueClass=\(valueClass)"
        )
    }

    private func axAttribute(_ element: AXUIElement, _ attr: CFString) -> AnyObject? {
        var value: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attr, &value)
        guard status == .success else { return nil }
        return value as AnyObject?
    }

    private func axPid(_ element: AXUIElement) -> pid_t? {
        var pid: pid_t = 0
        let status = AXUIElementGetPid(element, &pid)
        guard status == .success else { return nil }
        return pid
    }

    private func axElement(_ element: AXUIElement, _ attr: CFString) -> AXUIElement? {
        var value: CFTypeRef?
        let status = AXUIElementCopyAttributeValue(element, attr, &value)
        guard status == .success, let value else { return nil }
        guard CFGetTypeID(value) == AXUIElementGetTypeID() else { return nil }
        return (value as! AXUIElement)
    }

    private func axAttributeSettable(_ element: AXUIElement, _ attr: CFString) -> Bool {
        var settable: DarwinBoolean = false
        let status = AXUIElementIsAttributeSettable(element, attr, &settable)
        guard status == .success else { return false }
        return settable.boolValue
    }

    // MARK: - Max Duration

    private func startMaxDurationTimer(session: Session) {
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(300))
            guard !Task.isCancelled else { return }
            if hotkeyState.session?.id == session.id {
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
            break  // no trace
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

        if hotkeyState.session?.id == resultID {
            transitionToIdle()
            activeSessionTargetPID = nil
            activeSessionTargetAppName = nil
            activeSessionTargetAppIcon = nil
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

    static let defaultModel =
        STTModelDefinition.allModels.first(where: { $0.id == "qwen3-1.7b-mlx-4bit" })
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

    func warmUpIME() {
        Task {
            // Launch the IME app so it connects to the broker.
            let imeAppURL = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Input Methods/beeInput.app")
            let config = NSWorkspace.OpenConfiguration()
            config.activates = false
            do {
                try await NSWorkspace.shared.openApplication(at: imeAppURL, configuration: config)
            } catch {
                beeLog("IME WARMUP: failed to launch IME app: \(error)")
            }
            let ready = await BeeInputClient.waitForIMEReady()
            await MainActor.run {
                self.imeReady = ready
                beeLog("IME WARMUP: done imeReady=\(ready)")
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

    private func playStartFailureSound() {
        SoundEffects.shared.playStartFailure()
    }

    private func pasteLastHistoryEntry() {
        // TODO: look up most recent history entry, paste via IME
    }

    // MARK: - External Events

    private func installExternalObservers() {
        let ncLocal = NotificationCenter.default
        let nc = NSWorkspace.shared.notificationCenter

        distributedObservers.append(
            ncLocal.addObserver(forName: Self.imeSubmitName, object: nil, queue: .main) {
                [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMESubmit(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            ncLocal.addObserver(forName: Self.imeCancelName, object: nil, queue: .main) {
                [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMECancel(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            ncLocal.addObserver(forName: Self.imeUserTypedName, object: nil, queue: .main) {
                [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMEUserTyped(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            ncLocal.addObserver(forName: Self.imeContextLostName, object: nil, queue: .main) {
                [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                let hadMarkedText = Self.extractBool(notification.userInfo, key: "hadMarkedText")
                Task { @MainActor in
                    self?.handleIMEContextLost(sessionID: sessionID, hadMarkedText: hadMarkedText)
                }
            }
        )
        distributedObservers.append(
            ncLocal.addObserver(forName: Self.imeSessionStartedName, object: nil, queue: .main) {
                [weak self] notification in
                let sessionID = Self.extractSessionID(notification.userInfo)
                Task { @MainActor in
                    self?.handleIMESessionStarted(sessionID: sessionID)
                }
            }
        )
        distributedObservers.append(
            ncLocal.addObserver(
                forName: Self.imeActivationRevokedName, object: nil, queue: .main
            ) {
                [weak self] _ in
                Task { @MainActor in
                    self?.handleIMEActivationRevoked()
                }
            }
        )
        workspaceObservers.append(
            nc.addObserver(
                forName: NSWorkspace.didActivateApplicationNotification,
                object: nil,
                queue: .main
            ) { [weak self] notification in
                let app =
                    notification.userInfo?[NSWorkspace.applicationUserInfoKey]
                    as? NSRunningApplication
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
                let app =
                    notification.userInfo?[NSWorkspace.applicationUserInfoKey]
                    as? NSRunningApplication
                let terminatedPID = app?.processIdentifier
                Task { @MainActor in
                    self?.handleDidTerminateApplication(processIdentifier: terminatedPID)
                }
            }
        )
    }

    private func handleIMESubmit(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }
        switch hotkeyState {
        case .held(let session), .released(let session):
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
        switch hotkeyState {
        case .held(let session), .released(let session):
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
        switch hotkeyState {
        case .held(let session), .released(let session):
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

    private func handleIMEContextLost(sessionID: UUID?, hadMarkedText: Bool?) {
        guard isNotificationForActiveSession(sessionID) else { return }
        if imeSessionState == .parked {
            return
        }
        if imeSessionState == .activating {
            // IME deferred claim handles this — ignore during activation
            return
        }
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        beeLog(
            "SESSION: imeContextLost id=\(hotkeyState.session?.id.uuidString.prefix(8) ?? "nil") targetPID=\(activeSessionTargetPID.map(String.init) ?? "nil") frontmostPID=\(frontmostPID.map(String.init) ?? "nil") hadMarkedText=\(hadMarkedText.map(String.init) ?? "nil")"
        )

        switch hotkeyState {
        case .held(let session), .released(let session):
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session):
            transitionToIdle()
            Task { await session.cancel() }
        case .locked(let session), .lockedOptionHeld(let session):
            parkSession(session, reason: "imeContextLost")
        case .idle:
            break
        }
    }

    private func handleIMEActivationRevoked() {
        guard imeSessionState == .activating else { return }
        guard let targetPID = activeSessionTargetPID else { return }
        // Cancel the fallback timeout — we're handling it now
        pendingIMEAckWorkItem?.cancel()
        pendingIMEAckWorkItem = nil
        beeLog("SESSION: IME activation revoked, doing focus cycle for pid=\(targetPID)")
        BeeInputClient.stealthFocusCycle(targetPID: targetPID)
    }

    private func handleIMESessionStarted(sessionID: UUID?) {
        guard isNotificationForActiveSession(sessionID) else { return }

        // Parked → active (resume after focus return)
        if imeSessionState == .parked, let session = hotkeyState.session {
            beeLog("SESSION: IME route restored id=\(session.id.uuidString.prefix(8))")
            imeSessionState = .active
            hideParkedOverlay()
            Task { await session.routeDidBecomeActive() }
            return
        }

        // Only process if we're still activating (ignore duplicate confirmations)
        guard imeSessionState == .activating else { return }

        switch hotkeyState {
        case .held(let session):
            beeLog("SESSION: IME confirmed id=\(session.id.uuidString.prefix(8))")
            imeSessionState = .active
            Task { await session.routeDidBecomeActive() }

        case .released(let session):
            beeLog("SESSION: IME confirmed id=\(session.id.uuidString.prefix(8)), locking")
            imeSessionState = .active
            pendingTimer?.cancel()
            hotkeyState = .locked(session)
            Task { await session.routeDidBecomeActive() }
            playRecordingStartedSound()

        default:
            break
        }
    }

    private func handleDidActivateApplication(processIdentifier: pid_t?, bundleIdentifier: String?)
    {
        guard let session = hotkeyState.session else { return }
        guard let targetPID = activeSessionTargetPID else { return }
        guard let processIdentifier else { return }

        // Ignore Bee + beeInput activations; they're implementation detail
        // churn and should not affect dictation lifecycle.
        if bundleIdentifier == "fasterthanlime.bee"
            || bundleIdentifier == "fasterthanlime.inputmethod.bee"
        {
            return
        }

        // During IME activation (including focus cycle fallback),
        // app switches are expected and should not abort the session.
        if imeSessionState == .activating {
            return
        }

        if processIdentifier == targetPID {
            if imeSessionState == .parked {
                beeLog("SESSION: resume requested targetPID=\(targetPID)")
                Task {
                    let resumed = await session.requestResumeActivation()
                    if !resumed {
                        await MainActor.run {
                            guard self.hotkeyState.session?.id == session.id else { return }
                            beeLog(
                                "SESSION: resume activation failed id=\(session.id.uuidString.prefix(8))"
                            )
                            self.transitionToIdle()
                        }
                        await session.cancel()
                    }
                }
            }
            return
        }

        switch hotkeyState {
        case .held(let session), .released(let session):
            pendingTimer?.cancel()
            transitionToIdle()
            Task { await session.abort() }
        case .pushToTalk(let session):
            transitionToIdle()
            Task { await session.cancel() }
        case .locked, .lockedOptionHeld:
            parkSession(session, reason: "appActivated:\(processIdentifier)")
        case .idle:
            break
        }
    }

    private func handleDidTerminateApplication(processIdentifier: pid_t?) {
        guard let processIdentifier else { return }
        guard let session = hotkeyState.session else { return }
        guard processIdentifier == activeSessionTargetPID else { return }

        beeLog("SESSION: target terminated pid=\(processIdentifier)")
        transitionToIdle()
        Task { await session.cancel() }
    }

    private func parkSession(_ session: Session, reason: String) {
        guard imeSessionState != .parked else { return }
        imeSessionState = .parked
        beeLog("SESSION: parked id=\(session.id.uuidString.prefix(8)) reason=\(reason)")
        showParkedOverlay(for: session)
        Task { await session.park() }
    }

    private func isNotificationForActiveSession(_ sessionID: UUID?) -> Bool {
        guard let currentID = hotkeyState.session?.id, let sessionID else { return false }
        return sessionID == currentID
    }

    nonisolated private static func extractSessionID(_ userInfo: [AnyHashable: Any]?) -> UUID? {
        guard let raw = userInfo?["sessionID"] as? String else {
            return nil
        }
        return UUID(uuidString: raw)
    }

    nonisolated private static func extractBool(_ userInfo: [AnyHashable: Any]?, key: String)
        -> Bool?
    {
        userInfo?[key] as? Bool
    }

    private func transitionToIdle() {
        hotkeyState = .idle
        pendingTimer?.cancel()
        pendingIMEAckWorkItem?.cancel()
        pendingIMEAckWorkItem = nil
        imeSessionState = .inactive
        hideParkedOverlay()
    }

    private func showParkedOverlay(for session: Session) {
        if activeSessionTargetAppName == nil,
            let targetPID = activeSessionTargetPID,
            let app = NSRunningApplication(processIdentifier: targetPID)
        {
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
                guard self.imeSessionState == .parked,
                    self.hotkeyState.session?.id == session.id
                else { return }
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
        guard hotkeyState.session == nil else { return }
        if hotkeyState.isRecording { return }

        if activeInputDeviceKeepWarm {
            if !audioEngine.isWarm {
                do {
                    try audioEngine.warmUp()
                } catch {
                    logger.error(
                        "Failed to warm audio engine for active device: \(error.localizedDescription, privacy: .public)"
                    )
                }
            }
        } else if audioEngine.isWarm {
            audioEngine.coolDown()
        }
    }

    private func installCaptureDeviceObservers() {
        let center = NotificationCenter.default

        captureDeviceObservers.append(
            center.addObserver(
                forName: AVCaptureDevice.wasConnectedNotification, object: nil, queue: .main
            ) {
                [weak self] _ in
                Task { @MainActor in
                    self?.refreshInputDevices(reason: "capture-device-connected")
                }
            }
        )
        captureDeviceObservers.append(
            center.addObserver(
                forName: AVCaptureDevice.wasDisconnectedNotification, object: nil, queue: .main
            ) {
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

        let info =
            captureDevices
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
        reconfigureAudioEngineIfNeeded(
            forceRestart: topologyChanged || previousUID != activeInputDeviceUID)

        logger.info(
            "Refreshed input devices (\(reason, privacy: .public)): count=\(info.count), selected=\(self.activeInputDeviceUID ?? "none", privacy: .public)"
        )
    }

    private func reconfigureAudioEngineIfNeeded(forceRestart: Bool) {
        guard modelStatus == .loaded else { return }
        if hotkeyState.isRecording {
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
                Text(
                    appState.parkedOverlayText.isEmpty ? "Listening..." : appState.parkedOverlayText
                )
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
