import AVFoundation
import AppKit
import ApplicationServices
import CoreAudio
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
        static let totalSessions = "stats.totalSessions"
        static let totalWords = "stats.totalWords"
        static let totalCharacters = "stats.totalCharacters"
        static let lowerVolumeDuringDictation = "audio.lowerVolumeDuringDictation"
        static let dictationVolumeLevel = "audio.dictationVolumeLevel"
        static let chunkSizeSec = "transcription.chunkSizeSec"
        static let maxNewTokensStreaming = "transcription.maxNewTokensStreaming"
        static let maxNewTokensFinal = "transcription.maxNewTokensFinal"
    }

    private static let imeSubmitName = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelName = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedName = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostName = NSNotification.Name("fasterthanlime.bee.imeContextLost")
    private static let imeSessionStartedName = NSNotification.Name(
        "fasterthanlime.bee.imeSessionStarted")
    private static let imeActivationRevokedName = NSNotification.Name(
        "fasterthanlime.bee.imeActivationRevoked")

    private(set) var hotkeyState: HotkeyState = .idle {
        didSet { NotificationCenter.default.post(name: Self.stateChangedNotification, object: nil) }
    }
    private(set) var imeSessionState: IMESessionState = .inactive
    private var pendingTimer: Task<Void, Never>?
    private var consumeNextROptUp = false
    fileprivate var activeSessionTarget: TargetApp?
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
    static let stateChangedNotification = NSNotification.Name("fasterthanlime.bee.stateChanged")

    var modelStatus: ModelStatus = .notLoaded {
        didSet { NotificationCenter.default.post(name: Self.stateChangedNotification, object: nil) }
    }

    var loadedModelDisplayName: String? {
        if case .loaded = modelStatus {
            return Self.defaultModel.displayName
        }
        return nil
    }

    struct PipelineComponent: Identifiable {
        let role: String
        let name: String
        let url: URL
        var id: String { name }
    }

    static let pipelineComponents: [PipelineComponent] = [
        PipelineComponent(
            role: "Aligner",
            name: "Qwen3 Forced Aligner 0.6B",
            url: URL(string: "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B")!
        ),
        PipelineComponent(
            role: "VAD",
            name: "Silero VAD v5",
            url: URL(string: "https://huggingface.co/aitytech/Silero-VAD-v5-MLX")!
        ),
    ]

    // IME readiness
    var imeReady: Bool = false {
        didSet { NotificationCenter.default.post(name: Self.stateChangedNotification, object: nil) }
    }

    // Input devices
    var availableInputDevices: [InputDeviceInfo] = []
    var activeInputDeviceUID: String?
    var activeInputDeviceName: String?
    var activeInputDeviceKeepWarm: Bool = false

    // History
    var transcriptionHistory: [TranscriptionHistoryItem] = []

    // Stats (persisted)
    var totalSessions: Int = 0
    var totalWords: Int = 0
    var totalCharacters: Int = 0

    // ASR settings
    var chunkSizeSec: Float = 0.5 {
        didSet { UserDefaults.standard.set(chunkSizeSec, forKey: DefaultsKey.chunkSizeSec) }
    }
    var maxNewTokensStreaming: UInt32 = 0 {  // 0 = Rust default (32)
        didSet { UserDefaults.standard.set(Int(maxNewTokensStreaming), forKey: DefaultsKey.maxNewTokensStreaming) }
    }
    var maxNewTokensFinal: UInt32 = 0 {  // 0 = Rust default (512)
        didSet { UserDefaults.standard.set(Int(maxNewTokensFinal), forKey: DefaultsKey.maxNewTokensFinal) }
    }

    // Volume ducking
    var lowerVolumeDuringDictation: Bool = false {
        didSet { UserDefaults.standard.set(lowerVolumeDuringDictation, forKey: DefaultsKey.lowerVolumeDuringDictation) }
    }
    var dictationVolumeLevel: Float = 0.25 {
        didSet { UserDefaults.standard.set(dictationVolumeLevel, forKey: DefaultsKey.dictationVolumeLevel) }
    }
    private var savedVolume: Float?

    // Debug
    var debugEnabled = false {
        didSet {
            UserDefaults.standard.set(debugEnabled, forKey: DefaultsKey.debugOverlayEnabled)
        }
    }
    var lastSessionDiag: SessionDiag.Snapshot?
    var parkedOverlayText = ""

    enum AudioTransport: String, Sendable {
        case builtIn = "Built-in"
        case usb = "USB"
        case bluetooth = "Bluetooth"
        case continuityCamera = "Continuity"
        case virtual = "Virtual"
        case aggregate = "Aggregate"
        case unknown = "External"
    }

    struct InputDeviceInfo: Sendable {
        let uid: String
        let name: String
        let isBuiltIn: Bool
        let isDefault: Bool
        let transport: AudioTransport
        let modelUID: String?
        let manufacturer: String?

        /// SF Symbol name, or nil if using a custom symbol
        var iconName: String? {
            if customSymbolName != nil { return nil }
            if let model = modelUID {
                if model.contains("iPhone") { return "iphone" }
                if model.contains("iPad") { return "ipad" }
                if model.contains("Studio Display") { return "display" }
            }
            switch transport {
            case .builtIn: return "laptopcomputer"
            case .usb: return "dot.radiowaves.up.forward"
            case .bluetooth: return "headphones"
            case .continuityCamera: return "iphone"
            case .virtual, .aggregate: return "waveform.circle"
            case .unknown: return "mic"
            }
        }

        /// Custom symbol name for known hardware
        var customSymbolName: String? {
            if let mfr = manufacturer, mfr.contains("Focusrite") { return "focusrite" }
            return nil
        }

        /// Clean model UID for display — strip USB vendor:product suffixes like ":1235:8218"
        private var cleanModelUID: String? {
            guard let model = modelUID, !model.isEmpty else { return nil }
            // Strip trailing :XXXX:XXXX patterns (USB vendor/product IDs)
            let cleaned = model.replacingOccurrences(
                of: #":[0-9A-Fa-f]{2,}:[0-9A-Fa-f]{2,}$"#,
                with: "",
                options: .regularExpression
            )
            return cleaned.isEmpty ? nil : cleaned
        }

        var subtitle: String? {
            var parts: [String] = []
            if let mfr = manufacturer, !mfr.isEmpty {
                parts.append(mfr)
            }
            if isDefault { parts.append("Default") }
            return parts.isEmpty ? nil : parts.joined(separator: " · ")
        }
    }

    static func queryTransportType(uid: String) -> AudioTransport {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        // First resolve UID to AudioDeviceID
        var deviceID = AudioDeviceID(0)
        var cfUID = uid as CFString
        var translation = AudioValueTranslation(
            mInputData: &cfUID,
            mInputDataSize: UInt32(MemoryLayout<CFString>.size),
            mOutputData: &deviceID,
            mOutputDataSize: UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        var translationSize = UInt32(MemoryLayout<AudioValueTranslation>.size)
        var lookupAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDeviceForUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status1 = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &lookupAddress, 0, nil,
            &translationSize, &translation
        )
        guard status1 == noErr, deviceID != 0 else { return .unknown }

        var transportType: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        let status2 = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &transportType)
        guard status2 == noErr else { return .unknown }

        // 0x63637764 = "ccwd" = Continuity Camera Wireless Device
        let kTransportTypeContinuityCamera: UInt32 = 0x63637764

        switch transportType {
        case kAudioDeviceTransportTypeBuiltIn: return .builtIn
        case kAudioDeviceTransportTypeUSB: return .usb
        case kAudioDeviceTransportTypeBluetooth, kAudioDeviceTransportTypeBluetoothLE: return .bluetooth
        case kTransportTypeContinuityCamera: return .continuityCamera
        case kAudioDeviceTransportTypeVirtual: return .virtual
        case kAudioDeviceTransportTypeAggregate: return .aggregate
        default: return .unknown
        }
    }

    static func queryModelUID(uid: String) -> String? {
        var deviceID = AudioDeviceID(0)
        var cfUID = uid as CFString
        var translation = AudioValueTranslation(
            mInputData: &cfUID,
            mInputDataSize: UInt32(MemoryLayout<CFString>.size),
            mOutputData: &deviceID,
            mOutputDataSize: UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        var translationSize = UInt32(MemoryLayout<AudioValueTranslation>.size)
        var lookupAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDeviceForUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let s1 = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &lookupAddress, 0, nil, &translationSize, &translation
        )
        guard s1 == noErr, deviceID != 0 else { return nil }

        var value: CFString?
        var size = UInt32(MemoryLayout<CFString?>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyModelUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let s2 = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &value)
        return s2 == noErr ? value as String? : nil
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

    func setDeviceWarmPolicy(uid: String, warm: Bool) {
        audioEngine.deviceWarmPolicy[uid] = warm
        if uid == activeInputDeviceUID {
            activeInputDeviceKeepWarm = warm
        }
        persistAudioPreferences()
        reconfigureAudioEngineIfNeeded(forceRestart: false)
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
        let defaults = UserDefaults.standard
        self.totalSessions = defaults.integer(forKey: DefaultsKey.totalSessions)
        self.totalWords = defaults.integer(forKey: DefaultsKey.totalWords)
        self.totalCharacters = defaults.integer(forKey: DefaultsKey.totalCharacters)
        self.lowerVolumeDuringDictation = defaults.bool(forKey: DefaultsKey.lowerVolumeDuringDictation)
        let savedLevel = defaults.float(forKey: DefaultsKey.dictationVolumeLevel)
        if savedLevel > 0 { self.dictationVolumeLevel = savedLevel }
        let savedChunk = defaults.float(forKey: DefaultsKey.chunkSizeSec)
        if savedChunk > 0 { self.chunkSizeSec = savedChunk }
        let savedStreaming = defaults.integer(forKey: DefaultsKey.maxNewTokensStreaming)
        self.maxNewTokensStreaming = UInt32(savedStreaming)
        let savedFinal = defaults.integer(forKey: DefaultsKey.maxNewTokensFinal)
        self.maxNewTokensFinal = UInt32(savedFinal)
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
            let target = TargetApp(from: NSWorkspace.shared.frontmostApplication)
            beeLog(
                "APP: hotkey down targetPID=\(target.pid.map(String.init) ?? "nil") targetApp=\(target.name ?? "nil")"
            )
            let session = createSession(targetApp: target)
            beeLog("APP: session created id=\(session.id.uuidString.prefix(8))")
            activeSessionTarget = target
            pendingIMEAckWorkItem?.cancel()
            pendingIMEAckWorkItem = nil
            imeSessionState = .activating
            parkedOverlayText = ""
            hotkeyState = .held(session)
            duckVolume()
            startPendingTimer(session: session)
            startIMEAckTimeoutIfNeeded(session: session)
            let config = TranscriptionService.SessionConfig(
                chunkSizeSec: chunkSizeSec,
                maxNewTokensStreaming: maxNewTokensStreaming,
                maxNewTokensFinal: maxNewTokensFinal
            )
            let language = detectLanguage()
            beeLog("APP: handleROptDown done, dispatching Tasks")
            // IME activation on MainActor — fires immediately, no actor hop
            Task { @MainActor in
                beeLog("APP: IME activate Task started")
                let ok = await inputClient.activate(sessionID: session.id, targetPID: target.pid)
                if !ok {
                    beeLog("APP: IME activation failed")
                }
            }
            // Audio/ASR pipeline on Session actor — runs in parallel
            Task {
                beeLog("APP: Session Task started")
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
        pendingIMEAckWorkItem = abortWork
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0, execute: abortWork)
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

    private func createSession(targetApp: TargetApp) -> Session {
        let session = Session(
            audioEngine: audioEngine,
            transcriptionService: transcriptionService,
            inputClient: inputClient,
            targetApp: targetApp
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
            activeSessionTarget = nil
        }

        applyWarmPolicyForCurrentState()
    }

    private func addHistoryEntry(text: String) {
        let item = TranscriptionHistoryItem(
            text: text,
            appName: activeSessionTarget?.name,
            appIcon: activeSessionTarget?.icon
        )
        transcriptionHistory.insert(item, at: 0)
        if transcriptionHistory.count > 20 {
            transcriptionHistory = Array(transcriptionHistory.prefix(20))
        }

        // Update persistent stats
        let words = text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        totalSessions += 1
        totalWords += words
        totalCharacters += text.count
        let defaults = UserDefaults.standard
        defaults.set(totalSessions, forKey: DefaultsKey.totalSessions)
        defaults.set(totalWords, forKey: DefaultsKey.totalWords)
        defaults.set(totalCharacters, forKey: DefaultsKey.totalCharacters)
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
            "SESSION: imeContextLost id=\(hotkeyState.session?.id.uuidString.prefix(8) ?? "nil") targetPID=\(activeSessionTarget?.pid.map(String.init) ?? "nil") frontmostPID=\(frontmostPID.map(String.init) ?? "nil") hadMarkedText=\(hadMarkedText.map(String.init) ?? "nil")"
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
        beeLog("SESSION: IME activation revoked (palette mode — no focus cycle needed)")
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
        guard let targetPID = activeSessionTarget?.pid else { return }
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
        guard processIdentifier == activeSessionTarget?.pid else { return }

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
        restoreVolume()
    }

    // MARK: - Volume Ducking

    private var volumeFadeTask: Task<Void, Never>?

    private func duckVolume() {
        guard lowerVolumeDuringDictation else { return }
        if let current = Self.getSystemVolume() {
            savedVolume = current
            fadeVolume(to: current * dictationVolumeLevel)
        }
    }

    private func restoreVolume() {
        guard let volume = savedVolume else { return }
        savedVolume = nil
        fadeVolume(to: volume)
    }

    private func fadeVolume(to target: Float, duration: Double = 0.18) {
        volumeFadeTask?.cancel()
        let steps = 20
        let stepMs = Int(duration * 1000) / steps
        volumeFadeTask = Task {
            guard let start = Self.getSystemVolume() else { return }
            let delta = target - start
            for i in 1...steps {
                if Task.isCancelled { break }
                let t = Float(i) / Float(steps)
                let smooth = t * t * (3 - 2 * t)
                Self.setSystemVolume(start + delta * smooth)
                try? await Task.sleep(for: .milliseconds(stepMs))
            }
            Self.setSystemVolume(target)
        }
    }

    nonisolated private static func getSystemVolume() -> Float? {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let s1 = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        guard s1 == noErr else { return nil }

        var volume: Float32 = 0
        size = UInt32(MemoryLayout<Float32>.size)
        address.mSelector = kAudioHardwareServiceDeviceProperty_VirtualMainVolume
        address.mScope = kAudioDevicePropertyScopeOutput
        let s2 = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &volume)
        return s2 == noErr ? volume : nil
    }

    nonisolated private static func setSystemVolume(_ volume: Float) {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let s1 = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID)
        guard s1 == noErr else { return }

        var vol = max(0, min(1, volume))
        size = UInt32(MemoryLayout<Float32>.size)
        address.mSelector = kAudioHardwareServiceDeviceProperty_VirtualMainVolume
        address.mScope = kAudioDevicePropertyScopeOutput
        AudioObjectSetPropertyData(deviceID, &address, 0, nil, size, &vol)
    }

    private func showParkedOverlay(for session: Session) {
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
            .filter { !$0.uniqueID.hasPrefix("CADefaultDevice") }
            .map { device in
                let transport = Self.queryTransportType(uid: device.uniqueID)
                let modelUID = Self.queryModelUID(uid: device.uniqueID)
                return InputDeviceInfo(
                    uid: device.uniqueID,
                    name: device.localizedName,
                    isBuiltIn: device.deviceType == .microphone,
                    isDefault: device.uniqueID == defaultUID,
                    transport: transport,
                    modelUID: modelUID,
                    manufacturer: device.manufacturer
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
            if let icon = appState.activeSessionTarget?.icon {
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
                Text("Dictating in \(appState.activeSessionTarget?.name ?? "Target App")")
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
