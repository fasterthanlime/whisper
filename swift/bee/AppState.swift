import AVFoundation
import AppKit
import ApplicationServices
import CoreAudio
import Foundation
import SwiftUI
import UserNotifications
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
        static let devicePriorityList = "audio.devicePriorityList"
        static let hiddenDeviceUIDs = "audio.hiddenDeviceUIDs"
        static let knownDevices = "audio.knownDevices"
        static let debugOverlayEnabled = "ui.debugOverlayEnabled"
        static let totalSessions = "stats.totalSessions"
        static let totalWords = "stats.totalWords"
        static let totalCharacters = "stats.totalCharacters"
        static let soundEffectsEnabled = "audio.soundEffectsEnabled"
        static let lowerVolumeDuringDictation = "audio.lowerVolumeDuringDictation"
        static let dictationVolumeLevel = "audio.dictationVolumeLevel"
        static let chunkSizeSec = "transcription.chunkSizeSec"
        static let commitTokenCount = "transcription.commitTokenCount"
        static let rollbackTokenNum = "transcription.rollbackTokenNum"
        static let animationMorphSpeed = "transcription.animationMorphSpeed"
        static let animationAppendSpeed = "transcription.animationAppendSpeed"
    }

    private(set) var hotkeyState: HotkeyState = .idle {
        didSet { NotificationCenter.default.post(name: Self.stateChangedNotification, object: nil) }
    }
    private(set) var imeSessionState: IMESessionState = .inactive
    private var pendingTimer: Task<Void, Never>?
    private var consumeNextROptUp = false
    fileprivate var activeSessionTarget: TargetApp?
    // pendingIMEAckWorkItem declared near startIMEAckTimeoutIfNeeded
    private var workspaceObservers: [NSObjectProtocol] = []
    private var captureDeviceObservers: [NSObjectProtocol] = []
    private var lastKnownInputDeviceUIDs: Set<String> = []
    private var pendingAudioReconfigureAfterSession = false
    private var parkedOverlayPanel: NSPanel?
    private var parkedOverlayPollTask: Task<Void, Never>?

    // Shared infrastructure
    let audioEngine: AudioEngine
    let beeEngine: BeeEngine
    let transcriptionService: TranscriptionService
    let correctionService: CorrectionService
    let inputClient: BeeInputClient

    // Correction state
    var lastCorrectionOutput: CorrectionService.Output?

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
    var devicePriorityList: [String] = []  // UIDs, highest priority first
    var hiddenDeviceUIDs: Set<String> = []  // UIDs hidden from menu bar
    /// All devices we've ever seen, keyed by UID. Used to show offline devices.
    private(set) var knownDevices: [String: InputDeviceInfo] = [:]

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
    var commitTokenCount: UInt32 = 12 {
        didSet { UserDefaults.standard.set(Int(commitTokenCount), forKey: DefaultsKey.commitTokenCount) }
    }
    var rollbackTokenNum: UInt32 = 5 {
        didSet { UserDefaults.standard.set(Int(rollbackTokenNum), forKey: DefaultsKey.rollbackTokenNum) }
    }
    var animationMorphSpeed: Float = 1.0 {
        didSet { UserDefaults.standard.set(animationMorphSpeed, forKey: DefaultsKey.animationMorphSpeed) }
    }
    var animationAppendSpeed: Float = 1.0 {
        didSet { UserDefaults.standard.set(animationAppendSpeed, forKey: DefaultsKey.animationAppendSpeed) }
    }

    // Volume ducking
    var soundEffectsEnabled: Bool = true {
        didSet { UserDefaults.standard.set(soundEffectsEnabled, forKey: DefaultsKey.soundEffectsEnabled) }
    }
    var lowerVolumeDuringDictation: Bool = false {
        didSet { UserDefaults.standard.set(lowerVolumeDuringDictation, forKey: DefaultsKey.lowerVolumeDuringDictation) }
    }
    var dictationVolumeLevel: Float = 0.25 {
        didSet { UserDefaults.standard.set(dictationVolumeLevel, forKey: DefaultsKey.dictationVolumeLevel) }
    }
    private var savedVolume: Float?

    // Debug
    var menuBarPanelOpen = false
    var audioSettingsOpen = false
    var echoActive = false
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

    func setDevicePriority(orderedUIDs: [String]) {
        devicePriorityList = orderedUIDs
        persistAudioPreferences()
        // If the highest-priority available device isn't currently selected, switch to it
        let bestUID = bestAvailableDeviceUID()
        if let bestUID, bestUID != activeInputDeviceUID {
            beeLog("AUDIO: priority changed, switching to \(bestUID)")
            selectInputDevice(uid: bestUID)
        }
    }

    /// Returns the UID of the highest-priority device that's currently available.
    private func bestAvailableDeviceUID() -> String? {
        let availableUIDs = Set(availableInputDevices.map(\.uid))
        // Check priority list first
        for uid in devicePriorityList {
            if availableUIDs.contains(uid) {
                return uid
            }
        }
        // Fall back to system default, then first available
        if let defaultDevice = availableInputDevices.first(where: { $0.isDefault }) {
            return defaultDevice.uid
        }
        return availableInputDevices.first?.uid
    }

    func toggleDeviceWarmPolicy(uid: String) {
        let current = audioEngine.deviceWarmPolicy[uid] ?? false
        setDeviceWarmPolicy(uid: uid, warm: !current)
    }

    func toggleDeviceHidden(uid: String) {
        if hiddenDeviceUIDs.contains(uid) {
            hiddenDeviceUIDs.remove(uid)
        } else {
            hiddenDeviceUIDs.insert(uid)
        }
        persistAudioPreferences()
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
        beeEngine: BeeEngine,
        transcriptionService: TranscriptionService,
        correctionService: CorrectionService,
        inputClient: BeeInputClient
    ) {
        self.audioEngine = audioEngine
        self.beeEngine = beeEngine
        self.transcriptionService = transcriptionService
        self.correctionService = correctionService
        self.inputClient = inputClient
        self.debugEnabled = UserDefaults.standard.bool(forKey: DefaultsKey.debugOverlayEnabled)
        let defaults = UserDefaults.standard
        self.totalSessions = defaults.integer(forKey: DefaultsKey.totalSessions)
        self.totalWords = defaults.integer(forKey: DefaultsKey.totalWords)
        self.totalCharacters = defaults.integer(forKey: DefaultsKey.totalCharacters)
        // soundEffectsEnabled defaults to true if key not set
        self.soundEffectsEnabled = defaults.object(forKey: DefaultsKey.soundEffectsEnabled) == nil
            ? true
            : defaults.bool(forKey: DefaultsKey.soundEffectsEnabled)
        self.lowerVolumeDuringDictation = defaults.bool(forKey: DefaultsKey.lowerVolumeDuringDictation)
        let savedLevel = defaults.float(forKey: DefaultsKey.dictationVolumeLevel)
        if savedLevel > 0 { self.dictationVolumeLevel = savedLevel }
        let savedChunk = defaults.float(forKey: DefaultsKey.chunkSizeSec)
        if savedChunk > 0 { self.chunkSizeSec = savedChunk }
        let savedCommit = defaults.integer(forKey: DefaultsKey.commitTokenCount)
        if savedCommit > 0 { self.commitTokenCount = UInt32(savedCommit) }
        let savedRollback = defaults.integer(forKey: DefaultsKey.rollbackTokenNum)
        if savedRollback > 0 { self.rollbackTokenNum = UInt32(savedRollback) }
        let savedMorphSpeed = defaults.float(forKey: DefaultsKey.animationMorphSpeed)
        if savedMorphSpeed > 0 { self.animationMorphSpeed = savedMorphSpeed }
        let savedAppendSpeed = defaults.float(forKey: DefaultsKey.animationAppendSpeed)
        if savedAppendSpeed > 0 { self.animationAppendSpeed = savedAppendSpeed }
        restoreAudioPreferences()
        BeeIPCServer.shared.appState = self
        installExternalObservers()
        installCaptureDeviceObservers()
        refreshInputDevices(reason: "startup")
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert]) { _, _ in }
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
            playRecordingStartedSound()
            startPendingTimer(session: session)
            startIMEAckTimeoutIfNeeded(session: session)
            let config = TranscriptionService.StreamingSessionConfig(
                language: detectLanguage(),
                chunkDuration: chunkSizeSec,
                vadThreshold: 0,
                rollbackTokens: rollbackTokenNum,
                commitTokenCount: commitTokenCount
            )
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
                await session.start(language: config.language, asrConfig: config, animationMorphSpeed: animationMorphSpeed, animationAppendSpeed: animationAppendSpeed)
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
            pendingTimer?.cancel()
            hotkeyState = .locked(session)
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

    func handleEscape(optionHeld: Bool) -> Bool {
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

        case .locked(let session):
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let targetPID = activeSessionTarget?.pid
            let shouldCancel =
                optionHeld || (imeSessionState == .active && targetPID != nil && frontmostPID == targetPID)
            guard shouldCancel else { return false }
            transitionToIdle()
            Task { await session.cancel() }
            return true  // swallowed

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
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let targetPID = activeSessionTarget?.pid
            guard imeSessionState == .active, targetPID != nil, frontmostPID == targetPID else {
                return false
            }
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

            // ROpt+C = open correction review panel
            if keyCode == 0x08 /* kVK_ANSI_C */ {
                Task { await session.abort() }
                if let output = lastCorrectionOutput, !output.edits.isEmpty {
                    CorrectionPanel.shared.show(
                        output: output,
                        correctionService: correctionService,
                        inputClient: inputClient
                    )
                }
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
            guard case .held(let s) = hotkeyState, s.id == session.id else { return }
            hotkeyState = .pushToTalk(s)
        }
    }

    private var pendingIMEAckWorkItem: DispatchWorkItem?

    private func startIMEAckTimeoutIfNeeded(session: Session) {
        guard pendingIMEAckWorkItem == nil else { return }

        let sessionID = session.id
        let abortWork = DispatchWorkItem { [weak self] in
            guard let self else { return }
            guard self.imeSessionState != .active else {
                self.pendingIMEAckWorkItem = nil
                return
            }
            guard self.hotkeyState.session?.id == sessionID else {
                self.pendingIMEAckWorkItem = nil
                return
            }
            beeLog("SESSION: IME confirm timeout id=\(sessionID.uuidString.prefix(8)) imeState=\(self.imeSessionState), showing overlay")
            self.pendingTimer?.cancel()
            switch self.hotkeyState {
            case .held(let current) where current.id == sessionID:
                self.hotkeyState = .pushToTalk(current)
                Task { await current.routeDidBecomeInactive(reason: "imeConfirmTimeout") }
            case .released(let current) where current.id == sessionID:
                self.hotkeyState = .locked(current)
                Task { await current.routeDidBecomeInactive(reason: "imeConfirmTimeout") }
            default:
                break
            }
            self.imeSessionState = .inactive
            self.showParkedOverlay(for: session)
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
            correctionService: correctionService,
            inputClient: inputClient,
            targetApp: targetApp
        )

        Task {
            await session.setOnFinalizing { [weak self] mode in
                Task { @MainActor in
                    guard let self, self.soundEffectsEnabled else { return }
                    switch mode {
                    case .commit(let submit):
                        if submit {
                            SoundEffects.shared.playCommitSubmit()
                        } else {
                            SoundEffects.shared.playCommit()
                        }
                    case .cancel:
                        SoundEffects.shared.playCancel()
                    }
                }
            }
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
            if !text.isEmpty {
                addHistoryEntry(text: text)
            }
        case .committed(let id, let text, _, let correction):
            resultID = id
            lastCorrectionOutput = correction
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
        let cacheDir = STTModelDefinition.cacheDirectory
        beeLog("APP: loading model from cache=\(cacheDir)")
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

                // Connect to Rust via vox-ffi
                try await beeEngine.connect()
                beeLog("APP: vox-ffi connected")

                // Check for required downloads
                guard let beeClient = await beeEngine.client else {
                    throw BeeError.engineNotLoaded
                }
                let repos = try await beeClient.requiredDownloads()
                let cacheURL = URL(fileURLWithPath: cacheDir)
                try FileManager.default.createDirectory(at: cacheURL, withIntermediateDirectories: true)

                let downloaded = try await HFDownloader.downloadMissing(
                    repos: repos,
                    cacheDir: cacheURL
                ) { progress, model in
                    Task { @MainActor in
                        self.modelStatus = .downloading(progress: progress, model: model)
                    }
                }
                if downloaded > 0 {
                    beeLog("APP: downloaded \(downloaded) files")
                }

                await MainActor.run {
                    self.modelStatus = .loading
                }

                // Load engine via vox-ffi
                try await transcriptionService.loadModel(cacheDir: cacheDir)

                // Load correction engine from group container dataset
                if let groupContainer = FileManager.default.containerURL(
                    forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee"
                ) {
                    let datasetDir = groupContainer.appendingPathComponent("phonetic-seed").path
                    let eventsPath = groupContainer.appendingPathComponent("correction-events.jsonl").path
                    if FileManager.default.fileExists(atPath: datasetDir) {
                        do {
                            try await correctionService.load(
                                datasetDir: datasetDir,
                                eventsPath: eventsPath
                            )
                            beeLog("APP: correction engine loaded from \(datasetDir)")
                        } catch {
                            beeLog("APP: correction engine failed to load: \(error)")
                        }
                    } else {
                        beeLog("APP: phonetic-seed dataset not found at \(datasetDir) — run install-bee.sh to copy it")
                    }
                }

                await MainActor.run {
                    self.modelStatus = .loaded
                    self.applyWarmPolicyForCurrentState()
                }
            } catch {
                beeLog("APP: model load failed: \(error)")
                await MainActor.run {
                    self.modelStatus = .error(error.localizedDescription)
                }
            }
        }
    }

    func warmUpIME() {
        Task {
            // beeInput is embedded in bee.app and launched automatically by the OS
            // via TIS when registered. Just wait for it to connect via Vox IPC.
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
        guard soundEffectsEnabled else { return }
        SoundEffects.shared.playRecordingStarted()
    }

    private func playStartFailureSound() {
        guard soundEffectsEnabled else { return }
        SoundEffects.shared.playStartFailure()
    }

    private func pasteLastHistoryEntry() {
        // TODO: look up most recent history entry, paste via IME
    }

    // MARK: - External Events

    private func installExternalObservers() {
        let nc = NSWorkspace.shared.notificationCenter

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

    func handleIMESubmit() {
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

    func handleIMECancel() {
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

    func handleIMEUserTyped() {
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

    func handleIMEContextLost(hadMarkedText: Bool) {
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        beeLog(
            "SESSION: imeContextLost id=\(hotkeyState.session?.id.uuidString.prefix(8) ?? "nil") hotkey=\(String(describing: hotkeyState)) imeState=\(imeSessionState) targetPID=\(activeSessionTarget?.pid.map(String.init) ?? "nil") frontmostPID=\(frontmostPID.map(String.init) ?? "nil") hadMarkedText=\(hadMarkedText)"
        )

        guard let session = hotkeyState.session else { return }
        imeSessionState = .inactive
        showParkedOverlay(for: session)
        Task { await session.routeDidBecomeInactive(reason: "imeContextLost") }
    }

    func handleIMESessionStarted() {
        guard imeSessionState != .active else { return }
        beeLog(
            "SESSION: handleIMESessionStarted hotkey=\(String(describing: hotkeyState)) imeState(before)=\(imeSessionState)"
        )

        switch hotkeyState {
        case .held(let session), .pushToTalk(let session), .locked(let session),
            .lockedOptionHeld(let session):
            beeLog("SESSION: IME attached id=\(session.id.uuidString.prefix(8))")
            imeSessionState = .active
            hideParkedOverlay()
            Task { await session.routeDidBecomeActive() }

        case .released(let session):
            beeLog("SESSION: IME attached id=\(session.id.uuidString.prefix(8)), locking")
            imeSessionState = .active
            pendingTimer?.cancel()
            hotkeyState = .locked(session)
            hideParkedOverlay()
            Task { await session.routeDidBecomeActive() }

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

    private func transitionToIdle() {
        beeLog("SESSION: transitionToIdle hotkey=\(String(describing: hotkeyState)) imeState=\(imeSessionState)")
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
        let level = dictationVolumeLevel
        Task {
            guard let current = Self.getSystemVolume() else { return }
            savedVolume = current
            fadeVolume(to: current * level)
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
        beeLog("OVERLAY: show session=\(session.id.uuidString.prefix(8)) imeState=\(imeSessionState)")
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
                guard self.imeSessionState != .active,
                    self.hotkeyState.session?.id == session.id
                else { return }
                self.parkedOverlayText = await session.liveText()
                try? await Task.sleep(for: .milliseconds(80))
            }
        }
    }

    private func hideParkedOverlay() {
        beeLog("OVERLAY: hide")
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

        if let savedPriority = defaults.stringArray(forKey: DefaultsKey.devicePriorityList) {
            devicePriorityList = savedPriority
        }
        if let savedHidden = defaults.stringArray(forKey: DefaultsKey.hiddenDeviceUIDs) {
            hiddenDeviceUIDs = Set(savedHidden)
        }
        restoreKnownDevices()
    }

    private func persistAudioPreferences() {
        let defaults = UserDefaults.standard
        defaults.set(activeInputDeviceUID, forKey: DefaultsKey.selectedInputDeviceUID)
        defaults.set(audioEngine.deviceWarmPolicy, forKey: DefaultsKey.deviceWarmPolicy)
        defaults.set(devicePriorityList, forKey: DefaultsKey.devicePriorityList)
        defaults.set(Array(hiddenDeviceUIDs), forKey: DefaultsKey.hiddenDeviceUIDs)
    }

    private func persistKnownDevices() {
        // Store as array of dictionaries
        let encoded = knownDevices.values.map { device -> [String: String] in
            var dict: [String: String] = [
                "uid": device.uid,
                "name": device.name,
                "transport": device.transport.rawValue,
            ]
            if device.isBuiltIn { dict["isBuiltIn"] = "true" }
            if let modelUID = device.modelUID { dict["modelUID"] = modelUID }
            if let manufacturer = device.manufacturer { dict["manufacturer"] = manufacturer }
            return dict
        }
        UserDefaults.standard.set(encoded, forKey: DefaultsKey.knownDevices)
    }

    private func restoreKnownDevices() {
        guard let saved = UserDefaults.standard.array(forKey: DefaultsKey.knownDevices) as? [[String: String]] else { return }
        for dict in saved {
            guard let uid = dict["uid"], let name = dict["name"] else { continue }
            let transport = AudioTransport(rawValue: dict["transport"] ?? "") ?? .unknown
            knownDevices[uid] = InputDeviceInfo(
                uid: uid,
                name: name,
                isBuiltIn: dict["isBuiltIn"] == "true",
                isDefault: false,
                transport: transport,
                modelUID: dict["modelUID"],
                manufacturer: dict["manufacturer"]
            )
        }
    }

    /// All known devices sorted by priority, with online status.
    func allDevicesForSettings() -> [(device: InputDeviceInfo, isOnline: Bool)] {
        let onlineUIDs = Set(availableInputDevices.map(\.uid))
        let priority = devicePriorityList

        // Merge: online devices (updated info) + offline known devices
        var seen = Set<String>()
        var result: [(device: InputDeviceInfo, isOnline: Bool)] = []

        // First, all devices in priority order
        for uid in priority {
            if let online = availableInputDevices.first(where: { $0.uid == uid }) {
                result.append((online, true))
                seen.insert(uid)
            } else if let known = knownDevices[uid] {
                result.append((known, false))
                seen.insert(uid)
            }
        }

        // Then online devices not in priority list
        for device in availableInputDevices where !seen.contains(device.uid) {
            result.append((device, true))
            seen.insert(device.uid)
        }

        // Then offline known devices not in priority list
        for (uid, device) in knownDevices where !seen.contains(uid) {
            result.append((device, false))
        }

        return result
    }

    private func applyWarmPolicyForCurrentState() {
        guard modelStatus == .loaded else { return }

        // During recording, always keep warm (the session needs audio)
        if hotkeyState.isRecording || hotkeyState.session != nil {
            if !audioEngine.isWarm {
                do {
                    try audioEngine.warmUp()
                    beeLog("AUDIO: re-warmed during active session")
                } catch {
                    beeLog("AUDIO: failed to re-warm during session: \(error)")
                }
            }
            return
        }

        let shouldBeWarm = activeInputDeviceKeepWarm || menuBarPanelOpen || audioSettingsOpen
        if shouldBeWarm {
            if !audioEngine.isWarm {
                do {
                    try audioEngine.warmUp()
                } catch {
                    beeLog("AUDIO: Failed to warm engine: \(error.localizedDescription)")
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

        // Record all seen devices for offline display
        for device in info {
            knownDevices[device.uid] = device
        }
        persistKnownDevices()

        let availableUIDs = Set(info.map(\.uid))
        let topologyChanged = availableUIDs != lastKnownInputDeviceUIDs
        lastKnownInputDeviceUIDs = availableUIDs

        // Priority-based device selection:
        // On connect: switch to highest-priority available device
        // On disconnect: fall back to highest-priority available device
        let selectedUID = bestAvailableDeviceUID()

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

        let deviceChanged = previousUID != activeInputDeviceUID
        let needsRestart = topologyChanged || deviceChanged
        beeLog("AUDIO: refreshInputDevices(\(reason)): count=\(info.count), prev=\(previousUID ?? "nil"), now=\(activeInputDeviceUID ?? "nil"), topologyChanged=\(topologyChanged), needsRestart=\(needsRestart)")

        // Notify user when device auto-switches
        if deviceChanged, let newName = activeInputDeviceName {
            let previousName = info.first(where: { $0.uid == previousUID })?.name ?? "unknown"
            sendDeviceSwitchNotification(from: previousName, to: newName, reason: reason)
        }

        reconfigureAudioEngineIfNeeded(forceRestart: needsRestart)
    }

    private func sendDeviceSwitchNotification(from previousName: String, to newName: String, reason: String) {
        beeLog("AUDIO: device switched from \(previousName) to \(newName) (\(reason))")

        let content = UNMutableNotificationContent()
        content.title = "Audio input changed"
        content.body = newName
        content.sound = nil

        let request = UNNotificationRequest(
            identifier: "device-switch",
            content: content,
            trigger: nil
        )
        UNUserNotificationCenter.current().add(request)
    }

    private func reconfigureAudioEngineIfNeeded(forceRestart: Bool) {
        guard modelStatus == .loaded else { return }

        if forceRestart && audioEngine.isWarm {
            beeLog("AUDIO: reconfigure: force restarting (recording=\(hotkeyState.isRecording))")
            audioEngine.coolDown()
        } else if hotkeyState.isRecording {
            if forceRestart {
                pendingAudioReconfigureAfterSession = true
            }
            return
        }

        applyWarmPolicyForCurrentState()
    }
}

// MARK: - Model Status

enum ModelStatus: Equatable {
    case notLoaded
    case downloading(progress: Double, model: String)
    case loading
    case loaded
    case error(String)

    var hasError: Bool {
        if case .error = self { return true }
        return false
    }

    var errorMessage: String? {
        if case .error(let msg) = self { return msg }
        return nil
    }
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
