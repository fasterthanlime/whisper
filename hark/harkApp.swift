import AppKit
import AudioToolbox
import AVFoundation
import Carbon.HIToolbox.Events
import CoreAudio
import os
import ServiceManagement
import SwiftUI

extension Notification.Name {
    static let cancelRecording = Notification.Name("cancelRecording")
    static let submitRecording = Notification.Name("submitRecording")
}

private enum StreamingSignal {
    case none              // normal exit (phase changed, key released)
    case over              // ". Over." — submit + keep recording
    case overAndOut        // ". Over and out." — submit + stop
}

private struct StreamingResult {
    let text: String
    let signal: StreamingSignal
    let processedSampleCount: Int
}

/// Intercepts Esc/Return while recording so those keys do not leak to the app behind Hark.
private final class RecordingControlInterceptor: @unchecked Sendable {
    nonisolated(unsafe) var onIntercept: ((UInt16) -> Void)?
    nonisolated(unsafe) var shouldIntercept: ((UInt16) -> Bool)?

    nonisolated(unsafe) private var eventTap: CFMachPort?
    nonisolated(unsafe) private var runLoopSource: CFRunLoopSource?
    nonisolated(unsafe) private var swallowedKeyUps: Set<UInt16> = []

    nonisolated func start() {
        guard eventTap == nil else { return }

        let mask: CGEventMask =
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.keyUp.rawValue)

        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: mask,
            callback: recordingControlCallback,
            userInfo: refcon
        ) else {
            return
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        if let source = runLoopSource {
            CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        }
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    nonisolated func stop() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        swallowedKeyUps.removeAll()
    }

    deinit {
        stop()
    }

    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            swallowedKeyUps.removeAll()
            if let tap = eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        guard type == .keyDown || type == .keyUp else {
            return Unmanaged.passUnretained(event)
        }

        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let isEscapeOrReturn = keyCode == UInt16(kVK_Escape) || keyCode == UInt16(kVK_Return)
        guard isEscapeOrReturn else {
            return Unmanaged.passUnretained(event)
        }

        switch type {
        case .keyDown:
            let isRepeat = event.getIntegerValueField(.keyboardEventAutorepeat) != 0
            if isRepeat {
                return swallowedKeyUps.contains(keyCode) ? nil : Unmanaged.passUnretained(event)
            }

            guard shouldIntercept?(keyCode) == true else {
                return Unmanaged.passUnretained(event)
            }

            swallowedKeyUps.insert(keyCode)
            onIntercept?(keyCode)
            return nil

        case .keyUp:
            guard swallowedKeyUps.contains(keyCode) else {
                return Unmanaged.passUnretained(event)
            }
            swallowedKeyUps.remove(keyCode)
            return nil

        default:
            return Unmanaged.passUnretained(event)
        }
    }
}

nonisolated(unsafe) private let recordingControlCallback: CGEventTapCallBack = { _, type, event, refcon in
    guard let refcon else {
        return Unmanaged.passUnretained(event)
    }
    let interceptor = Unmanaged<RecordingControlInterceptor>.fromOpaque(refcon).takeUnretainedValue()
    return interceptor.handle(type: type, event: event)
}

@main
struct HarkApp: App {
    private static let sharedTranscriptionService = TranscriptionService()
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "startup"
    )

    @State private var appState = AppState()
    @State private var overlayManager = OverlayManager()
    @State private var audioRecorder = AudioRecorder()
    private let transcriptionService = sharedTranscriptionService
    @State private var hotkeyMonitor = HotkeyMonitor()
    @State private var modelLoadTask: Task<Void, Never>?
    @State private var modelLoadGeneration: UInt64 = 0
    @State private var hasLaunched = false
    @State private var recordingTimeoutTask: Task<Void, Never>?
    @State private var streamingTask: Task<StreamingResult, Never>?
    @State private var streamingSession: StreamingSession?
    @State private var inputDeviceMonitor = InputDeviceMonitor()
    @State private var keyDownTime: Date?
    /// Skip the next keyUp after locking (so releasing the hotkey after ⌘-lock doesn't submit).
    @State private var skipNextKeyUp = false
    @State private var recordingControlInterceptor = RecordingControlInterceptor()
    @State private var recordingControlObservers: [NSObjectProtocol] = []

    private static let maxRecordingDurationSeconds = AudioRecorder.defaultMaximumDuration
    private static let toggleModeThresholdSeconds: TimeInterval = 0.3
    private static let minimumSpeechDurationSeconds = 0.2
    private static let transcriptionSampleRate = 16_000.0
    private static let finalizationSilencePaddingSeconds = 0.35

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(
                appState: appState,
                onModelSelect: { model in
                    selectModel(model)
                },
                onDeleteLocalModel: { model in
                    Task { @MainActor in
                        deleteLocalModel(model)
                    }
                },
                onHotkeyBindingSave: { binding in
                    updateHotkeyBinding(binding)
                },
                onHotkeyEditorPresentedChange: { isPresented in
                    appState.isEditingHotkey = isPresented
                },
                runOnStartupEnabled: appState.runOnStartupEnabled,
                onRunOnStartupToggle: {
                    toggleRunOnStartup()
                },
                onSelectInputDevice: { uid in
                    selectInputDevice(uid: uid)
                },
                onSetActiveInputDeviceKeepWarm: { keepWarm in
                    setActiveInputDeviceKeepWarm(keepWarm)
                },
                onRequestMicrophonePermission: {
                    Task { @MainActor in
                        await requestMicrophonePermissionFromMenu()
                    }
                },
                onRequestAccessibilityPermission: {
                    requestAccessibilityPermissionFromMenu()
                },
                onRecheckPermissions: {
                    refreshPermissionState()
                },
                onQuit: {
                    NSApplication.shared.terminate(nil)
                }
            )
        } label: {
            let icon = menuBarNSImage(symbolName: appState.menuBarIcon, size: 18)
            Image(nsImage: icon)
                .task {
                    guard !hasLaunched else { return }
                    hasLaunched = true
                    await onLaunch()
                }
        }
        .menuBarExtraStyle(.window)
    }

    private func menuBarNSImage(symbolName: String, size: CGFloat) -> NSImage {
        let config = NSImage.SymbolConfiguration(pointSize: size, weight: .regular)
        let image = NSImage(systemSymbolName: symbolName, accessibilityDescription: nil)?
            .withSymbolConfiguration(config) ?? NSImage()
        image.isTemplate = true
        return image
    }

    // MARK: - Launch

    @MainActor
    private func onLaunch() async {
        registerBundledFonts()

        let savedID = UserDefaults.standard.string(forKey: "selectedModelID")
        let validIDs = Set(STTModelDefinition.allModels.map(\.id))
        let defaultID = savedID.flatMap { validIDs.contains($0) ? $0 : nil }
            ?? STTModelDefinition.default.id

        appState.selectedModelID = defaultID
        if let saved = UserDefaults.standard.dictionary(forKey: "appLanguages") as? [String: String] {
            appState.appLanguages = saved
        }
        if let saved = UserDefaults.standard.dictionary(forKey: "appVocabPrompts") as? [String: String] {
            appState.appVocabPrompts = saved
        }
        if let saved = UserDefaults.standard.dictionary(forKey: AppState.appAutoSubmitDefaultsKey) as? [String: Bool] {
            appState.appAutoSubmit = saved
        }
        if let saved = UserDefaults.standard.dictionary(
            forKey: AppState.inputDeviceWarmPreferencesDefaultsKey
        ) as? [String: Bool] {
            appState.inputDeviceKeepWarmByUID = saved
        }
        syncRunOnStartupState()
        configureHotkeyFromDefaults()

        await requestPermissions()

        let model = STTModelDefinition.allModels.first { $0.id == defaultID }
            ?? STTModelDefinition.default
        await loadModel(model)

        setupHotkey()
        startInputDeviceMonitoring()
    }

    @MainActor
    private func startInputDeviceMonitoring() {
        inputDeviceMonitor.start { [weak appState] snapshot in
            Task { @MainActor in
                guard let appState else { return }

                let previousDeviceUID = appState.activeInputDeviceUID
                let wasRecording = appState.phase == .recording
                appState.applyInputDeviceSnapshot(snapshot)

                if let active = snapshot.activeDevice {
                    let keepWarm = appState.keepWarmPreference(for: active.uid)
                    Self.logger.info(
                        "Active input device: \(active.name, privacy: .public) (keepWarm=\(keepWarm, privacy: .public))"
                    )
                }

                if previousDeviceUID != nil, previousDeviceUID != snapshot.activeDevice?.uid {
                    Self.logger.info(
                        "Input device changed (wasRecording=\(wasRecording, privacy: .public)); refreshing audio engine"
                    )
                    await reconfigureAudioForCurrentDevice(wasRecording: wasRecording)
                } else {
                    applyActiveInputWarmPolicy()
                }
            }
        }
    }

    @MainActor
    private func warmUpAudio(force: Bool = false) {
        guard appState.hasMicrophonePermission else { return }
        guard force || appState.activeInputDeviceKeepWarm else { return }
        guard !audioRecorder.isWarmedUp else { return }

        do {
            try audioRecorder.warmUp(
                onLevel: { [appState] level in
                    Task { @MainActor in
                        appState.audioLevel = level
                    }
                },
                onSpectrum: { [appState] bands in
                    Task { @MainActor in
                        appState.spectrumBands = bands
                    }
                }
            )
        } catch {
            Self.logger.error("Failed to warm up audio: \(error.localizedDescription, privacy: .public)")
        }
    }

    @MainActor
    private func selectInputDevice(uid: String) {
        guard !isAudioBusy else { return }
        let selected = inputDeviceMonitor.setDefaultInputDevice(uid: uid)
        if !selected {
            _ = appState.transition(to: .error("Could not switch to the selected input device."))
            resetAfterDelay(seconds: 2)
        }
    }

    @MainActor
    private func setActiveInputDeviceKeepWarm(_ keepWarm: Bool) {
        appState.setKeepWarmForActiveInputDevice(keepWarm)
        Task { @MainActor in
            applyActiveInputWarmPolicy()
        }
    }

    @MainActor
    private func reconfigureAudioForCurrentDevice(wasRecording: Bool) async {
        let shouldKeepWarm = appState.activeInputDeviceKeepWarm
        let shouldRunEngine = shouldKeepWarm || wasRecording

        audioRecorder.coolDown()
        appState.audioLevel = 0
        appState.spectrumBands = Array(repeating: 0, count: AudioRecorder.spectrumBandCount)

        guard shouldRunEngine, appState.hasMicrophonePermission else { return }
        try? await Task.sleep(for: .milliseconds(100))
        warmUpAudio(force: true)
        if wasRecording, audioRecorder.isWarmedUp {
            audioRecorder.startCapture()
            Self.logger.info("Restarted capture on newly selected input device")
        }
    }

    @MainActor
    private func applyActiveInputWarmPolicy() {
        guard appState.hasMicrophonePermission else { return }
        let shouldKeepWarm = appState.activeInputDeviceKeepWarm

        if shouldKeepWarm {
            warmUpAudio()
            return
        }

        if !isAudioBusy, audioRecorder.isWarmedUp {
            audioRecorder.coolDown()
            appState.audioLevel = 0
            appState.spectrumBands = Array(repeating: 0, count: AudioRecorder.spectrumBandCount)
        }
    }

    @MainActor
    private var isAudioBusy: Bool {
        switch appState.phase {
        case .recording, .transcribing, .pasting:
            return true
        default:
            return false
        }
    }

    // MARK: - Startup Login Item

    @MainActor
    private func toggleRunOnStartup() {
        let service = SMAppService.mainApp
        let statusBefore = service.status
        let shouldDisable = isRunOnStartupEnabled(statusBefore)
        let action = shouldDisable ? "disable" : "enable"

        do {
            if shouldDisable {
                try service.unregister()
            } else {
                try service.register()
            }

            appState.runOnStartupError = nil
        } catch {
            appState.runOnStartupError =
                "Could not \(action) Run on Startup: \(error.localizedDescription)"
            Self.logger.error(
                "Run on startup toggle failed. action=\(action, privacy: .public) statusBefore=\(String(describing: statusBefore), privacy: .public) error=\(error.localizedDescription, privacy: .public)"
            )
        }

        appState.runOnStartupEnabled = isRunOnStartupEnabled(service.status)
    }

    @MainActor
    private func syncRunOnStartupState() {
        appState.runOnStartupEnabled = isRunOnStartupEnabled(SMAppService.mainApp.status)
    }

    private func isRunOnStartupEnabled(_ status: SMAppService.Status) -> Bool {
        status == .enabled || status == .requiresApproval
    }

    // MARK: - Hotkey Handling

    @MainActor
    private func setupHotkey() {
        hotkeyMonitor.binding = appState.hotkeyBinding

        hotkeyMonitor.onKeyDown = {
            Task { @MainActor in
                await handleKeyDown()
            }
        }
        hotkeyMonitor.onKeyUp = {
            Task { @MainActor in
                await handleKeyUp()
            }
        }
        hotkeyMonitor.onModifierWhileHeld = { keyCode in
            Task { @MainActor in
                // Command pressed while hotkey is held → lock into toggle mode
                if keyCode == 55 || keyCode == 54 { // left/right Command
                    guard appState.phase == .recording, !appState.isLockedMode else { return }
                    appState.isLockedMode = true
                    skipNextKeyUp = true
                }
            }
        }
        hotkeyMonitor.start()
    }

    @MainActor
    private func configureHotkeyFromDefaults() {
        let (binding, fallbackMessage) = HotkeyBinding.load()
        appState.hotkeyBinding = binding
        appState.hotkeySettingsMessage = fallbackMessage
    }

    @MainActor
    private func updateHotkeyBinding(_ binding: HotkeyBinding) {
        appState.hotkeyBinding = binding
        appState.hotkeySettingsMessage = nil
        hotkeyMonitor.binding = binding
        binding.save()
    }

    @MainActor
    private func handleKeyDown() async {
        refreshPermissionState()

        guard !appState.isEditingHotkey else { return }

        // In toggle mode, stop on keyUp (not keyDown) so the user can
        // hold hotkey + press ESC to cancel before releasing.
        if appState.phase == .recording && appState.isLockedMode {
            return
        }

        // Allow starting a new recording even if the previous one is still wrapping up
        // (e.g. pasting or showing the success animation).
        if appState.phase == .transcribing || appState.phase == .pasting {
            overlayManager.hide()
            _ = appState.transition(to: .idle)
        }

        guard appState.phase == .idle else { return }
        guard appState.modelStatus == .loaded else { return }

        guard appState.hasRequiredPermissions else {
            _ = appState.transition(
                to: .error("Missing \(appState.missingPermissionSummary). Open the menu to grant access.")
            )
            resetAfterDelay(seconds: 4)
            return
        }

        _ = appState.transition(to: .recording)
        appState.partialTranscript = ""
        appState.partialTranscriptCommittedUTF16 = 0
        keyDownTime = Date()
        appState.isLockedMode = false
        hotkeyMonitor.allowExtraModifiers = true

        // Pause media if setting is enabled.
        if MediaController.isEnabled {
            MediaController.pauseIfPlaying()
        }
        overlayManager.show(appState: appState)
        startRecordingTimeout()
        installEscapeMonitor()
        playStartSound()

        // Create streaming session with per-app language preference
        streamingSession = transcriptionService.createSession(
            language: appState.currentLanguage,
            prompt: appState.vocabPrompt
        )

        // If audio is warm, just start capturing (instant).
        if audioRecorder.isWarmedUp {
            audioRecorder.startCapture()
        } else {
            do {
                try audioRecorder.start { [appState] (level: Float) in
                    Task { @MainActor in
                        appState.audioLevel = level
                    }
                }
            } catch {
                cancelRecordingTimeout()
                _ = appState.transition(to: .error(error.localizedDescription))
                overlayManager.hide()
                resetAfterDelay()
                return
            }
        }

        // Start streaming transcription
        startStreamingTranscription()
    }

    @MainActor
    private func startStreamingTranscription() {
        startStreamingLoop()
    }

    /// Start (or restart) the streaming loop. On "over", this pastes + submits
    /// then calls itself to keep recording. On "over and out", it stops entirely.
    @MainActor
    private func startStreamingLoop() {
        let transcriptionService = self.transcriptionService
        let audioRecorder = self.audioRecorder
        let appState = self.appState
        let session = self.streamingSession!

        streamingTask = Task.detached { () -> StreamingResult in
            var processedCount = 0
            var lastText = ""
            var signal: StreamingSignal = .none

            while !Task.isCancelled {
                // Check phase on main actor
                let isRecording = await MainActor.run { appState.phase == .recording }
                guard isRecording else { break }

                let allSamples = await MainActor.run { audioRecorder.peekCapture() }

                guard allSamples.count > processedCount + 800 else {
                    try? await Task.sleep(for: .milliseconds(30))
                    continue
                }

                let newChunk = Array(allSamples[processedCount...])
                processedCount = allSamples.count

                let update: StreamingTranscriptUpdate? = transcriptionService.feed(session: session, samples: newChunk)

                if let update {
                    let trimmed = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        lastText = trimmed
                        let committedUTF16 = min(
                            max(0, update.committedUTF16Count),
                            (trimmed as NSString).length
                        )
                        await MainActor.run {
                            appState.partialTranscript = trimmed
                            appState.partialTranscriptCommittedUTF16 = committedUTF16
                        }

                        // Check "over and out" first (higher priority).
                        if trimmed.range(of: #"(?i)[.!?,]\s+over\s+and\s+out\.?\s*$"#, options: .regularExpression) != nil {
                            signal = .overAndOut
                            break
                        }
                        // Then check "over".
                        if trimmed.range(of: #"(?i)[.!?,]\s+over\.?\s*$"#, options: .regularExpression) != nil {
                            signal = .over
                            break
                        }
                    }
                }
            }

            // Strip the trigger phrase from the text.
            lastText = HarkApp.stripTrigger(lastText, signal: signal)

            return StreamingResult(text: lastText, signal: signal, processedSampleCount: processedCount)
        }

        // Watcher: handles "over" and "over and out" when they fire mid-recording.
        Task { @MainActor in
            guard let result = await streamingTask?.value else { return }
            guard result.signal != .none && appState.phase == .recording else { return }

            print("[hark] signal: \(result.signal) text='\(result.text)'")
            appState.partialTranscript = ""
            appState.partialTranscriptCommittedUTF16 = 0

            if !result.text.isEmpty {
                // Paste + Enter for the current sentence.
                // Temporarily leave recording to paste, then come back if "over" (not "over and out").
                _ = appState.transition(to: .transcribing)
                _ = appState.transition(to: .pasting)
                do {
                    try await PasteController.paste(result.text, submit: true)
                } catch {
                    print("[hark] paste error: \(error)")
                }
                appState.addToHistory(result.text)
            }

            if result.signal == .over {
                // Keep recording — reset audio buffer and start a fresh streaming session.
                if audioRecorder.isWarmedUp {
                    _ = audioRecorder.stopCapture()
                    audioRecorder.startCapture()
                }
                _ = appState.transition(to: .idle)
                _ = appState.transition(to: .recording)
                appState.partialTranscript = ""
                appState.partialTranscriptCommittedUTF16 = 0
                streamingSession = transcriptionService.createSession(
                    language: appState.currentLanguage
                )
                startStreamingLoop()
            } else {
                // "Over and out" — stop recording entirely.
                cancelRecordingTimeout()
                removeEscapeMonitor()
                appState.isLockedMode = false
                keyDownTime = nil
                streamingTask = nil
                streamingSession = nil

                if audioRecorder.isWarmedUp {
                    _ = audioRecorder.stopCapture()
                } else {
                    _ = audioRecorder.stop()
                }
                appState.audioLevel = 0
                if MediaController.isEnabled { MediaController.resumeIfPaused() }
                _ = appState.transition(to: .idle)
                overlayManager.hideWithResult(.success)
            }
        }
    }

    /// Strip "over" or "over and out" from the end of the text.
    private nonisolated static func stripTrigger(_ text: String, signal: StreamingSignal) -> String {
        guard signal != .none else { return text }
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let lower = s.lowercased()

        let suffix: String
        switch signal {
        case .overAndOut:
            suffix = "over and out"
        case .over:
            suffix = "over"
        case .none:
            return s
        }

        // Strip trailing period/comma, then the trigger word(s).
        if s.hasSuffix(".") || s.hasSuffix(",") { s = String(s.dropLast()) }
        s = s.trimmingCharacters(in: .whitespaces)
        if s.lowercased().hasSuffix(suffix) {
            s = String(s.dropLast(suffix.count))
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }

    @MainActor
    private func handleKeyUp() async {
        guard appState.phase == .recording else { return }

        // After ⌘-lock, skip the keyUp from releasing the original hotkey hold.
        if skipNextKeyUp {
            skipNextKeyUp = false
            return
        }

        // In locked mode, this keyUp is the "stop and submit" action.
        if appState.isLockedMode {
            await stopRecordingAndTranscribe()
            return
        }

        // Check if this was a quick press (toggle mode)
        if let downTime = keyDownTime {
            let pressDuration = Date().timeIntervalSince(downTime)
            if pressDuration < Self.toggleModeThresholdSeconds {
                appState.isLockedMode = true
                return
            }
        }

        await stopRecordingAndTranscribe()
    }

    @MainActor
    private func startRecordingTimeout() {
        cancelRecordingTimeout()
        recordingTimeoutTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(Self.maxRecordingDurationSeconds))
            } catch {
                return
            }

            guard appState.phase == .recording else { return }
            recordingTimeoutTask = nil
            await stopRecordingAndTranscribe(cancelTimeoutTask: false)
        }
    }

    @MainActor
    private func cancelRecordingTimeout() {
        recordingTimeoutTask?.cancel()
        recordingTimeoutTask = nil
    }

    @MainActor
    private func stopRecordingAndTranscribe(cancelTimeoutTask: Bool = true, skipPaste: Bool = false, forceSubmit: Bool = false) async {
        guard appState.phase == .recording else { return }

        _ = appState.transition(to: .transcribing)

        if cancelTimeoutTask {
            cancelRecordingTimeout()
        }
        removeEscapeMonitor()
        appState.isLockedMode = false
        keyDownTime = nil
        hotkeyMonitor.allowExtraModifiers = false

        // Stop the streaming loop.
        let stask = streamingTask
        streamingTask = nil

        // Stop capture and use the definitive captured buffer at stop time.
        // Using `peekCapture()` here can miss tail audio arriving between peek and stop.
        let allSamples: [Float]
        if audioRecorder.isWarmedUp {
            allSamples = audioRecorder.stopCapture()
        } else {
            allSamples = audioRecorder.stop()
        }
        appState.audioLevel = 0
        if !appState.activeInputDeviceKeepWarm, audioRecorder.isWarmedUp {
            audioRecorder.coolDown()
            appState.spectrumBands = Array(repeating: 0, count: AudioRecorder.spectrumBandCount)
        }

        // Feed any remaining samples and finalize the session to get the complete transcript.
        var text = appState.partialTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        Self.logger.warning("[hark] stop: partial='\(text, privacy: .public)' sessionExists=\(streamingSession != nil) samples=\(allSamples.count)")
        if let session = streamingSession {
            appState.isFinishing = true

            // Wait for the streaming loop to exit so we don't race on the session.
            let result = await stask?.value
            let processedCount = result?.processedSampleCount ?? 0
            let remainingCount = max(0, allSamples.count - processedCount)

            Self.logger.warning("[hark] finalize: processed=\(processedCount) total=\(allSamples.count) remaining=\(remainingCount)")

            // Feed remaining unprocessed audio and finalize — off main thread.
            let transcriptionService = self.transcriptionService
            let remaining = processedCount < allSamples.count ? Array(allSamples[processedCount...]) : nil
            let padSampleCount = max(
                0,
                Int((Self.finalizationSilencePaddingSeconds * Self.transcriptionSampleRate).rounded())
            )
            var finalizeChunk = remaining ?? []
            if padSampleCount > 0 {
                finalizeChunk.append(contentsOf: repeatElement(Float(0), count: padSampleCount))
            }
            let finalText: String? = await Task.detached {
                if !finalizeChunk.isEmpty {
                    _ = transcriptionService.feed(session: session, samples: finalizeChunk)
                }
                return transcriptionService.finish(session: session)
            }.value
            if let finalText {
                let trimmed = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    text = trimmed
                }
            }

            // If streaming produced nothing, fall back to a single-shot
            // transcription of all captured audio.
            if text.isEmpty && !allSamples.isEmpty && !Self.isEffectivelySilent(allSamples) {
                let transcriptionService = self.transcriptionService
                var samples = allSamples
                if padSampleCount > 0 {
                    samples.append(contentsOf: repeatElement(Float(0), count: padSampleCount))
                }
                let fallbackText: String? = await Task.detached {
                    transcriptionService.transcribeSamples(samples)
                }.value
                if let fallbackText {
                    let trimmed = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        text = trimmed
                    }
                }
            }

            // Show the final transcript in the overlay (no animation).
            appState.partialTranscript = text
            appState.partialTranscriptCommittedUTF16 = (text as NSString).length
            appState.isFinishing = false
        }
        streamingSession = nil

        await finishAndPaste(text: text, skipPaste: skipPaste, forceSubmit: forceSubmit)
    }

    private nonisolated static func isEffectivelySilent(_ samples: [Float], rmsThreshold: Float = 0.006) -> Bool {
        guard !samples.isEmpty else { return true }
        let sumSquares = samples.reduce(Float(0)) { $0 + ($1 * $1) }
        let rms = sqrtf(sumSquares / Float(samples.count))
        return rms < rmsThreshold
    }

    @MainActor
    private func finishAndPaste(text: String, skipPaste: Bool = false, forceSubmit: Bool = false) async {
        // Don't clear partialTranscript here — let the dismiss animation show the final text.
        if MediaController.isEnabled { MediaController.resumeIfPaused() }

        if text.isEmpty || skipPaste {
            _ = appState.transition(to: .idle)
            overlayManager.hideWithResult(.cancelled)
            if !text.isEmpty { appState.addToHistory(text) }
            return
        }

        appState.addToHistory(text)

        // Log for training data collection
        Task {
            await TranscriptionLogger.shared.log(
                text: text,
                app: NSWorkspace.shared.frontmostApplication?.bundleIdentifier
            )
        }

        let shouldSubmit = forceSubmit || PasteController.isReturnKeyPressed() || appState.currentAutoSubmit

        // Paste immediately (don't wait for overlay dismiss).
        _ = appState.transition(to: .pasting)
        let pasteTask = Task {
            try await PasteController.paste(text, submit: shouldSubmit)
        }

        // Keep overlay showing final text briefly, then dismiss.
        try? await Task.sleep(for: .milliseconds(350))
        overlayManager.hideWithResult(.success)

        do {
            try await pasteTask.value
            _ = appState.transition(to: .idle)
        } catch {
            _ = appState.transition(to: .error(error.localizedDescription))
            overlayManager.hide()
            resetAfterDelay(seconds: 1)
        }
    }

    @MainActor
    private func installEscapeMonitor() {
        removeEscapeMonitor()

        let currentAppState = appState
        let currentHotkeyMonitor = hotkeyMonitor

        recordingControlInterceptor.shouldIntercept = { keyCode in
            guard currentAppState.phase == .recording else { return false }
            guard keyCode == UInt16(kVK_Escape) || keyCode == UInt16(kVK_Return) else { return false }

            if currentAppState.isLockedMode {
                // In locked mode, require that the configured hotkey keys are still held.
                guard currentHotkeyMonitor.binding.keyCodeSet.isSubset(of: currentHotkeyMonitor.pressedKeyCodes) else {
                    return false
                }
            }

            return true
        }

        recordingControlInterceptor.onIntercept = { keyCode in
            if keyCode == UInt16(kVK_Escape) {
                NotificationCenter.default.post(name: .cancelRecording, object: nil)
            } else if keyCode == UInt16(kVK_Return) {
                NotificationCenter.default.post(name: .submitRecording, object: nil)
            }
        }
        recordingControlInterceptor.start()

        let cancelObserver = NotificationCenter.default.addObserver(
            forName: .cancelRecording,
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                await self.stopRecordingAndTranscribe(skipPaste: true, forceSubmit: false)
            }
        }
        recordingControlObservers.append(cancelObserver)

        let submitObserver = NotificationCenter.default.addObserver(
            forName: .submitRecording,
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                await self.stopRecordingAndTranscribe(skipPaste: false, forceSubmit: true)
            }
        }
        recordingControlObservers.append(submitObserver)
    }

    @MainActor
    private func removeEscapeMonitor() {
        recordingControlInterceptor.stop()
        recordingControlInterceptor.shouldIntercept = nil
        recordingControlInterceptor.onIntercept = nil

        for observer in recordingControlObservers {
            NotificationCenter.default.removeObserver(observer)
        }
        recordingControlObservers.removeAll()
    }

    // MARK: - Model Management

    @MainActor
    private func selectModel(_ model: STTModelDefinition) {
        switch appState.phase {
        case .recording, .transcribing, .pasting:
            return
        default:
            break
        }

        appState.selectedModelID = model.id
        UserDefaults.standard.set(model.id, forKey: "selectedModelID")

        _ = startModelLoad(model)
    }

    @MainActor
    private func loadModel(_ model: STTModelDefinition) async {
        let task = startModelLoad(model)
        await task.value
    }

    @MainActor
    private func deleteLocalModel(_ model: STTModelDefinition) {
        let cacheDir = STTModelDefinition.cacheDirectory
        let modelDir = URL(fileURLWithPath: cacheDir).appendingPathComponent(model.cacheDirName)

        if appState.selectedModelID == model.id {
            modelLoadTask?.cancel()
            transcriptionService.unloadModel()
        }

        do {
            if FileManager.default.fileExists(atPath: modelDir.path) {
                try FileManager.default.removeItem(at: modelDir)
            }
            appState.downloadedModelIDs.remove(model.id)

            if appState.selectedModelID == model.id {
                appState.modelStatus = .notLoaded
                if case .loading = appState.phase {
                    _ = appState.transition(to: .idle)
                }
            }
        } catch {
            _ = appState.transition(to: .error("Failed to delete model: \(error.localizedDescription)"))
            resetAfterDelay()
        }
    }

    @discardableResult
    @MainActor
    private func startModelLoad(_ model: STTModelDefinition) -> Task<Void, Never> {
        modelLoadTask?.cancel()
        modelLoadGeneration &+= 1
        let generation = modelLoadGeneration
        let modelID = model.id

        appState.modelStatus = .loading
        _ = appState.transition(to: .loading("Checking model files..."))

        let task = Task(priority: .userInitiated) {
            do {
                try await transcriptionService.loadModel(
                    model: model,
                    cacheDir: STTModelDefinition.cacheDirectory
                ) { update in
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == modelID else { return }

                    switch update {
                    case .downloading(let progress):
                        appState.modelStatus = .downloading(progress: progress)
                        _ = appState.transition(to: .loading("Downloading model..."))
                    case .initializing:
                        appState.modelStatus = .loading
                        _ = appState.transition(to: .loading("Initializing model..."))
                    }
                }

                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == modelID else { return }

                    appState.modelStatus = .loaded
                    appState.downloadedModelIDs.insert(modelID)
                    _ = appState.transition(to: .idle)
                    modelLoadTask = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    modelLoadTask = nil

                    switch appState.modelStatus {
                    case .loading, .downloading:
                        appState.modelStatus = .notLoaded
                    default:
                        break
                    }
                    if case .loading = appState.phase {
                        _ = appState.transition(to: .idle)
                    }
                }
            } catch {
                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == modelID else { return }

                    appState.modelStatus = .error(error.localizedDescription)
                    _ = appState.transition(to: .error("Model load failed: \(error.localizedDescription)"))
                    modelLoadTask = nil
                    resetAfterDelay()
                }
            }
        }

        modelLoadTask = task
        return task
    }

    // MARK: - Permissions

    @MainActor
    private func requestPermissions() async {
        let microphoneGranted = await AudioRecorder.requestPermission()
        appState.microphonePermission = microphoneGranted ? .granted : .denied

        if !PasteController.hasAccessibilityPermission {
            PasteController.requestAccessibilityPermission()
        }

        appState.accessibilityPermission =
            PasteController.hasAccessibilityPermission ? .granted : .denied
    }

    @MainActor
    private func requestMicrophonePermissionFromMenu() async {
        let granted = await AudioRecorder.requestPermission()
        refreshPermissionState()

        if granted {
            applyActiveInputWarmPolicy()
            return
        }
        openPrivacySettings(anchor: "Privacy_Microphone")
    }

    @MainActor
    private func requestAccessibilityPermissionFromMenu() {
        PasteController.requestAccessibilityPermission()
        refreshPermissionState()

        if !appState.hasAccessibilityPermission {
            openPrivacySettings(anchor: "Privacy_Accessibility")
        }
    }

    @MainActor
    private func refreshPermissionState() {
        appState.microphonePermission = microphonePermissionStatus()
        appState.accessibilityPermission =
            PasteController.hasAccessibilityPermission ? .granted : .denied
    }

    private func microphonePermissionStatus() -> PermissionStatus {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return .granted
        case .denied, .restricted:
            return .denied
        case .notDetermined:
            return .unknown
        @unknown default:
            return .denied
        }
    }

    private func openPrivacySettings(anchor: String) {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)"
        ) else {
            return
        }

        NSWorkspace.shared.open(url)
    }

    // MARK: - Helpers

    @MainActor
    private func resetAfterDelay(seconds: Int = 3) {
        Task {
            try? await Task.sleep(for: .seconds(seconds))
            if case .error = appState.phase {
                _ = appState.transition(to: .idle)
            }
        }
    }

    // MARK: - Fonts

    private func registerBundledFonts() {
        guard let resourceURL = Bundle.main.resourceURL else { return }
        let fontExtensions: Set<String> = ["ttf", "otf"]
        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL, includingPropertiesForKeys: nil
        ) else { return }
        for case let url as URL in enumerator where fontExtensions.contains(url.pathExtension.lowercased()) {
            CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
        }
    }

    // MARK: - Sound Effects

    private func getSystemOutputVolume() -> Float {
        var defaultOutputDeviceID = AudioDeviceID(0)
        var defaultOutputDeviceIDSize = UInt32(MemoryLayout.size(ofValue: defaultOutputDeviceID))

        var getDefaultOutputDevicePropertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let status1 = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &getDefaultOutputDevicePropertyAddress,
            0,
            nil,
            &defaultOutputDeviceIDSize,
            &defaultOutputDeviceID
        )

        guard status1 == noErr else { return 1.0 }

        var volume = Float32(0.0)
        var volumeSize = UInt32(MemoryLayout.size(ofValue: volume))

        var volumePropertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwareServiceDeviceProperty_VirtualMainVolume,
            mScope: kAudioDevicePropertyScopeOutput,
            mElement: kAudioObjectPropertyElementMain
        )

        let status2 = AudioObjectGetPropertyData(
            defaultOutputDeviceID,
            &volumePropertyAddress,
            0,
            nil,
            &volumeSize,
            &volume
        )

        guard status2 == noErr else { return 1.0 }
        return volume
    }

    private func calculateSoundVolume() -> Float {
        let systemVolume = getSystemOutputVolume()
        let effectiveSystemVol = max(systemVolume, 0.2)
        return min(0.2 / effectiveSystemVol, 1.0)
    }

    private func playStartSound() {
        guard let sound = NSSound(named: "Tink") else { return }
        sound.volume = calculateSoundVolume()
        sound.play()
    }
}
