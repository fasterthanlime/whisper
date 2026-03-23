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
    @State private var escapeMonitor: Any?
    @State private var notificationObserver: NSObjectProtocol?

    private static let maxRecordingDurationSeconds = AudioRecorder.defaultMaximumDuration
    private static let toggleModeThresholdSeconds: TimeInterval = 0.3
    private static let minimumSpeechDurationSeconds = 0.2
    private static let transcriptionSampleRate = 16_000.0

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
        syncRunOnStartupState()
        configureHotkeyFromDefaults()

        await requestPermissions()

        let model = STTModelDefinition.allModels.first { $0.id == defaultID }
            ?? STTModelDefinition.default
        await loadModel(model)

        setupHotkey()
        startInputDeviceMonitoring()
        warmUpAudio()
    }

    @MainActor
    private func startInputDeviceMonitoring() {
        inputDeviceMonitor.start { [weak appState, weak audioRecorder] deviceName in
            Task { @MainActor in
                guard let appState, let audioRecorder else { return }

                let previousDevice = appState.activeInputDeviceName
                appState.activeInputDeviceName = deviceName

                if previousDevice != nil, previousDevice != deviceName, audioRecorder.isWarmedUp {
                    let wasRecording = appState.phase == .recording
                    Self.logger.info("Input device changed (wasRecording=\(wasRecording)), reinitializing audio")

                    audioRecorder.coolDown()
                    try? await Task.sleep(for: .milliseconds(100))

                    do {
                        try audioRecorder.warmUp(
                            onLevel: { level in
                                Task { @MainActor in
                                    appState.audioLevel = level
                                }
                            },
                            onSpectrum: { bands in
                                Task { @MainActor in
                                    appState.spectrumBands = bands
                                }
                            }
                        )

                        // If we were recording, restart capture on the new device.
                        if wasRecording {
                            audioRecorder.startCapture()
                            Self.logger.info("Restarted capture on new device")
                        }
                    } catch {
                        Self.logger.error("Failed to reinitialize audio after device change: \(error.localizedDescription, privacy: .public)")
                    }
                }
            }
        }
    }

    @MainActor
    private func warmUpAudio() {
        guard appState.hasMicrophonePermission else { return }
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

                let text: String? = transcriptionService.feed(session: session, samples: newChunk)

                if let text {
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        lastText = trimmed
                        await MainActor.run { appState.partialTranscript = trimmed }

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

        // Grab all remaining audio before stopping capture.
        let allSamples = audioRecorder.peekCapture()

        if audioRecorder.isWarmedUp {
            _ = audioRecorder.stopCapture()
        } else {
            _ = audioRecorder.stop()
        }
        appState.audioLevel = 0

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
            let finalText: String? = await Task.detached {
                if let remaining, !remaining.isEmpty {
                    _ = transcriptionService.feed(session: session, samples: remaining)
                }
                return transcriptionService.finish(session: session)
            }.value
            if let finalText {
                let trimmed = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    text = trimmed
                }
            }

            // If streaming produced nothing (e.g. too short for a chunk boundary),
            // fall back to a single-shot transcription of all captured audio.
            if text.isEmpty && !allSamples.isEmpty {
                let transcriptionService = self.transcriptionService
                let samples = allSamples
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
            appState.isFinishing = false
        }
        streamingSession = nil

        await finishAndPaste(text: text, skipPaste: skipPaste, forceSubmit: forceSubmit)
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

        let shouldSubmit = forceSubmit || PasteController.isReturnKeyPressed() || Self.isAutoSubmitApp()

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
        escapeMonitor = NSEvent.addGlobalMonitorForEvents(matching: .keyDown) { [self] event in
            guard currentAppState.phase == .recording else { return }

            let isEscape = event.keyCode == UInt16(kVK_Escape)
            let isReturn = event.keyCode == UInt16(kVK_Return)

            guard isEscape || isReturn else { return }

            if appState.isLockedMode {
                // Check if the hotkey's keys are still pressed (not exact isHeld,
                // because the ESC/Return key itself gets added to pressedKeyCodes
                // before this monitor fires, breaking the exact match).
                guard currentHotkeyMonitor.binding.keyCodeSet.isSubset(of: currentHotkeyMonitor.pressedKeyCodes) else { return }
            }

            if isEscape {
                NotificationCenter.default.post(name: .cancelRecording, object: nil)
            } else if isReturn {
                NotificationCenter.default.post(name: .submitRecording, object: nil)
            }
        }

        notificationObserver = NotificationCenter.default.addObserver(
            forName: .cancelRecording,
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                await self.stopRecordingAndTranscribe(skipPaste: true, forceSubmit: false)
            }
        }

        NotificationCenter.default.addObserver(
            forName: .submitRecording,
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                await self.stopRecordingAndTranscribe(skipPaste: false, forceSubmit: true)
            }
        }
    }

    @MainActor
    private func removeEscapeMonitor() {
        if let monitor = escapeMonitor {
            NSEvent.removeMonitor(monitor)
            escapeMonitor = nil
        }
        if let observer = notificationObserver {
            NotificationCenter.default.removeObserver(observer)
            notificationObserver = nil
        }
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

        guard !granted else { return }
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

    // MARK: - Auto-Submit

    /// Apps where dictated text should automatically be followed by Enter.
    private static let autoSubmitBundleIDs: Set<String> = [
        "com.googlecode.iterm2",
        "com.mitchellh.ghostty",
    ]

    private static func isAutoSubmitApp() -> Bool {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return false }
        return autoSubmitBundleIDs.contains(bundleID)
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
