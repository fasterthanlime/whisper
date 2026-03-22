import AppKit
import AudioToolbox
import AVFoundation
import CoreAudio
import os
import ServiceManagement
import SwiftUI

@main
struct WhisperApp: App {
    private static let sharedTranscriptionService = TranscriptionService()
    private static let sharedVADService = VADService()
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "whisper",
        category: "startup"
    )

    @State private var appState = AppState()
    @State private var overlayManager = OverlayManager()
    @State private var audioRecorder = AudioRecorder()
    private let transcriptionService = sharedTranscriptionService
    private let vadService = sharedVADService
    @State private var hotkeyMonitor = HotkeyMonitor()
    @State private var modelLoadTask: Task<Void, Never>?
    // UI stale-completion guard. Kept separate from TranscriptionService.loadGeneration,
    // which guards actor-owned model state.
    @State private var modelLoadGeneration: UInt64 = 0
    @State private var hasLaunched = false
    @State private var recordingTimeoutTask: Task<Void, Never>?
    @State private var vadProcessingTask: Task<Void, Never>?
    @State private var accumulatedTranscript: String = ""
    @State private var hadSpeechActivity: Bool = false
    @State private var inputDeviceMonitor = InputDeviceMonitor()

    private static let maxRecordingDurationSeconds = AudioRecorder.defaultMaximumDuration
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
                        await deleteLocalModel(model)
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

    /// Create an NSImage from an SF Symbol at an exact pixel size, marked as template
    /// so macOS handles light/dark menu bar correctly.
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
        let defaultRepoID = UserDefaults.standard.string(forKey: "selectedModelID")
            ?? STTModelDefinition.default.repoID

        appState.selectedModelID = defaultRepoID
        syncRunOnStartupState()
        configureHotkeyFromDefaults()

        // Request permissions
        await requestPermissions()

        // Discover locally cached models
        await refreshDownloadedModels()

        // Load STT and VAD models in parallel
        async let sttLoad: () = loadModel(repoID: defaultRepoID)
        async let vadLoad: () = loadVADModel()
        _ = await (sttLoad, vadLoad)

        // Start listening for hotkey events
        setupHotkey()

        // Start monitoring input device changes
        startInputDeviceMonitoring()

        // Warm up audio device (keeps mic ready with pre-buffer)
        warmUpAudio()
    }

    @MainActor
    private func startInputDeviceMonitoring() {
        inputDeviceMonitor.start { [weak appState, weak audioRecorder] deviceName in
            Task { @MainActor in
                guard let appState, let audioRecorder else { return }

                let previousDevice = appState.activeInputDeviceName
                appState.activeInputDeviceName = deviceName

                // If device changed and we're warmed up, reinitialize audio
                if previousDevice != nil, previousDevice != deviceName, audioRecorder.isWarmedUp {
                    Self.logger.info("Input device changed, reinitializing audio")
                    audioRecorder.coolDown()

                    // Small delay to let the system settle
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

    @MainActor
    private func loadVADModel() async {
        do {
            try await vadService.loadModel()
            Self.logger.info("VAD model loaded successfully")
        } catch {
            Self.logger.error("Failed to load VAD model: \(error.localizedDescription, privacy: .public)")
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
        accumulatedTranscript = ""
        hadSpeechActivity = false
        overlayManager.show(appState: appState)
        startRecordingTimeout()
        playStartSound()

        // Reset VAD state for new recording
        await vadService.reset()

        // If audio is warm, just start capturing (instant).
        // Otherwise, start from cold (may have latency).
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

        // Start VAD-driven transcription
        startVADProcessing()
    }

    @MainActor
    private func startVADProcessing() {
        vadProcessingTask?.cancel()
        vadProcessingTask = Task { @MainActor in
            // Process audio chunks as they come in
            // We'll poll for new audio and feed it to VAD
            let chunkSize = 1600  // 100ms at 16kHz
            var lastProcessedCount = 0

            while !Task.isCancelled && appState.phase == .recording {
                let allSamples = audioRecorder.peekCapture()

                // Check if we have new audio to process
                guard allSamples.count > lastProcessedCount + chunkSize else {
                    try? await Task.sleep(for: .milliseconds(50))
                    continue
                }

                // Extract new chunk
                let newChunk = Array(allSamples[lastProcessedCount...])
                lastProcessedCount = allSamples.count

                // Feed to VAD
                do {
                    let events = try await vadService.processAudio(newChunk)

                    for event in events {
                        guard !Task.isCancelled && appState.phase == .recording else { break }

                        switch event {
                        case .speechStarted:
                            Self.logger.debug("VAD: speech started")
                            hadSpeechActivity = true

                        case .speechSegment(let audio):
                            // Speech segment detected - transcribe it
                            Self.logger.error("VAD: segment detected, \(audio.count) samples")
                            print("VAD SEGMENT: \(audio.count) samples")
                            if audio.count < 1600 {
                                print("VAD SEGMENT: too short, skipping")
                                continue
                            }
                            let text = await transcriptionService.transcribeLive(audio: audio)
                            print("VAD TRANSCRIPTION: '\(text ?? "nil")'")
                            if let text, !text.isEmpty {
                                guard !Task.isCancelled && appState.phase == .recording else { break }
                                if !accumulatedTranscript.isEmpty {
                                    accumulatedTranscript += " "
                                }
                                accumulatedTranscript += text
                                appState.partialTranscript = accumulatedTranscript
                                print("VAD PARTIAL: '\(accumulatedTranscript)'")
                            }

                        case .speechEnded:
                            Self.logger.debug("VAD: speech ended")
                        }
                    }
                } catch {
                    Self.logger.error("VAD processing error: \(error.localizedDescription, privacy: .public)")
                    print("VAD ERROR: \(error)")
                }
            }
        }
    }

    @MainActor
    private func handleKeyUp() async {
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
    private func stopRecordingAndTranscribe(cancelTimeoutTask: Bool = true) async {
        guard appState.phase == .recording else { return }

        if cancelTimeoutTask {
            cancelRecordingTimeout()
        }

        // Stop VAD processing
        vadProcessingTask?.cancel()
        vadProcessingTask = nil

        // Get final audio from recorder
        let samples: [Float]
        if audioRecorder.isWarmedUp {
            samples = audioRecorder.stopCapture()
        } else {
            samples = audioRecorder.stop()
        }
        appState.audioLevel = 0

        let minimumSamples = Int(Self.minimumSpeechDurationSeconds * Self.transcriptionSampleRate)
        guard samples.count >= minimumSamples else {
            overlayManager.hide()
            appState.partialTranscript = ""
            _ = appState.transition(to: .error("Recording too short. Hold the hotkey briefly and try again."))
            resetAfterDelay(seconds: 2)
            return
        }

        // Skip transcription if no voice activity was detected
        guard hadSpeechActivity else {
            overlayManager.hide()
            appState.partialTranscript = ""
            _ = appState.transition(to: .idle)
            playStopSound()
            return
        }

        // Get any remaining audio from VAD buffer
        let remainingAudio = await vadService.flush()

        // Start with accumulated transcript from completed segments
        var text = accumulatedTranscript.trimmingCharacters(in: .whitespacesAndNewlines)

        // Transcribe remaining audio if there is any
        if let remaining = remainingAudio, !remaining.isEmpty {
            _ = appState.transition(to: .transcribing)
            overlayManager.show(appState: appState)

            if let result = await transcriptionService.transcribeLive(audio: remaining) {
                let trimmed = result.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    if !text.isEmpty {
                        text += " "
                    }
                    text += trimmed
                }
            }
        } else if text.isEmpty {
            // No accumulated text and no remaining audio - transcribe all captured audio
            _ = appState.transition(to: .transcribing)
            overlayManager.show(appState: appState)

            if let result = await transcriptionService.transcribeLive(audio: samples) {
                text = result.trimmingCharacters(in: .whitespacesAndNewlines)
            }
        }

        appState.partialTranscript = ""

        guard !text.isEmpty else {
            _ = appState.transition(to: .error("No speech detected"))
            overlayManager.hide()
            resetAfterDelay()
            return
        }

        // Add to history
        appState.addToHistory(text)

        // Check if Return is held - if so, submit after pasting
        let shouldSubmit = PasteController.isReturnKeyPressed()

        _ = appState.transition(to: .pasting)
        do {
            try await PasteController.paste(text, submit: shouldSubmit)
            playStopSound()
            overlayManager.hide()
            _ = appState.transition(to: .idle)
        } catch {
            _ = appState.transition(to: .error(error.localizedDescription))
            overlayManager.hide()
            resetAfterDelay()
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

        appState.selectedModelID = model.repoID
        UserDefaults.standard.set(model.repoID, forKey: "selectedModelID")

        _ = startModelLoad(repoID: model.repoID)
    }

    @MainActor
    private func loadModel(repoID: String) async {
        let task = startModelLoad(repoID: repoID)
        await task.value
    }

    @MainActor
    private func refreshDownloadedModels() async {
        let repoIDs = STTModelDefinition.allModels.map(\.repoID)
        appState.downloadedModelRepoIDs = await transcriptionService.downloadedModelRepoIDs(for: repoIDs)
    }

    @MainActor
    private func deleteLocalModel(_ model: STTModelDefinition) async {
        let repoID = model.repoID

        if appState.selectedModelID == repoID {
            modelLoadTask?.cancel()
        }

        do {
            try await transcriptionService.deleteLocalModel(repoID: repoID)
            appState.downloadedModelRepoIDs.remove(repoID)

            if appState.selectedModelID == repoID {
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
    private func startModelLoad(repoID: String) -> Task<Void, Never> {
        modelLoadTask?.cancel()
        modelLoadGeneration &+= 1
        let generation = modelLoadGeneration

        appState.modelStatus = .loading
        _ = appState.transition(to: .loading("Checking model files..."))

        let task = Task(priority: .userInitiated) {
            do {
                try await transcriptionService.loadModel(repoID: repoID) { update in
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == repoID else { return }

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
                    guard appState.selectedModelID == repoID else { return }

                    appState.modelStatus = .loaded
                    appState.downloadedModelRepoIDs.insert(repoID)
                    _ = appState.transition(to: .idle)
                    modelLoadTask = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    modelLoadTask = nil

                    // Today, cancellation usually means a newer load has already started.
                    // If this task is cancelled without a replacement load, clear loading UI.
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
                    guard appState.selectedModelID == repoID else { return }

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

        // Accessibility (prompts user if not trusted)
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

    // MARK: - Sound Effects

    /// Get system output volume (0.0 to 1.0)
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

    /// Calculate sound effect volume (inverse of system volume, capped at 20% of system)
    private func calculateSoundVolume() -> Float {
        let systemVolume = getSystemOutputVolume()
        // At 20% system volume -> 100% sound volume
        // At 100% system volume -> 20% sound volume
        // Linear interpolation: soundVol = 1.0 - 0.8 * (systemVol - 0.2) / 0.8
        // Simplified: soundVol = 1.0 - (systemVol - 0.2) = 1.2 - systemVol (clamped)
        // Actually: we want sound to be at most 20% of system's max perceived loudness
        // So: soundVol = 0.2 / max(systemVol, 0.2)
        let effectiveSystemVol = max(systemVolume, 0.2)
        return min(0.2 / effectiveSystemVol, 1.0)
    }

    private func playStartSound() {
        guard let sound = NSSound(named: "Tink") else { return }
        sound.volume = calculateSoundVolume()
        sound.play()
    }

    private func playStopSound() {
        guard let sound = NSSound(named: "Pop") else { return }
        sound.volume = calculateSoundVolume()
        sound.play()
    }
}
