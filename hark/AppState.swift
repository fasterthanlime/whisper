import SwiftUI
import Observation

/// The phases of the app's push-to-talk state machine.
enum AppPhase: Equatable {
    case idle
    case loading(String) // loading model, message
    case recording
    case transcribing
    case pasting
    case error(String)
}

private enum AppPhaseKind: Equatable {
    case idle
    case loading
    case recording
    case transcribing
    case pasting
    case error
}

private extension AppPhase {
    var kind: AppPhaseKind {
        switch self {
        case .idle:
            return .idle
        case .loading:
            return .loading
        case .recording:
            return .recording
        case .transcribing:
            return .transcribing
        case .pasting:
            return .pasting
        case .error:
            return .error
        }
    }
}

/// Central observable state for the entire app.
@Observable
@MainActor
final class AppState {
    static let inputDeviceWarmPreferencesDefaultsKey = "inputDeviceKeepWarmByUID"
    static let appAutoSubmitDefaultsKey = "appAutoSubmit"

    var phase: AppPhase = .idle
    var selectedModelID: String = STTModelDefinition.default.id
    var hotkeyBinding: HotkeyBinding = .defaultBinding
    var hotkeySettingsMessage: String?
    var isEditingHotkey = false
    var modelStatus: ModelStatus = .notLoaded
    var downloadedModelIDs: Set<String> = []
    var microphonePermission: PermissionStatus = .unknown
    var accessibilityPermission: PermissionStatus = .unknown
    var runOnStartupEnabled = false
    var runOnStartupError: String?
    /// Name of the currently active input device.
    var activeInputDeviceName: String?
    /// UID of the currently active/default input device.
    var activeInputDeviceUID: String?
    /// All visible input devices from CoreAudio (default device first).
    var availableInputDevices: [InputAudioDevice] = []
    /// Per-device "keep warm" preference keyed by device UID.
    var inputDeviceKeepWarmByUID: [String: Bool] = [:] {
        didSet {
            UserDefaults.standard.set(
                inputDeviceKeepWarmByUID,
                forKey: Self.inputDeviceWarmPreferencesDefaultsKey
            )
        }
    }

    var activeInputDevice: InputAudioDevice? {
        if let activeInputDeviceUID,
           let match = availableInputDevices.first(where: { $0.uid == activeInputDeviceUID }) {
            return match
        }
        return availableInputDevices.first(where: \.isDefault)
    }

    var activeInputDeviceKeepWarm: Bool {
        guard let activeInputDevice else { return true }
        if let persisted = inputDeviceKeepWarmByUID[activeInputDevice.uid] {
            return persisted
        }
        // Default policy: built-in laptop microphones stay cold; external mics stay warm.
        return !activeInputDevice.isBuiltIn
    }

    func applyInputDeviceSnapshot(_ snapshot: InputDeviceSnapshot) {
        availableInputDevices = snapshot.devices
        activeInputDeviceUID = snapshot.activeDevice?.uid
        activeInputDeviceName = snapshot.activeDevice?.name
    }

    func setKeepWarmForActiveInputDevice(_ keepWarm: Bool) {
        guard let activeUID = activeInputDevice?.uid else { return }
        inputDeviceKeepWarmByUID[activeUID] = keepWarm
    }

    func keepWarmPreference(for deviceUID: String) -> Bool {
        if let persisted = inputDeviceKeepWarmByUID[deviceUID] {
            return persisted
        }
        if let device = availableInputDevices.first(where: { $0.uid == deviceUID }) {
            return !device.isBuiltIn
        }
        return true
    }

    // MARK: - Per-app language

    /// Language forced per bundle ID. Missing key = auto-detect.
    /// Values are Qwen3 language names: "english", "french", etc.
    var appLanguages: [String: String] = [:] {
        didSet {
            UserDefaults.standard.set(appLanguages, forKey: "appLanguages")
        }
    }

    /// The language that will be used for the current frontmost app.
    /// `nil` means auto-detect.
    var currentLanguage: String? {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else {
            return nil
        }
        return appLanguages[bundleID]
    }

    /// Short display label for the current language ("EN", "FR", "Auto").
    var currentLanguageLabel: String {
        guard let lang = currentLanguage else { return "Auto" }
        return Self.languageShortLabel(lang)
    }

    /// Set the language for the frontmost app's bundle ID.
    func setLanguageForFrontmostApp(_ language: String?) {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return }
        if let language {
            appLanguages[bundleID] = language
        } else {
            appLanguages.removeValue(forKey: bundleID)
        }
    }

    /// Supported languages with their Qwen3 names and short labels.
    static let supportedLanguages: [(name: String, label: String)] = [
        ("english", "EN"),
        ("french", "FR"),
        ("spanish", "ES"),
        ("german", "DE"),
        ("polish", "PL"),
    ]

    static func languageShortLabel(_ name: String) -> String {
        supportedLanguages.first { $0.name == name }?.label ?? name.prefix(2).uppercased()
    }

    // MARK: - Per-app auto-submit

    /// Apps where dictated text should be followed by Enter by default.
    static let defaultAutoSubmitBundleIDs: Set<String> = [
        "com.googlecode.iterm2",
        "com.mitchellh.ghostty",
    ]

    /// Explicit auto-submit preference per app. Missing key = default policy.
    var appAutoSubmit: [String: Bool] = [:] {
        didSet {
            UserDefaults.standard.set(appAutoSubmit, forKey: Self.appAutoSubmitDefaultsKey)
        }
    }

    /// Whether the current frontmost app should auto-submit after paste.
    var currentAutoSubmit: Bool {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else {
            return false
        }
        if let configured = appAutoSubmit[bundleID] {
            return configured
        }
        return Self.defaultAutoSubmitBundleIDs.contains(bundleID)
    }

    /// Set auto-submit behavior for the current frontmost app.
    func setAutoSubmitForFrontmostApp(_ enabled: Bool) {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return }
        appAutoSubmit[bundleID] = enabled
    }

    // MARK: - Vocabulary prompt (per-app)

    /// Vocab prompts per bundle ID. Missing key = no prompt.
    var appVocabPrompts: [String: String] = [:] {
        didSet {
            UserDefaults.standard.set(appVocabPrompts, forKey: "appVocabPrompts")
        }
    }

    /// The vocab prompt for the current frontmost app, or nil.
    var currentVocabPrompt: String? {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else {
            return nil
        }
        let text = appVocabPrompts[bundleID] ?? ""
        return text.isEmpty ? nil : text
    }

    /// Set the vocab prompt for the frontmost app's bundle ID.
    func setVocabPromptForFrontmostApp(_ prompt: String) {
        guard let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier else { return }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            appVocabPrompts.removeValue(forKey: bundleID)
        } else {
            appVocabPrompts[bundleID] = trimmed
        }
    }

    /// Compat shim — returns currentVocabPrompt for session creation.
    var vocabPrompt: String? { currentVocabPrompt }

    /// Real-time microphone audio level (0–1), updated from the audio tap.
    var audioLevel: Float = 0

    /// Real-time spectrum bands (0–1 each), updated from FFT.
    var spectrumBands: [Float] = Array(repeating: 0, count: 6)

    /// Partial transcript during streaming transcription.
    var partialTranscript: String = ""
    /// UTF-16 length of the committed prefix inside `partialTranscript`.
    var partialTranscriptCommittedUTF16: Int = 0

    /// Whether recording is in toggle/locked mode (hands-free).
    var isLockedMode = false

    /// Whether final inference is running (spinner in overlay).
    var isFinishing = false

    /// Overlay dismiss animation state (driven by OverlayManager, observed by the view).
    var overlayDismiss: OverlayResult = .none
    /// When true, render the overlay footer above the text body; otherwise below.
    var overlayFooterAbove = false
    /// When true, overlay is pinned to a concrete input frame and includes an in-bounds footer.
    var overlayControlFrameMode = false
    /// Bundle ID of the app the current recording is tethered to.
    var overlayLockedBundleID: String?
    /// Human-readable app name for tether lock UI.
    var overlayLockedAppName: String?
    /// True when focus moved to a different app while tether lock is active.
    var overlayTetherOutOfApp = false

    /// Recent transcription history (newest first), max 20 items.
    var transcriptionHistory: [TranscriptionHistoryItem] = []

    /// Add a transcription to history, keeping max 20 items.
    func addToHistory(_ text: String) {
        let item = TranscriptionHistoryItem(text: text, timestamp: Date())
        transcriptionHistory.insert(item, at: 0)
        if transcriptionHistory.count > 20 {
            transcriptionHistory.removeLast()
        }
    }

    /// Brief status text shown in the menu bar dropdown.
    var statusText: String {
        switch phase {
        case .loading(let msg):
            return msg
        case .recording:
            return "Recording..."
        case .transcribing:
            return "Transcribing..."
        case .pasting:
            return "Pasting..."
        case .error(let msg):
            return "Error: \(msg)"
        case .idle:
            switch modelStatus {
            case .loaded where hasRequiredPermissions:
                return "Ready"
            case .loaded:
                return "Missing \(missingPermissionSummary)."
            case .downloading(let progress):
                let percent = Int((min(max(progress, 0), 1) * 100).rounded())
                return "Downloading model (\(percent)%)."
            case .loading:
                return "Initializing model..."
            case .error(let message):
                return "Model error: \(message)"
            case .notLoaded:
                return downloadedModelIDs.isEmpty ? "No local models available." : "Model not loaded."
            }
        }
    }

    var menuStatusLabel: String {
        switch phase {
        case .recording:
            return "Hark Recording"
        case .transcribing:
            return "Hark Transcribing"
        case .pasting:
            return "Hark Pasting"
        case .loading:
            switch modelStatus {
            case .downloading:
                return "Hark Downloading"
            case .loading:
                return "Hark Initializing"
            default:
                return "Hark Loading"
            }
        case .error:
            return "Hark Error"
        case .idle:
            switch modelStatus {
            case .loaded where hasRequiredPermissions:
                return "Hark Ready"
            case .loaded:
                return "Hark Needs Permission"
            case .error:
                return "Model Error"
            case .notLoaded where downloadedModelIDs.isEmpty:
                return "No Local Models"
            default:
                return "Hark Loading"
            }
        }
    }

    var menuStatusColor: Color {
        switch phase {
        case .recording:
            return .red
        case .transcribing, .pasting:
            return .blue
        case .error:
            return .red
        case .idle:
            switch modelStatus {
            case .loaded where hasRequiredPermissions:
                return .green
            case .error:
                return .red
            default:
                return .orange
            }
        case .loading:
            return .orange
        }
    }

    var shouldShowStatusDetail: Bool {
        if case .idle = phase, modelStatus == .loaded, hasRequiredPermissions {
            return false
        }
        return true
    }

    /// SF Symbol name for the menu bar icon.
    var menuBarIcon: String {
        switch phase {
        case .idle:
            return "waveform.circle"
        case .loading:
            switch modelStatus {
            case .downloading:
                return "arrow.down.circle"
            case .loading:
                return "arrow.right.circle"
            default:
                return "arrow.right.circle"
            }
        case .recording, .transcribing, .pasting:
            return "waveform.circle.fill"
        case .error:
            return "exclamationmark.triangle"
        }
    }

    var hasMicrophonePermission: Bool { microphonePermission == .granted }
    var hasAccessibilityPermission: Bool { accessibilityPermission == .granted }
    var hasRequiredPermissions: Bool { hasMicrophonePermission && hasAccessibilityPermission }

    var missingPermissionSummary: String {
        var missing: [String] = []
        if !hasMicrophonePermission {
            missing.append("Microphone")
        }
        if !hasAccessibilityPermission {
            missing.append("Accessibility")
        }

        switch missing.count {
        case 2:
            return "Microphone and Accessibility permissions"
        case 1:
            return "\(missing[0]) permission"
        default:
            return "required permissions"
        }
    }

    @discardableResult
    func transition(to newPhase: AppPhase) -> Bool {
        let currentKind = phase.kind
        let nextKind = newPhase.kind

        guard canTransition(from: currentKind, to: nextKind) else {
            assertionFailure("Invalid phase transition from \(phase) to \(newPhase)")
            return false
        }

        phase = newPhase
        return true
    }

    private func canTransition(from current: AppPhaseKind, to next: AppPhaseKind) -> Bool {
        if current == next {
            return true
        }

        if next == .error {
            return true
        }

        switch current {
        case .idle:
            return next == .loading || next == .recording
        case .loading:
            return next == .idle || next == .loading
        case .recording:
            return next == .transcribing || next == .idle
        case .transcribing:
            return next == .pasting || next == .idle
        case .pasting:
            return next == .idle
        case .error:
            return next == .idle || next == .loading
        }
    }
}

enum ModelStatus: Equatable {
    case notLoaded
    case downloading(progress: Double)
    case loading
    case loaded
    case error(String)
}

enum PermissionStatus: Equatable {
    case unknown
    case granted
    case denied
}

struct TranscriptionHistoryItem: Identifiable {
    let id = UUID()
    let text: String
    let timestamp: Date

    /// Truncated display text for menu (max 50 chars).
    var displayText: String {
        if text.count <= 50 {
            return text
        }
        return "..." + String(text.suffix(47))
    }
}
