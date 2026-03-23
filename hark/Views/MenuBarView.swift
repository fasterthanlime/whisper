import AppKit
import Carbon.HIToolbox.Events
import SwiftUI

/// The dropdown content shown when clicking the menu bar icon.
struct MenuBarView: View {
    @Bindable var appState: AppState
    var onModelSelect: (STTModelDefinition) -> Void
    var onDeleteLocalModel: (STTModelDefinition) -> Void
    var onHotkeyBindingSave: (HotkeyBinding) -> Void
    var onHotkeyEditorPresentedChange: (Bool) -> Void
    var runOnStartupEnabled: Bool
    var onRunOnStartupToggle: () -> Void
    var onRequestMicrophonePermission: () -> Void
    var onRequestAccessibilityPermission: () -> Void
    var onRecheckPermissions: () -> Void
    var onQuit: () -> Void

    @State private var showSettings = false
    @State private var hoveredModelID: String?
    @State private var hoveredDeleteModelID: String?
    @State private var hoveredDownloadModelID: String?
    @State private var isHoveringRunOnStartup = false
    @State private var isHoveringPauseMedia = false
    @State private var pauseMediaEnabled = MediaController.isEnabled
    @State private var isHoveringQuit = false
    @State private var isHoveringSettings = false
    @State private var vocabPromptText: String = ""
    @State private var isCapturingHotkey = false
    @State private var capturePressedKeyCodes: Set<UInt16> = []
    @State private var latestCapturedKeyCodes: Set<UInt16> = []
    @State private var hotkeyCaptureLocalMonitor: Any?
    @State private var hotkeyCaptureGlobalMonitor: Any?
    private let infoLabelWidth: CGFloat = 94

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            statusSection
            Divider().padding(.horizontal, 2)

            languageSection

            if case .downloading(let progress) = appState.modelStatus {
                ProgressView(value: progress)
                    .padding(.horizontal, 2)
            }

            if !appState.transcriptionHistory.isEmpty {
                Divider().padding(.horizontal, 2)
                historySection
            }

            Divider().padding(.horizontal, 2)
            settingsDisclosure

            if showSettings {
                settingsContent
            }

            quitSection
        }
        .padding(10)
        .frame(width: 300)
        .onAppear {
            // Load per-app vocab prompt for the current frontmost app
            if let bundleID = NSWorkspace.shared.frontmostApplication?.bundleIdentifier {
                vocabPromptText = appState.appVocabPrompts[bundleID] ?? ""
            }
        }
        .onDisappear {
            stopHotkeyCapture(notifyState: true)
        }
    }

    // MARK: - Status

    @ViewBuilder
    private var statusSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                Circle()
                    .fill(appState.menuStatusColor)
                    .frame(width: 9, height: 9)
                Text(appState.menuStatusLabel)
                    .font(.system(.body, weight: .semibold))
            }

            if appState.shouldShowStatusDetail {
                Text(appState.statusText)
                    .font(.system(.caption))
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }
        }
        .padding(.horizontal, 6)
    }

    // MARK: - Language & Vocab (per-app, always visible)

    @ViewBuilder
    private var languageSection: some View {
        let frontmostApp = NSWorkspace.shared.frontmostApplication
        let appName = frontmostApp?.localizedName ?? "Current App"

        VStack(alignment: .leading, spacing: 4) {
            Text("Language — \(appName)")
                .font(.system(.caption))
                .foregroundStyle(.secondary)

            HStack(spacing: 4) {
                languageButton(label: "Auto", language: nil)
                ForEach(AppState.supportedLanguages, id: \.name) { lang in
                    languageButton(label: lang.label, language: lang.name)
                }
            }
            .padding(.horizontal, 2)

            Text("Vocab Prompt — \(appName)")
                .font(.system(.caption))
                .foregroundStyle(.secondary)
                .padding(.top, 4)

            TextField("e.g. Working with serde, candle, GGUF...", text: $vocabPromptText, axis: .vertical)
                .font(.system(.caption2))
                .textFieldStyle(.plain)
                .padding(6)
                .lineLimit(2...4)
                .background {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.primary.opacity(0.05))
                }
                .onChange(of: vocabPromptText) { _, newValue in
                    appState.setVocabPromptForFrontmostApp(newValue)
                }
        }
    }

    @ViewBuilder
    private func languageButton(label: String, language: String?) -> some View {
        let isSelected = appState.currentLanguage == language
        Button {
            appState.setLanguageForFrontmostApp(language)
        } label: {
            Text(label)
                .font(.system(.caption2, weight: isSelected ? .bold : .medium))
                .foregroundStyle(isSelected ? .white : .primary)
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background {
                    RoundedRectangle(cornerRadius: 5, style: .continuous)
                        .fill(isSelected ? Color.accentColor : Color.primary.opacity(0.08))
                }
        }
        .buttonStyle(.plain)
    }

    // MARK: - Recent transcriptions

    @ViewBuilder
    private var historySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Recent")
                .font(.system(.caption, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)

            ForEach(appState.transcriptionHistory.prefix(10)) { item in
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(item.text, forType: .string)
                } label: {
                    Text(item.displayText)
                        .font(.system(.caption))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background {
                            RoundedRectangle(cornerRadius: 4, style: .continuous)
                                .fill(Color.primary.opacity(0.05))
                        }
                        .contentShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                }
                .buttonStyle(.plain)
                .help(item.text)
            }
        }
    }

    // MARK: - Collapsible settings

    @ViewBuilder
    private var settingsDisclosure: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                showSettings.toggle()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "gearshape")
                    .font(.system(.caption))
                Text("Settings")
                    .font(.system(.body))
                Spacer()
                Image(systemName: showSettings ? "chevron.up" : "chevron.down")
                    .font(.system(.caption2, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isHoveringSettings ? Color.primary.opacity(0.1) : .clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
        .onHover { isHoveringSettings = $0 }
    }

    @ViewBuilder
    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Model picker
            modelSection

            Divider().padding(.horizontal, 2)

            // Hotkey
            infoSection

            Divider().padding(.horizontal, 2)

            // Toggles
            startupSection
        }
        .padding(.leading, 8)
    }

    // MARK: - Model (inside settings)

    @ViewBuilder
    private var modelSection: some View {
        Text("Model")
            .font(.system(.caption))
            .foregroundStyle(.secondary)

        ForEach(STTModelDefinition.allModels) { model in
            let isDownloaded = appState.downloadedModelIDs.contains(model.id)
            let isSelectedDownloadedModel =
                appState.selectedModelID == model.id && isDownloaded
            let isHoveringDelete = hoveredDeleteModelID == model.id
            let isHoveringDownload = hoveredDownloadModelID == model.id

            HStack(spacing: 6) {
                Button {
                    onModelSelect(model)
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(.green)
                            .font(.system(.caption, weight: .semibold))
                            .opacity(isSelectedDownloadedModel ? 1 : 0)
                        Text(model.displayName)
                            .font(.system(.caption))

                        Spacer()
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background {
                        RoundedRectangle(cornerRadius: 6, style: .continuous)
                            .fill(hoveredModelID == model.id ? Color.primary.opacity(0.1) : .clear)
                    }
                    .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
                }
                .buttonStyle(.plain)
                .disabled(isModelInteractionDisabled)
                .onHover { isHovering in
                    if isHovering {
                        hoveredModelID = model.id
                    } else if hoveredModelID == model.id {
                        hoveredModelID = nil
                    }
                }

                if isDownloaded {
                    Button {
                        onDeleteLocalModel(model)
                    } label: {
                        Image(systemName: "trash")
                            .font(.system(.caption2, weight: .semibold))
                            .foregroundStyle(isHoveringDelete ? .red : .secondary)
                            .frame(width: 16, height: 16)
                            .background {
                                RoundedRectangle(cornerRadius: 4, style: .continuous)
                                    .fill(isHoveringDelete ? Color.red.opacity(0.14) : .clear)
                            }
                    }
                    .buttonStyle(.plain)
                    .disabled(isModelInteractionDisabled)
                    .help("Delete local model files")
                    .onHover { isHovering in
                        if isHovering {
                            hoveredDeleteModelID = model.id
                        } else if hoveredDeleteModelID == model.id {
                            hoveredDeleteModelID = nil
                        }
                    }
                } else {
                    Button {
                        onModelSelect(model)
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.system(.caption2, weight: .semibold))
                            .foregroundStyle(isHoveringDownload ? .blue : .secondary)
                            .frame(width: 16, height: 16)
                            .background {
                                RoundedRectangle(cornerRadius: 4, style: .continuous)
                                    .fill(isHoveringDownload ? Color.blue.opacity(0.16) : .clear)
                            }
                    }
                    .buttonStyle(.plain)
                    .disabled(isModelInteractionDisabled)
                    .help("Download model")
                    .onHover { isHovering in
                        if isHovering {
                            hoveredDownloadModelID = model.id
                        } else if hoveredDownloadModelID == model.id {
                            hoveredDownloadModelID = nil
                        }
                    }
                }
            }
        }
    }

    // MARK: - Info (hotkey, permissions, input — inside settings)

    @ViewBuilder
    private var infoSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            infoRow(label: "Push to Talk") {
                Button {
                    toggleHotkeyCapture()
                } label: {
                    Text(captureButtonLabel)
                        .font(.system(.caption, weight: .semibold))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .foregroundStyle(isCapturingHotkey ? .red : .primary)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
                .disabled(appState.phase != .idle)
                .help("Click and press key combination")
            }

            if let hotkeySettingsMessage = appState.hotkeySettingsMessage {
                Text(hotkeySettingsMessage)
                    .font(.system(.caption2, weight: .medium))
                    .foregroundStyle(.orange)
                    .padding(.horizontal, 6)
            }

            permissionRow("Mic", status: appState.microphonePermission)
            permissionRow("Accessibility", status: appState.accessibilityPermission)

            if let inputDevice = appState.activeInputDeviceName {
                HStack(spacing: 8) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.blue)
                    Text("Input")
                        .font(.system(.caption))
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text(inputDevice)
                        .font(.system(.caption2, weight: .medium))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                .padding(.horizontal, 6)
            }

            if !appState.hasRequiredPermissions {
                permissionActionsSection
            }
        }
    }

    @ViewBuilder
    private var permissionActionsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            if appState.microphonePermission != .granted {
                Button(appState.microphonePermission == .denied ? "Open Microphone Settings" : "Request Microphone Access") {
                    onRequestMicrophonePermission()
                }
                .font(.system(.caption, weight: .medium))
            }

            if appState.accessibilityPermission != .granted {
                Button("Open Accessibility Settings") {
                    onRequestAccessibilityPermission()
                }
                .font(.system(.caption, weight: .medium))
            }

            Button("Re-check Permissions") {
                onRecheckPermissions()
            }
            .font(.system(.caption, weight: .medium))
        }
        .padding(.horizontal, 6)
        .padding(.top, 2)
    }

    // MARK: - Toggles (inside settings)

    @ViewBuilder
    private var startupSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            toggleRow(
                label: "Run on Startup",
                isOn: runOnStartupEnabled,
                isHovering: $isHoveringRunOnStartup,
                action: onRunOnStartupToggle
            )

            if let error = appState.runOnStartupError {
                Text(error)
                    .font(.system(.caption2, weight: .medium))
                    .foregroundStyle(.red)
                    .padding(.horizontal, 6)
            }

            toggleRow(
                label: "Pause Media While Dictating",
                isOn: pauseMediaEnabled,
                isHovering: $isHoveringPauseMedia,
                action: { pauseMediaEnabled.toggle(); MediaController.isEnabled = pauseMediaEnabled }
            )
        }
    }

    @ViewBuilder
    private func toggleRow(label: String, isOn: Bool, isHovering: Binding<Bool>, action: @escaping () -> Void) -> some View {
        Button {
            action()
        } label: {
            HStack(spacing: 6) {
                if isOn {
                    Image(systemName: "checkmark.circle.fill")
                        .font(.system(.caption, weight: .semibold))
                        .foregroundStyle(.green)
                }
                Text(label)
                    .font(.system(.caption))
                Spacer()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isHovering.wrappedValue ? Color.primary.opacity(0.1) : .clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
        .onHover { h in
            isHovering.wrappedValue = h
        }
    }

    // MARK: - Quit

    @ViewBuilder
    private var quitSection: some View {
        Button {
            onQuit()
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "power")
                    .font(.system(.caption))
                Text("Quit Hark")
                    .font(.system(.body))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .background {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isHoveringQuit ? Color.primary.opacity(0.1) : .clear)
            }
            .contentShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
        }
        .buttonStyle(.plain)
        .keyboardShortcut("q")
        .onHover { isHoveringQuit = $0 }
    }

    // MARK: - Helpers

    @ViewBuilder
    private func infoRow<Content: View>(label: String, @ViewBuilder trailing: () -> Content) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(.caption))
                .foregroundStyle(.secondary)
                .frame(width: infoLabelWidth, alignment: .leading)
            Spacer(minLength: 0)
            trailing()
        }
    }

    private func permissionRow(_ name: String, status: PermissionStatus) -> some View {
        let granted = status == .granted

        return HStack(spacing: 8) {
            Image(systemName: granted ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.system(size: 10, weight: .semibold))
                .foregroundStyle(granted ? .green : .red)
            Text(name)
                .font(.system(.caption))
                .foregroundStyle(.secondary)
            Spacer()
            Text(permissionLabel(for: status))
                .font(.system(.caption2, weight: .semibold))
                .foregroundStyle(granted ? .green : .orange)
        }
        .padding(.horizontal, 6)
    }

    private func permissionLabel(for status: PermissionStatus) -> String {
        switch status {
        case .granted:
            return "Granted"
        case .unknown:
            return "Unknown"
        case .denied:
            return "Missing"
        }
    }

    private var isModelInteractionDisabled: Bool {
        switch appState.phase {
        case .recording, .transcribing, .pasting:
            return true
        default:
            return false
        }
    }

    /// Label shown on the hotkey capture button, with live feedback of currently pressed keys.
    private var captureButtonLabel: String {
        guard isCapturingHotkey else {
            return appState.hotkeyBinding.displayLabel
        }
        if latestCapturedKeyCodes.isEmpty {
            return "Press keys... (Esc to cancel)"
        }
        return HotkeyBinding(keyCodes: latestCapturedKeyCodes).displayLabel + "..."
    }

    private func toggleHotkeyCapture() {
        if isCapturingHotkey {
            stopHotkeyCapture(notifyState: true)
        } else {
            startHotkeyCapture()
        }
    }

    private func startHotkeyCapture() {
        guard !isCapturingHotkey else { return }
        isCapturingHotkey = true
        capturePressedKeyCodes.removeAll()
        latestCapturedKeyCodes.removeAll()
        installHotkeyCaptureMonitorIfNeeded()
        onHotkeyEditorPresentedChange(true)
    }

    private func stopHotkeyCapture(notifyState: Bool) {
        guard isCapturingHotkey || hotkeyCaptureLocalMonitor != nil || hotkeyCaptureGlobalMonitor != nil else { return }
        isCapturingHotkey = false
        capturePressedKeyCodes.removeAll()
        latestCapturedKeyCodes.removeAll()
        removeHotkeyCaptureMonitor()

        if notifyState {
            onHotkeyEditorPresentedChange(false)
        }
    }

    private func installHotkeyCaptureMonitorIfNeeded() {
        guard hotkeyCaptureLocalMonitor == nil else { return }

        hotkeyCaptureLocalMonitor = NSEvent.addLocalMonitorForEvents(
            matching: [.flagsChanged, .keyDown, .keyUp]
        ) { event in
            handleHotkeyCaptureEvent(event)
            return nil
        }

        hotkeyCaptureGlobalMonitor = NSEvent.addGlobalMonitorForEvents(
            matching: [.flagsChanged, .keyDown, .keyUp]
        ) { event in
            handleHotkeyCaptureEvent(event)
        }
    }

    private func removeHotkeyCaptureMonitor() {
        if let local = hotkeyCaptureLocalMonitor {
            NSEvent.removeMonitor(local)
            hotkeyCaptureLocalMonitor = nil
        }
        if let global = hotkeyCaptureGlobalMonitor {
            NSEvent.removeMonitor(global)
            hotkeyCaptureGlobalMonitor = nil
        }
    }

    private func handleHotkeyCaptureEvent(_ event: NSEvent) {
        guard isCapturingHotkey else { return }

        switch event.type {
        case .keyDown:
            guard !event.isARepeat else { return }
            if event.keyCode == UInt16(kVK_Escape) {
                stopHotkeyCapture(notifyState: true)
                return
            }
            capturePressedKeyCodes.insert(event.keyCode)
            latestCapturedKeyCodes = capturePressedKeyCodes

        case .keyUp:
            capturePressedKeyCodes.remove(event.keyCode)
            maybeCommitCapturedHotkeyIfComplete()

        case .flagsChanged:
            let keyCode = event.keyCode
            guard HotkeyKeyCode.modifierCodes.contains(keyCode) else { return }

            let isPressed = HotkeyBinding.isModifierPressed(
                keyCode: keyCode,
                flagsRaw: UInt64(event.modifierFlags.rawValue)
            )

            if isPressed {
                capturePressedKeyCodes.insert(keyCode)
                latestCapturedKeyCodes = capturePressedKeyCodes
            } else {
                capturePressedKeyCodes.remove(keyCode)
                maybeCommitCapturedHotkeyIfComplete()
            }

        default:
            return
        }
    }

    private func maybeCommitCapturedHotkeyIfComplete() {
        guard capturePressedKeyCodes.isEmpty else { return }
        guard !latestCapturedKeyCodes.isEmpty else { return }

        let binding = HotkeyBinding(keyCodes: latestCapturedKeyCodes)
        onHotkeyBindingSave(binding)
        stopHotkeyCapture(notifyState: true)
    }
}
