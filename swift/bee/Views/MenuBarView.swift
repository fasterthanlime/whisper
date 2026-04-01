import AppKit
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState

    @State private var showSettings = false
    @State private var pauseMediaEnabled = false // TODO: wire to MediaController
    @State private var runOnStartupEnabled = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            statusSection

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

            Divider().padding(.horizontal, 2)
            quitButton
        }
        .padding(10)
        .frame(width: 280)
    }

    // MARK: - Status

    @ViewBuilder
    private var statusSection: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(statusColor)
                .frame(width: 9, height: 9)
            Text(statusLabel)
                .font(.system(.body, weight: .semibold))
        }
        .padding(.horizontal, 6)
    }

    private var statusLabel: String {
        switch appState.uiState {
        case .idle:
            switch appState.modelStatus {
            case .loaded: "Bee Ready"
            case .loading: "Loading Model..."
            case .downloading: "Downloading Model..."
            case .notLoaded: "No Model"
            case .error(let msg): "Error: \(msg)"
            }
        case .pending: "Starting..."
        case .pushToTalk: "Recording"
        case .locked: "Recording (Locked)"
        case .lockedOptionHeld: "Recording (Locked)"
        }
    }

    private var statusColor: Color {
        switch appState.uiState {
        case .idle:
            switch appState.modelStatus {
            case .loaded: .green
            case .error: .red
            default: .orange
            }
        case .pending, .pushToTalk, .locked, .lockedOptionHeld:
            .red
        }
    }

    // MARK: - History

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

    // MARK: - Settings disclosure

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
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Settings content

    @ViewBuilder
    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            inputDeviceSection
            Divider().padding(.horizontal, 2)
            togglesSection
        }
        .padding(.leading, 8)
    }

    @ViewBuilder
    private var inputDeviceSection: some View {
        let deviceName = appState.activeInputDeviceName ?? "No Input"

        HStack(spacing: 8) {
            Text("Input")
                .font(.system(.caption))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .leading)
            Spacer(minLength: 0)
            Menu {
                if appState.availableInputDevices.isEmpty {
                    Text("No Input Devices")
                } else {
                    ForEach(appState.availableInputDevices, id: \.uid) { device in
                        Button {
                            appState.selectInputDevice(uid: device.uid)
                        } label: {
                            if device.uid == appState.activeInputDeviceUID {
                                Label(device.name, systemImage: "checkmark")
                            } else {
                                Text(device.name)
                            }
                        }
                    }
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "mic.fill")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundStyle(.blue)
                    Text(deviceName)
                        .font(.system(.caption2, weight: .medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(.caption2, weight: .semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 4)
                .background {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(Color.primary.opacity(0.05))
                }
            }
            .menuStyle(.borderlessButton)
            .fixedSize(horizontal: false, vertical: true)
        }

        toggleRow(label: "Keep Warm", isOn: appState.activeInputDeviceKeepWarm) {
            appState.toggleActiveInputDeviceKeepWarm()
        }
    }

    @ViewBuilder
    private var togglesSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            chunkSizePicker
            tokenLimitPicker(label: "Tokens/step", options: Self.streamingTokenOptions, current: appState.maxNewTokensStreaming) {
                appState.maxNewTokensStreaming = $0
            }
            tokenLimitPicker(label: "Tokens/final", options: Self.finalTokenOptions, current: appState.maxNewTokensFinal) {
                appState.maxNewTokensFinal = $0
            }

            toggleRow(label: "Run on Startup", isOn: runOnStartupEnabled) {
                runOnStartupEnabled.toggle()
                // TODO: SMAppService register/unregister
            }

            toggleRow(label: "Pause Media While Dictating", isOn: pauseMediaEnabled) {
                pauseMediaEnabled.toggle()
                // TODO: persist to UserDefaults
            }

            toggleRow(label: "Debug Overlay", isOn: appState.debugEnabled) {
                appState.debugEnabled.toggle()
                DebugPanel.shared.toggle(appState: appState)
            }
        }
    }

    private static let chunkSizeOptions: [(label: String, value: Float)] = [
        ("0.2s", 0.2),
        ("0.35s", 0.35),
        ("0.5s", 0.5),
        ("0.75s", 0.75),
        ("1s", 1.0),
    ]

    private static let streamingTokenOptions: [(label: String, value: UInt32)] = [
        ("default", 0), ("8", 8), ("16", 16), ("32", 32), ("64", 64),
    ]

    private static let finalTokenOptions: [(label: String, value: UInt32)] = [
        ("default", 0), ("64", 64), ("128", 128), ("256", 256), ("512", 512),
    ]

    @ViewBuilder
    private func tokenLimitPicker(label: String, options: [(label: String, value: UInt32)], current: UInt32, action: @escaping (UInt32) -> Void) -> some View {
        HStack(spacing: 8) {
            Text(label)
                .font(.system(.caption))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .leading)
            Spacer(minLength: 0)
            HStack(spacing: 2) {
                ForEach(options, id: \.value) { opt in
                    Button {
                        action(opt.value)
                    } label: {
                        Text(opt.label)
                            .font(.system(size: 10, weight: current == opt.value ? .bold : .regular))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background {
                                if current == opt.value {
                                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                                        .fill(Color.accentColor.opacity(0.2))
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(.horizontal, 6)
    }

    @ViewBuilder
    private var chunkSizePicker: some View {
        HStack(spacing: 8) {
            Text("Chunk")
                .font(.system(.caption))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .leading)
            Spacer(minLength: 0)
            HStack(spacing: 2) {
                ForEach(Self.chunkSizeOptions, id: \.value) { opt in
                    Button {
                        appState.chunkSizeSec = opt.value
                    } label: {
                        Text(opt.label)
                            .font(.system(size: 10, weight: appState.chunkSizeSec == opt.value ? .bold : .regular))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 3)
                            .background {
                                if appState.chunkSizeSec == opt.value {
                                    RoundedRectangle(cornerRadius: 4, style: .continuous)
                                        .fill(Color.accentColor.opacity(0.2))
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .padding(.horizontal, 6)
    }

    @ViewBuilder
    private func toggleRow(label: String, isOn: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
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
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    // MARK: - Quit

    @ViewBuilder
    private var quitButton: some View {
        Button {
            BeeInputClient.restoreInputSourceIfNeeded()
            NSApplication.shared.terminate(nil)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "power")
                    .font(.system(.caption))
                Text("Quit Bee")
                    .font(.system(.body))
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 6)
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .keyboardShortcut("q")
    }
}
