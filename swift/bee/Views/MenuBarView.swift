import AppKit
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState
    @Environment(\.openSettings) private var openSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Circle()
                    .fill(stateColor)
                    .frame(width: 8, height: 8)
                Text(currentStateLabel)
                    .font(.body.weight(.semibold))
            }
            .padding(.horizontal, 6)

            VStack(alignment: .leading, spacing: 4) {
                menuInfoRow(icon: "cpu", text: modelInfoLabel)
                menuInfoRow(icon: "mic.fill", text: appState.activeInputDeviceName ?? "No input")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 6)

            if let last = appState.transcriptionHistory.first {
                Divider().padding(.horizontal, 2)
                Text(last.text)
                    .font(.caption)
                    .lineLimit(2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
            }

            Divider().padding(.horizontal, 2)

            Button {
                dismiss()
                NSApp.activate(ignoringOtherApps: true)
                openSettings()
                DispatchQueue.main.async {
                    let normalWindow = NSApp.orderedWindows.first { window in
                        !(window is NSPanel)
                    }
                    normalWindow?.makeKeyAndOrderFront(nil)
                }
            }
            label: {
                MenuActionRow(title: "Open settings", shortcut: "⌘,")
            }
            .buttonStyle(.plain)
            .keyboardShortcut(",", modifiers: [.command])

            Button {
                dismiss()
                BeeInputClient.restoreInputSourceIfNeeded()
                NSApplication.shared.terminate(nil)
            }
            label: {
                MenuActionRow(title: "Quit", shortcut: "⌘Q")
            }
            .buttonStyle(.plain)
            .keyboardShortcut("q", modifiers: [.command])
        }
        .padding(10)
        .frame(width: 260)
    }

    private func menuInfoRow(icon: String, text: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption2)
            Text(text)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }

    private var stateColor: Color {
        switch appState.hotkeyState {
        case .idle:
            return appState.modelStatus == .loaded ? .green : .gray
        case .held, .released:
            return .orange
        case .pushToTalk, .locked, .lockedOptionHeld:
            return .red
        }
    }

    private var modelInfoLabel: String {
        switch appState.modelStatus {
        case .loaded: return AppState.defaultModel.displayName
        case .loading: return "Loading model..."
        case .downloading(let p): return "Downloading (\(Int(p * 100))%)..."
        case .notLoaded: return "No model"
        case .error(let e): return "Error: \(e.prefix(20))"
        }
    }

    private var currentStateLabel: String {
        switch appState.hotkeyState {
        case .idle:
            switch appState.modelStatus {
            case .loaded: return "Ready"
            case .loading: return "Loading model"
            case .downloading: return "Downloading model"
            case .notLoaded: return "No model"
            case .error(let message): return "Error: \(message)"
            }
        case .held, .released:
            return "Starting session"
        case .pushToTalk:
            return "Listening"
        case .locked:
            return "Listening (locked)"
        case .lockedOptionHeld:
            return "Listening (option held)"
        }
    }

}

private struct MenuActionRow: View {
    let title: String
    let shortcut: String

    var body: some View {
        HStack(spacing: 8) {
            Text(title)
            Spacer(minLength: 12)
            Text(shortcut)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .contentShape(Rectangle())
    }
}

struct BeeSettingsView: View {
    @Bindable var appState: AppState

    @State private var runOnStartupEnabled = false
    @State private var pauseMediaEnabled = false

    var body: some View {
        NavigationSplitView {
            List {
                Section("bee") {
                    NavigationLink {
                        BeeOverviewView(appState: appState)
                    } label: {
                        Label("Overview", systemImage: "sparkles")
                    }

                    NavigationLink {
                        BeeHistoryView(appState: appState)
                    } label: {
                        Label("History", systemImage: "clock.arrow.circlepath")
                    }
                }

                Section("guide") {
                    NavigationLink {
                        HowBeeWorksView()
                    } label: {
                        Label("How bee works", systemImage: "keyboard")
                    }
                }

                Section("settings") {
                    NavigationLink {
                        AdvancedSettingsView(
                            appState: appState,
                            runOnStartupEnabled: $runOnStartupEnabled,
                            pauseMediaEnabled: $pauseMediaEnabled
                        )
                    } label: {
                        Label("Advanced", systemImage: "slider.horizontal.3")
                    }
                }
            }
            .listStyle(.sidebar)
            .navigationSplitViewColumnWidth(min: 180, ideal: 220)
        } detail: {
            BeeOverviewView(appState: appState)
        }
        .toolbar(removing: .sidebarToggle)
        .frame(minWidth: 820, minHeight: 560)
        .tint(.orange)
    }
}

private struct BeeOverviewView: View {
    @Bindable var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SettingsCard("Status") {
                    StatusRow(label: "State", value: statusLabel, color: stateColor)
                    StatusRow(label: "Model", value: modelLabel, color: modelColor)
                    KeyValueRow(label: "Input", value: appState.activeInputDeviceName ?? "None")
                }

                if appState.loadedModelDisplayName != nil {
                    SettingsCard("Pipeline") {
                        PipelineRow(
                            label: "ASR",
                            value: AppState.defaultModel.displayName,
                            url: URL(string: "https://huggingface.co/\(AppState.defaultModel.repoID)")!
                        )
                        ForEach(AppState.pipelineComponents, id: \.name) { component in
                            PipelineRow(
                                label: component.role,
                                value: component.name,
                                url: component.url
                            )
                        }
                    }
                }

                SettingsCard("Last transcript") {
                    if let last = appState.transcriptionHistory.first {
                        TranscriptRow(text: last.text, timestamp: last.timestamp)
                    } else {
                        Text("No transcript yet")
                            .foregroundStyle(.secondary)
                    }
                }

                Button("Quit bee") {
                    BeeInputClient.restoreInputSourceIfNeeded()
                    NSApp.terminate(nil)
                }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
                .padding(.top, 4)
            }
            .padding(24)
            .frame(maxWidth: 600, alignment: .leading)
        }
        .navigationTitle("bee")
    }

    private var statusLabel: String {
        switch appState.hotkeyState {
        case .idle: return "Idle"
        case .held, .released: return "Pending"
        case .pushToTalk: return "Push to talk"
        case .locked: return "Locked"
        case .lockedOptionHeld: return "Locked (option held)"
        }
    }

    private var stateColor: Color {
        switch appState.hotkeyState {
        case .idle:
            return appState.modelStatus == .loaded ? .green : .gray
        case .held, .released:
            return .orange
        case .pushToTalk, .locked, .lockedOptionHeld:
            return .red
        }
    }

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: return "Not loaded"
        case .downloading(let progress): return "Downloading (\(Int(progress * 100))%)"
        case .loading: return "Loading"
        case .loaded: return AppState.defaultModel.displayName
        case .error(let message): return "Error: \(message)"
        }
    }

    private var modelColor: Color {
        switch appState.modelStatus {
        case .loaded: return .green
        case .loading, .downloading: return .orange
        case .error: return .red
        case .notLoaded: return .gray
        }
    }
}

private struct BeeHistoryView: View {
    @Bindable var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SettingsCard("Recent transcripts") {
                    if appState.transcriptionHistory.isEmpty {
                        Text("No transcripts yet")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(appState.transcriptionHistory.prefix(50)) { item in
                            TranscriptRow(text: item.text, timestamp: item.timestamp)
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 600, alignment: .leading)
        }
        .navigationTitle("History")
    }
}

private struct HowBeeWorksView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                SettingsCard("Shortcuts") {
                    ShortcutRow(action: "Push-to-talk", keys: ["Right ⌥"], gesture: "hold")
                    ShortcutRow(action: "Hands-free mode", keys: ["Right ⌥"], gesture: "tap")
                    ShortcutRow(action: "Lock (while PTT)", keys: ["Right ⌘"])
                    ShortcutRow(action: "Submit", keys: ["↩"])
                    ShortcutRow(action: "Cancel", keys: ["⎋"])
                }

                Text("Hold Right Option to record while held (push-to-talk). Tap it quickly to start hands-free mode — press Return to submit or Escape to cancel.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: 600, alignment: .leading)
            .padding(24)
        }
        .navigationTitle("How bee works")
    }
}

private struct AdvancedSettingsView: View {
    @Bindable var appState: AppState
    @Binding var runOnStartupEnabled: Bool
    @Binding var pauseMediaEnabled: Bool

    private static let chunkSizeOptions: [(label: String, value: Float)] = [
        ("0.2s", 0.2),
        ("0.35s", 0.35),
        ("0.5s", 0.5),
        ("0.75s", 0.75),
        ("1s", 1.0),
        ("1.5s", 1.5),
        ("2s", 2.0),
        ("2.5s", 2.5),
        ("3s", 3.0),
    ]

    private static let streamingTokenOptions: [(label: String, value: UInt32)] = [
        ("default", 0), ("8", 8), ("16", 16), ("32", 32), ("64", 64),
    ]

    private static let finalTokenOptions: [(label: String, value: UInt32)] = [
        ("default", 0), ("64", 64), ("128", 128), ("256", 256), ("512", 512),
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SettingsCard("Audio") {
                    VStack(alignment: .leading, spacing: 0) {
                        if appState.availableInputDevices.isEmpty {
                            Text("No input devices")
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(Array(appState.availableInputDevices.enumerated()), id: \.element.uid) { index, device in
                                AudioDeviceRow(
                                    device: device,
                                    isActive: device.uid == appState.activeInputDeviceUID,
                                    isWarm: appState.audioEngine.deviceWarmPolicy[device.uid] ?? false,
                                    onSelect: { appState.selectInputDevice(uid: device.uid) },
                                    onToggleWarm: { appState.setDeviceWarmPolicy(uid: device.uid, warm: $0) }
                                )
                                if index < appState.availableInputDevices.count - 1 {
                                    Divider().padding(.vertical, 4)
                                }
                            }
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                SettingsCard("Transcription") {
                    VStack(alignment: .leading, spacing: 12) {
                        chunkSizePicker
                        tokenLimitPicker(
                            label: "Streaming tokens",
                            options: Self.streamingTokenOptions,
                            current: appState.maxNewTokensStreaming
                        ) {
                            appState.maxNewTokensStreaming = $0
                        }
                        tokenLimitPicker(
                            label: "Final tokens",
                            options: Self.finalTokenOptions,
                            current: appState.maxNewTokensFinal
                        ) {
                            appState.maxNewTokensFinal = $0
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                SettingsCard("Behavior") {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle("Run on startup", isOn: $runOnStartupEnabled)
                        Toggle("Pause media while dictating", isOn: $pauseMediaEnabled)
                        Toggle("Debug overlay", isOn: Binding(
                            get: { appState.debugEnabled },
                            set: {
                                appState.debugEnabled = $0
                                if $0 {
                                    DebugPanel.shared.show(appState: appState)
                                } else {
                                    DebugPanel.shared.hide()
                                }
                            }
                        ))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                Text("Startup/media toggles are UI scaffolding for now and still need backend wiring.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: 600, alignment: .leading)
            .padding(24)
        }
        .navigationTitle("Advanced")
        .onAppear {
            if appState.debugEnabled {
                DebugPanel.shared.show(appState: appState)
            }
        }
    }


    private var chunkSizePicker: some View {
        Picker("Chunk size", selection: Binding(
            get: { appState.chunkSizeSec },
            set: { appState.chunkSizeSec = $0 }
        )) {
            ForEach(Self.chunkSizeOptions, id: \.value) { option in
                Text(option.label).tag(option.value)
            }
        }
        .pickerStyle(.menu)
    }

    private func tokenLimitPicker(
        label: String,
        options: [(label: String, value: UInt32)],
        current: UInt32,
        action: @escaping (UInt32) -> Void
    ) -> some View {
        Picker(label, selection: Binding(
            get: { current },
            set: action
        )) {
            ForEach(options, id: \.value) { option in
                Text(option.label).tag(option.value)
            }
        }
        .pickerStyle(.menu)
    }
}

private struct SettingsCard<Content: View>: View {
    let title: String
    @ViewBuilder var content: Content

    init(_ title: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(title)
                .font(.headline)
            content
        }
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.12), lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.05), radius: 4, x: 0, y: 1)
        )
    }
}

private struct StatusRow: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(value)
                .fontWeight(.medium)
                .multilineTextAlignment(.trailing)
        }
    }
}

private struct PipelineRow: View {
    let label: String
    let value: String
    let url: URL

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Link(value, destination: url)
                .fontWeight(.medium)
        }
    }
}

private struct KeyValueRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 10) {
            if !label.isEmpty {
                Text(label)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            Text(value)
                .fontWeight(.medium)
                .multilineTextAlignment(.trailing)
        }
    }
}

private struct TranscriptRow: View {
    let text: String
    var timestamp: Date? = nil

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        } label: {
            VStack(alignment: .leading, spacing: 4) {
                if let timestamp {
                    Text(timestamp, style: .relative)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                Text(text)
                    .lineLimit(3)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .fill(Color(nsColor: .quaternaryLabelColor).opacity(0.08))
            )
        }
        .buttonStyle(.plain)
        .help("Click to copy")
    }
}

private struct ShortcutRow: View {
    let action: String
    let keys: [String]
    var gesture: String? = nil

    var body: some View {
        HStack(spacing: 12) {
            Text(action)
                .foregroundStyle(.secondary)
            Spacer()
            HStack(spacing: 6) {
                ForEach(Array(keys.enumerated()), id: \.offset) { _, key in
                    ShortcutKeycap(text: key)
                }
                if let gesture {
                    Text(gesture)
                        .font(.system(size: 12, design: .rounded))
                        .foregroundStyle(.tertiary)
                        .italic()
                }
            }
        }
    }
}

private struct AudioDeviceRow: View {
    let device: AppState.InputDeviceInfo
    let isActive: Bool
    let isWarm: Bool
    let onSelect: () -> Void
    let onToggleWarm: (Bool) -> Void

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isActive ? .orange : .secondary)

            VStack(alignment: .leading, spacing: 2) {
                Text(device.name)
                    .fontWeight(isActive ? .medium : .regular)
                if device.isBuiltIn {
                    Text("Built-in")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer()

            Toggle("Keep warm", isOn: Binding(
                get: { isWarm },
                set: { onToggleWarm($0) }
            ))
            .toggleStyle(.switch)
            .labelsHidden()
            .controlSize(.small)
            .help("Keep this device's audio engine running for instant start")
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture { onSelect() }
    }
}

private struct ShortcutKeycap: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .semibold, design: .rounded))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color(nsColor: .controlBackgroundColor))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor), lineWidth: 1)
            )
    }
}
