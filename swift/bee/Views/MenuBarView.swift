import AppKit
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState
    @Environment(\.openSettings) private var openSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("CURRENT STATE")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 6)

            Text(currentStateLabel)
                .font(.body.weight(.medium))
                .padding(.horizontal, 6)

            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Image(systemName: "mic.fill")
                    .symbolRenderingMode(.monochrome)
                    .foregroundStyle(.primary)
                    .font(.caption)
                Text(appState.activeInputDeviceName ?? "No input")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .foregroundStyle(.primary)
            }
            .font(.caption)
            .padding(.horizontal, 6)

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
        .frame(width: 240)
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
                HStack(spacing: 12) {
                    Image("BeeColor")
                        .resizable()
                        .frame(width: 34, height: 34)
                    Text("bee")
                        .font(.title2.weight(.semibold))
                }

                SettingsCard("Status") {
                    KeyValueRow(label: "State", value: statusLabel)
                    KeyValueRow(label: "Model", value: modelLabel)
                    KeyValueRow(label: "Input", value: appState.activeInputDeviceName ?? "None")
                }

                SettingsCard("Last transcript") {
                    if let last = appState.transcriptionHistory.first {
                        TranscriptRow(text: last.text)
                    } else {
                        Text("No transcript yet")
                            .foregroundStyle(.secondary)
                    }
                }

                SettingsCard("App") {
                    Button("Quit bee") {
                        BeeInputClient.restoreInputSourceIfNeeded()
                        NSApp.terminate(nil)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
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

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: return "Not loaded"
        case .downloading(let progress): return "Downloading (\(Int(progress * 100))%)"
        case .loading: return "Loading"
        case .loaded: return "Loaded"
        case .error(let message): return "Error: \(message)"
        }
    }
}

private struct BeeHistoryView: View {
    @Bindable var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("History")
                    .font(.title2.weight(.semibold))

                SettingsCard("Recent transcripts") {
                    if appState.transcriptionHistory.isEmpty {
                        Text("No transcripts yet")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(appState.transcriptionHistory.prefix(50)) { item in
                            TranscriptRow(text: item.text)
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
        }
        .navigationTitle("History")
    }
}

private struct HowBeeWorksView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("How bee works")
                    .font(.title2.weight(.semibold))

                SettingsCard("Core flow") {
                    ShortcutRow(action: "Start listening", keys: ["⌥", "→"])
                    ShortcutRow(action: "Lock listening", keys: ["⌘", "→"])
                    ShortcutRow(action: "Submit", keys: ["↩"])
                    ShortcutRow(action: "Cancel", keys: ["esc"])
                }
            }
            .frame(maxWidth: 760, alignment: .leading)
            .padding(24)
        }
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
                Text("Advanced")
                    .font(.title2.weight(.semibold))

                SettingsCard("Audio") {
                    VStack(alignment: .leading, spacing: 12) {
                        inputDevicePicker
                        Toggle("Keep active input warm", isOn: Binding(
                            get: { appState.activeInputDeviceKeepWarm },
                            set: { _ in appState.toggleActiveInputDeviceKeepWarm() }
                        ))
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
            .frame(maxWidth: 760, alignment: .leading)
            .padding(24)
        }
        .onAppear {
            if appState.debugEnabled {
                DebugPanel.shared.show(appState: appState)
            }
        }
    }

    private var inputDevicePicker: some View {
        Picker("Input device", selection: Binding(
            get: { appState.activeInputDeviceUID ?? "" },
            set: { appState.selectInputDevice(uid: $0) }
        )) {
            if appState.availableInputDevices.isEmpty {
                Text("No input devices")
                    .tag("")
            } else {
                ForEach(appState.availableInputDevices, id: \.uid) { device in
                    Text(device.name).tag(device.uid)
                }
            }
        }
        .pickerStyle(.menu)
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
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(Color(nsColor: .windowBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                )
                .shadow(color: .black.opacity(0.05), radius: 4, x: 0, y: 1)
        )
    }
}

private struct KeyValueRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
            Spacer()
            Text(value)
                .fontWeight(.medium)
                .multilineTextAlignment(.trailing)
        }
    }
}

private struct TranscriptRow: View {
    let text: String

    var body: some View {
        Button {
            NSPasteboard.general.clearContents()
            NSPasteboard.general.setString(text, forType: .string)
        } label: {
            Text(text)
                .lineLimit(3)
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

    var body: some View {
        HStack(spacing: 12) {
            Text(action)
                .foregroundStyle(.secondary)
            Spacer()
            HStack(spacing: 6) {
                ForEach(Array(keys.enumerated()), id: \.offset) { _, key in
                    ShortcutKeycap(text: key)
                }
            }
        }
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
