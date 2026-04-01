import AppKit
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                NSApp.activate(ignoringOtherApps: true)
                openSettings()
                DispatchQueue.main.async {
                    let normalWindow = NSApp.orderedWindows.first { window in
                        !(window is NSPanel)
                    }
                    normalWindow?.makeKeyAndOrderFront(nil)
                }
            } label: {
                Label("Open Bee", systemImage: "app.badge")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            Divider().padding(.horizontal, 2)

            Button {
                BeeInputClient.restoreInputSourceIfNeeded()
                NSApplication.shared.terminate(nil)
            } label: {
                Label("Quit Bee", systemImage: "power")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
        }
        .padding(10)
        .frame(width: 220)
    }
}

struct BeeSettingsView: View {
    @Bindable var appState: AppState

    @State private var runOnStartupEnabled = false
    @State private var pauseMediaEnabled = false

    var body: some View {
        NavigationSplitView {
            List {
                Section("Bee") {
                    NavigationLink {
                        BeeOverviewView(appState: appState)
                    } label: {
                        Label("Overview", systemImage: "app.badge")
                    }
                }

                Section("Guide") {
                    NavigationLink {
                        HowBeeWorksView()
                    } label: {
                        Label("How Bee Works", systemImage: "book.closed")
                    }
                }

                Section("Settings") {
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
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            BeeOverviewView(appState: appState)
        }
        .frame(minWidth: 760, minHeight: 520)
        .tint(.orange)
    }
}

private struct BeeOverviewView: View {
    @Bindable var appState: AppState

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("Bee")
                    .font(.title2.weight(.semibold))

                SettingsCard("Status") {
                    KeyValueRow(label: "State", value: statusLabel)
                    KeyValueRow(label: "Model", value: modelLabel)
                    KeyValueRow(label: "Input", value: appState.activeInputDeviceName ?? "None")
                }

                SettingsCard("Last Transcript") {
                    if let last = appState.transcriptionHistory.first {
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(last.text, forType: .string)
                        } label: {
                            Text(last.text)
                                .lineLimit(4)
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
                    } else {
                        Text("No transcript yet")
                            .foregroundStyle(.secondary)
                    }
                }

                SettingsCard("Recent Transcripts") {
                    if appState.transcriptionHistory.isEmpty {
                        Text("No transcripts yet")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(appState.transcriptionHistory.prefix(12)) { item in
                            Button {
                                NSPasteboard.general.clearContents()
                                NSPasteboard.general.setString(item.text, forType: .string)
                            } label: {
                                Text(item.displayText)
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
                            .help("Click to copy full transcript")
                        }
                    }
                }

                SettingsCard("App") {
                    Button("Quit Bee") {
                        BeeInputClient.restoreInputSourceIfNeeded()
                        NSApp.terminate(nil)
                    }
                    .buttonStyle(.borderedProminent)
                }
            }
            .padding(24)
            .frame(maxWidth: 760, alignment: .leading)
        }
        .navigationTitle("Bee")
    }

    private var statusLabel: String {
        switch appState.uiState {
        case .idle: return "Idle"
        case .pending: return "Pending"
        case .pushToTalk: return "Push To Talk"
        case .locked: return "Locked"
        case .lockedOptionHeld: return "Locked (Option Held)"
        }
    }

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: return "Not Loaded"
        case .downloading(let progress): return "Downloading (\(Int(progress * 100))%)"
        case .loading: return "Loading"
        case .loaded: return "Loaded"
        case .error(let message): return "Error: \(message)"
        }
    }
}

private struct HowBeeWorksView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("How Bee Works")
                    .font(.title2.weight(.semibold))

                SettingsCard("Core Flow") {
                    guideRow(number: "1", text: "Hold Right Option to start.")
                    guideRow(number: "2", text: "Keep holding for push-to-talk, or tap Right Command to lock.")
                    guideRow(number: "3", text: "Press Enter to submit or Escape to cancel.")
                }

                SettingsCard("Mental Model") {
                    Text("Bee is designed to stay out of your way: use hotkeys for speed, use this window for visibility and tuning.")
                        .foregroundStyle(.secondary)
                }

                SettingsCard("Best Practices") {
                    Text("Leave defaults unless you are solving a specific latency or quality issue. Start with input device and keep-warm if audio quality shifts.")
                        .foregroundStyle(.secondary)
                }
            }
            .frame(maxWidth: 700, alignment: .leading)
            .padding(24)
        }
    }

    private func guideRow(number: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Text(number)
                .font(.caption.weight(.bold))
                .foregroundStyle(.white)
                .frame(width: 20, height: 20)
                .background(
                    Circle()
                        .fill(.orange)
                )
            Text(text)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
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
                                DebugPanel.shared.toggle(appState: appState)
                            }
                        ))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                Text("Startup/media toggles are UI scaffolding for now and still need backend wiring.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: 700, alignment: .leading)
            .padding(24)
        }
    }

    private var inputDevicePicker: some View {
        Picker("Input device", selection: Binding(
            get: { appState.activeInputDeviceUID ?? "" },
            set: { appState.selectInputDevice(uid: $0) }
        )) {
            if appState.availableInputDevices.isEmpty {
                Text("No Input Devices")
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
