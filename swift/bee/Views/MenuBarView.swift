import AppKit
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                openWindow(id: "bee-main")
            } label: {
                Label("Open Bee", systemImage: "app.badge")
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)

            SettingsLink {
                Label("Settings…", systemImage: "gearshape")
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
            .keyboardShortcut("q")
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
                NavigationLink {
                    HowBeeWorksView()
                } label: {
                    Label("How Bee Works", systemImage: "book")
                }

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
            .navigationSplitViewColumnWidth(min: 180, ideal: 200)
        } detail: {
            HowBeeWorksView()
        }
        .frame(minWidth: 760, minHeight: 520)
    }
}

private struct HowBeeWorksView: View {
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("How Bee Works")
                    .font(.title2.weight(.semibold))

                settingSection(
                    title: "Core Flow",
                    body: "1. Hold Right Option to start.\n2. Keep holding for push-to-talk, or tap Right Command to lock.\n3. Press Enter to submit or Escape to cancel."
                )

                settingSection(
                    title: "Mental Model",
                    body: "Bee is designed to stay out of your way: use hotkeys for speed, use the Bee window for runtime visibility, and use this Settings window for learning + advanced tuning."
                )

                settingSection(
                    title: "Best Practices",
                    body: "Use the default chunk and token settings unless you have a specific latency/quality issue to solve. If audio quality changes, revisit input device + keep-warm first."
                )
            }
            .frame(maxWidth: 700, alignment: .leading)
            .padding(24)
        }
    }

    private func settingSection(title: String, body: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            Text(body)
                .font(.body)
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

                GroupBox("Audio") {
                    VStack(alignment: .leading, spacing: 12) {
                        inputDevicePicker
                        Toggle("Keep active input warm", isOn: Binding(
                            get: { appState.activeInputDeviceKeepWarm },
                            set: { _ in appState.toggleActiveInputDeviceKeepWarm() }
                        ))
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                GroupBox("Transcription") {
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

                GroupBox("Behavior") {
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
