import AppKit
import ServiceManagement
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState
    @Environment(\.openSettings) private var openSettings
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            InputDeviceList(appState: appState)
                .padding(.horizontal, 6)

            Divider().padding(.horizontal, 2)
            if appState.transcriptionHistory.isEmpty {
                Text("Transcripts will appear here")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
            } else {
                VStack(spacing: 4) {
                    ForEach(appState.transcriptionHistory.prefix(3)) { item in
                        TranscriptRow(text: item.text, timestamp: item.timestamp)
                    }
                }
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
        .frame(width: 340)
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

    @State private var isHovered = false

    var body: some View {
        HStack(spacing: 8) {
            Text(title)
            Spacer(minLength: 12)
            Text(shortcut)
                .font(.system(.body, design: .rounded))
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .fill(isHovered ? Color.primary.opacity(0.08) : .clear)
        )
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
    }
}

enum SidebarItem: Hashable {
    case overview, history, howBeeWorks, advanced
}

struct BeeSettingsView: View {
    @Bindable var appState: AppState

    @State private var selection: SidebarItem = .overview

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                List(selection: $selection) {
                    Section {
                        Label("Overview", systemImage: "sparkles")
                            .tag(SidebarItem.overview)
                        Label("History", systemImage: "clock.arrow.circlepath")
                            .tag(SidebarItem.history)
                        Label("How bee works", systemImage: "keyboard")
                            .tag(SidebarItem.howBeeWorks)
                    }

                    Section {
                        Label("Settings", systemImage: "gearshape")
                            .tag(SidebarItem.advanced)
                    }
                }
                .listStyle(.sidebar)

                Spacer()

                HStack(spacing: 8) {
                    Image("BeeColor")
                        .resizable()
                        .frame(width: 20, height: 20)
                    Text("bee")
                        .font(.callout.weight(.semibold))
                    if let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String {
                        Text("v\(version)")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 12)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(width: 200)

            Divider()

            Group {
                switch selection {
                case .overview:
                    BeeOverviewView(appState: appState)
                case .history:
                    BeeHistoryView(appState: appState)
                case .howBeeWorks:
                    HowBeeWorksView()
                case .advanced:
                    AdvancedSettingsView(appState: appState)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(minWidth: 780, minHeight: 520)
        .tint(.orange)
    }
}

private struct BeeOverviewView: View {
    @Bindable var appState: AppState

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Stats
                if appState.totalSessions > 0 {
                    HStack(spacing: 12) {
                        StatCard(
                            icon: "waveform",
                            value: "\(appState.totalSessions)",
                            label: appState.totalSessions == 1 ? "session" : "sessions"
                        )
                        StatCard(
                            icon: "text.word.spacing",
                            value: formatNumber(appState.totalWords),
                            label: appState.totalWords == 1 ? "word" : "words"
                        )
                        StatCard(
                            icon: "character.cursor.ibeam",
                            value: formatNumber(appState.totalCharacters),
                            label: "characters"
                        )
                    }
                }

                // Input device + status
                HStack(spacing: 12) {
                    InputDeviceList(appState: appState)

                    Spacer()

                    HStack(spacing: 6) {
                        Circle()
                            .fill(stateColor)
                            .frame(width: 8, height: 8)
                        Text(statusLabel)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                }

                // Last transcript
                SettingsCard("Recent") {
                    if appState.transcriptionHistory.isEmpty {
                        HStack(spacing: 8) {
                            Image(systemName: "text.bubble")
                                .foregroundStyle(.quaternary)
                            Text("No transcripts yet")
                                .foregroundStyle(.secondary)
                        }
                    } else {
                        ForEach(appState.transcriptionHistory.prefix(3)) { item in
                            TranscriptRow(text: item.text, timestamp: item.timestamp)
                        }
                    }
                }

                // Footer
                HStack(spacing: 16) {
                    Link("GitHub", destination: URL(string: "https://github.com/fasterthanlime/bee")!)
                    Text("·").foregroundStyle(.quaternary)
                    Button("Quit bee") {
                        BeeInputClient.restoreInputSourceIfNeeded()
                        NSApp.terminate(nil)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.secondary)
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .padding(24)
            .frame(maxWidth: 600)
        }
    }

    private func formatNumber(_ n: Int) -> String {
        if n >= 1000 {
            return String(format: "%.1fk", Double(n) / 1000)
        }
        return "\(n)"
    }

    private var statusLabel: String {
        switch appState.hotkeyState {
        case .idle:
            switch appState.modelStatus {
            case .loaded: return "Ready"
            case .loading: return "Loading model..."
            case .downloading(let p): return "Downloading (\(Int(p * 100))%)..."
            case .notLoaded: return "No model loaded"
            case .error(let e): return "Error: \(e)"
            }
        case .held, .released: return "Starting session..."
        case .pushToTalk: return "Listening"
        case .locked: return "Listening (locked)"
        case .lockedOptionHeld: return "Listening (option held)"
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

}

private struct StatCard: View {
    let icon: String
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 6) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(.orange)
            Text(value)
                .font(.title2.weight(.bold).monospacedDigit())
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 14)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay(
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.08), lineWidth: 1)
                )
        )
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

    }
}

private struct AdvancedSettingsView: View {
    @Bindable var appState: AppState

    @State private var showDiagSheet = false
    @State private var diagOutput = ""

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
                SettingsCard("Pipeline") {
                    HStack(spacing: 10) {
                        Text("ASR")
                            .foregroundStyle(.secondary)
                        Spacer()
                        Circle()
                            .fill(modelColor)
                            .frame(width: 8, height: 8)
                        if appState.modelStatus == .loaded {
                            Link(destination: URL(string: "https://huggingface.co/\(AppState.defaultModel.repoID)")!) {
                                HStack(spacing: 4) {
                                    Text(modelLabel)
                                        .underline()
                                    Image(systemName: "arrow.up.right.square")
                                        .font(.caption)
                                }
                            }
                        } else {
                            Text(modelLabel)
                                .fontWeight(.medium)
                        }
                    }

                    if appState.modelStatus == .loaded {
                        ForEach(AppState.pipelineComponents, id: \.name) { component in
                            PipelineRow(
                                label: component.role,
                                value: component.name,
                                url: component.url
                            )
                        }
                    }
                }

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

                SettingsCard("Behavior") {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle("Run on startup", isOn: Binding(
                            get: { SMAppService.mainApp.status == .enabled },
                            set: { enable in
                                do {
                                    if enable {
                                        try SMAppService.mainApp.register()
                                    } else {
                                        try SMAppService.mainApp.unregister()
                                    }
                                } catch {
                                    beeLog("SMAppService error: \(error)")
                                }
                            }
                        ))

                        Toggle("Lower volume during dictation", isOn: Binding(
                            get: { appState.lowerVolumeDuringDictation },
                            set: { appState.lowerVolumeDuringDictation = $0 }
                        ))
                        if appState.lowerVolumeDuringDictation {
                            HStack(spacing: 8) {
                                Image(systemName: "speaker.fill")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Slider(
                                    value: Binding(
                                        get: { appState.dictationVolumeLevel },
                                        set: { appState.dictationVolumeLevel = $0 }
                                    ),
                                    in: 0...1
                                )
                                Image(systemName: "speaker.wave.3.fill")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text("\(Int(appState.dictationVolumeLevel * 100))%")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                                    .frame(width: 36, alignment: .trailing)
                            }
                            .padding(.leading, 20)
                        }
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

                Text("Advanced")
                    .font(.headline)
                    .padding(.top, 8)

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
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                SettingsCard("Diagnostics") {
                    VStack(alignment: .leading, spacing: 8) {
                        Button("Dump audio device info") {
                            diagOutput = AudioDiagnostics.dumpAllDevices()
                            showDiagSheet = true
                        }
                    }
                }
            }
            .frame(maxWidth: 600, alignment: .leading)
            .padding(24)
        }

        .onAppear {
            if appState.debugEnabled {
                DebugPanel.shared.show(appState: appState)
            }
        }
        .sheet(isPresented: $showDiagSheet) {
            VStack(spacing: 0) {
                HStack {
                    Text("Audio Device Info")
                        .font(.headline)
                    Spacer()
                    Button("Copy") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(diagOutput, forType: .string)
                    }
                    Button("Close") { showDiagSheet = false }
                        .keyboardShortcut(.cancelAction)
                }
                .padding()

                Divider()

                ScrollView {
                    Text(diagOutput)
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundStyle(.primary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                }
                .background(Color(nsColor: .textBackgroundColor))
            }
            .frame(width: 600, height: 500)
        }
    }

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: return "Not loaded"
        case .downloading(let progress): return "Downloading (\(Int(progress * 100))%)"
        case .loading: return "Loading..."
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

private struct InputDeviceList: View {
    @Bindable var appState: AppState

    var body: some View {
        if appState.availableInputDevices.isEmpty {
            Text("No input devices")
                .foregroundStyle(.secondary)
        } else {
            VStack(spacing: 2) {
                ForEach(appState.availableInputDevices, id: \.uid) { device in
                    InputDeviceListRow(
                        device: device,
                        isActive: device.uid == appState.activeInputDeviceUID,
                        onSelect: { appState.selectInputDevice(uid: device.uid) }
                    )
                }
            }
        }
    }
}

private struct InputDeviceListRow: View {
    let device: AppState.InputDeviceInfo
    let isActive: Bool
    let onSelect: () -> Void

    @State private var isHovered = false

    var body: some View {
        Button(action: onSelect) {
            HStack(spacing: 8) {
                Image(systemName: device.iconName)
                    .font(.title3)
                    .foregroundStyle(isActive ? .orange : .secondary)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 1) {
                    Text(device.name)
                        .fontWeight(isActive ? .medium : .regular)
                    if let subtitle = device.subtitle {
                        Text(subtitle)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                }
                Spacer()
                if isActive {
                    Image(systemName: "checkmark")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.orange)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isHovered ? Color.primary.opacity(0.06) : .clear)
            )
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
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
            Link(destination: url) {
                HStack(spacing: 4) {
                    Text(value)
                        .underline()
                    Image(systemName: "arrow.up.right.square")
                        .font(.caption)
                }
            }
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

    @State private var copied = false
    @State private var isHovered = false

    private func copyText() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
        copied = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            copied = false
        }
    }

    private var tailText: String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count > 200 {
            let start = trimmed.index(trimmed.endIndex, offsetBy: -200)
            return "…" + trimmed[start...]
        }
        return trimmed
    }

    var body: some View {
        Button {
            copyText()
        } label: {
            HStack(alignment: .center, spacing: 8) {
                Text(tailText)
                    .lineLimit(1)
                Spacer(minLength: 4)
                ZStack {
                    // Always reserve space for timestamp text
                    Text("00 min")
                        .font(.caption)
                        .hidden()
                    // Show either checkmark, copy icon, or timestamp
                    if copied {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption2)
                            .foregroundColor(.green)
                    } else if isHovered {
                        Image(systemName: "doc.on.doc")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    } else if let timestamp {
                        Text(timestamp, style: .relative)
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    }
                }
                .frame(alignment: .trailing)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isHovered ? Color.primary.opacity(0.06) : .clear)
            )
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(text)
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
            ZStack {
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(isActive ? Color.orange.opacity(0.12) : Color.primary.opacity(0.04))
                    .frame(width: 32, height: 32)
                Image(systemName: device.iconName)
                    .font(.system(size: 14))
                    .foregroundStyle(isActive ? .orange : .secondary)
            }

            VStack(alignment: .leading, spacing: 2) {
                Text(device.name)
                    .fontWeight(isActive ? .medium : .regular)
                if let subtitle = device.subtitle {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer()

            VStack(spacing: 2) {
                Toggle("Keep warm", isOn: Binding(
                    get: { isWarm },
                    set: { onToggleWarm($0) }
                ))
                .toggleStyle(.switch)
                .labelsHidden()
                .controlSize(.small)
                Text("Keep warm")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onTapGesture { onSelect() }
    }
}

private struct TooltipArrow: Shape {
    func path(in rect: CGRect) -> Path {
        Path { p in
            p.move(to: CGPoint(x: rect.minX, y: rect.minY))
            p.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
            p.addLine(to: CGPoint(x: rect.maxX, y: rect.minY))
            p.closeSubpath()
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
