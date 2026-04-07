import AppKit
import ServiceManagement
import SwiftUI

struct MenuBarView: View {
    @Bindable var appState: AppState
    var body: some View {
        HStack(spacing: 0) {
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
                            TranscriptRow(text: item.text, timestamp: item.timestamp, appIcon: item.appIcon)
                        }
                    }
                    .padding(.horizontal, 6)
                }

                Divider().padding(.horizontal, 2)

                Button {
                    NSApp.keyWindow?.orderOut(nil)
                    SettingsOpener.action?()
                    NSApp.activate(ignoringOtherApps: true)
                    DispatchQueue.main.async {
                        for window in NSApp.windows where !(window is NSPanel) {
                            window.makeKeyAndOrderFront(nil)
                            return
                        }
                    }
                }
                label: {
                    MenuActionRow(title: "Settings", shortcut: "⌘,")
                }
                .buttonStyle(.plain)
                .keyboardShortcut(",", modifiers: [.command])

                Button {
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

            Divider()
                .padding(.vertical, 8)

            VStack(spacing: 6) {
                Button {
                    if appState.echoActive {
                        appState.audioEngine.stopEcho()
                    } else {
                        appState.audioEngine.startEcho()
                    }
                    appState.echoActive = appState.audioEngine.echoEnabled
                } label: {
                    Image(systemName: appState.echoActive ? "ear.fill" : "ear")
                        .font(.body)
                        .foregroundStyle(appState.echoActive ? .orange : .secondary)
                        .frame(width: 28, height: 28)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(appState.echoActive ? Color.orange.opacity(0.15) : .clear)
                        )
                }
                .buttonStyle(.plain)
                .help("Listen to yourself (1s delay)")

                HStack(spacing: 6) {
                    VerticalVolumeSlider(
                        audioEngine: appState.audioEngine,
                        selectedDeviceUID: appState.activeInputDeviceUID
                    )
                    .frame(width: 6)
                    VerticalLevelMeter(audioEngine: appState.audioEngine)
                }
            }
            .padding(.vertical, 12)
            .padding(.trailing, 12)
            .padding(.leading, 4)
        }
        .frame(width: 420)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(.primary.opacity(0.1), lineWidth: 0.5)
        }
        .shadow(color: .black.opacity(0.2), radius: 12, y: 4)
        .onAppear {
            beeLog("MENUBAR: panel opened")
            appState.menuBarPanelOpen = true
            if !appState.audioEngine.isWarm {
                beeLog("MENUBAR: warming up audio engine for level meter")
                do {
                    try appState.audioEngine.warmUp()
                    beeLog("MENUBAR: audio engine warm")
                } catch {
                    beeLog("MENUBAR: warmUp failed: \(error)")
                }
            }
        }
        .onDisappear {
            beeLog("MENUBAR: panel closed")
            appState.menuBarPanelOpen = false
            if appState.echoActive {
                appState.audioEngine.stopEcho()
                appState.echoActive = false
            }
            // Grace period: keep engine warm for 5 seconds after closing
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak appState] in
                guard let appState else { return }
                // Only cool down if panel wasn't reopened and no session/policy needs warmth
                guard !appState.menuBarPanelOpen,
                      !appState.hotkeyState.isRecording,
                      !appState.activeInputDeviceKeepWarm else { return }
                beeLog("MENUBAR: grace period expired, cooling down")
                appState.audioEngine.coolDown()
            }
        }
        .background {
            Button("") {
                appState.debugEnabled.toggle()
                if appState.debugEnabled {
                    DebugPanel.shared.show(appState: appState)
                } else {
                    DebugPanel.shared.hide()
                }
            }
            .keyboardShortcut("d", modifiers: [.command])
            .hidden()
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
    case overview, history, howBeeWorks
    case audio, transcription, general
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

                    Section("Settings") {
                        Label("Audio", systemImage: "waveform")
                            .tag(SidebarItem.audio)
                        Label("Transcription", systemImage: "text.quote")
                            .badge(appState.modelStatus.hasError ? 1 : 0)
                            .tag(SidebarItem.transcription)
                        Label("General", systemImage: "gearshape")
                            .tag(SidebarItem.general)
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
                    BeeOverviewView(appState: appState, selection: $selection)
                case .audio:
                    AudioSettingsView(appState: appState)
                case .transcription:
                    TranscriptionSettingsView(appState: appState)
                case .history:
                    BeeHistoryView(appState: appState)
                case .howBeeWorks:
                    HowBeeWorksView()
                case .general:
                    GeneralSettingsView(appState: appState)
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
    @Binding var selection: SidebarItem

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
                    DeviceDropdown(appState: appState)
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
                        if let first = appState.transcriptionHistory.first {
                            TranscriptRow(text: first.text, timestamp: first.timestamp, appIcon: first.appIcon)
                        }
                        if appState.transcriptionHistory.count > 1 {
                            Button {
                                selection = .history
                            } label: {
                                HStack(spacing: 4) {
                                    Text("View all \(appState.transcriptionHistory.count) transcripts")
                                    Image(systemName: "arrow.right")
                                        .font(.caption)
                                }
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
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
            if appState.modelStatus.hasError { return .red }
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
                            TranscriptRow(text: item.text, timestamp: item.timestamp, appIcon: item.appIcon)
                        }
                    }
                }
            }
            .padding(24)
            .frame(maxWidth: 600, alignment: .leading)
        }

    }
}

// MARK: - Audio Settings

private struct AudioSettingsView: View {
    @Bindable var appState: AppState
    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Input Devices")
                    .font(.headline)
                Text("Drag to set priority. bee auto-switches to the highest-priority available device.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                List {
                    ForEach(Array(allDevices.enumerated()), id: \.element.device.uid) { index, entry in
                        AudioSettingsDeviceRow(
                            device: entry.device,
                            rank: index + 1,
                            isActive: entry.device.uid == appState.activeInputDeviceUID,
                            isOnline: entry.isOnline,
                            isWarm: appState.audioEngine.deviceWarmPolicy[entry.device.uid] ?? false,
                            isHidden: appState.hiddenDeviceUIDs.contains(entry.device.uid),
                            onSelect: entry.isOnline ? { appState.selectInputDevice(uid: entry.device.uid) } : nil,
                            onToggleWarm: { appState.toggleDeviceWarmPolicy(uid: entry.device.uid) },
                            onToggleHidden: { appState.toggleDeviceHidden(uid: entry.device.uid) }
                        )
                    }
                    .onMove { source, destination in
                        var uids = allDevices.map(\.device.uid)
                        uids.move(fromOffsets: source, toOffset: destination)
                        appState.setDevicePriority(orderedUIDs: uids)
                    }
                }
                .listStyle(.bordered(alternatesRowBackgrounds: false))
            }

            // Level meter + controls
            VStack(spacing: 12) {
                Button {
                    if appState.echoActive {
                        appState.audioEngine.stopEcho()
                    } else {
                        if !appState.audioEngine.isWarm {
                            try? appState.audioEngine.warmUp()
                        }
                        appState.audioEngine.startEcho()
                    }
                    appState.echoActive = appState.audioEngine.echoEnabled
                } label: {
                    Image(systemName: appState.echoActive ? "ear.fill" : "ear")
                        .font(.title3)
                        .foregroundStyle(appState.echoActive ? .orange : .secondary)
                        .frame(width: 36, height: 36)
                        .background(
                            RoundedRectangle(cornerRadius: 8, style: .continuous)
                                .fill(appState.echoActive ? Color.orange.opacity(0.15) : Color.primary.opacity(0.04))
                        )
                }
                .buttonStyle(.plain)
                .help("Listen to yourself (1s delay)")

                HStack(spacing: 6) {
                    VerticalVolumeSlider(
                        audioEngine: appState.audioEngine,
                        selectedDeviceUID: appState.activeInputDeviceUID
                    )
                    .frame(width: 8)
                    VerticalLevelMeter(audioEngine: appState.audioEngine)
                }
            }
            .frame(width: 60)
            .padding(.top, 40)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onAppear {
            appState.audioSettingsOpen = true
            if !appState.audioEngine.isWarm {
                beeLog("AUDIO SETTINGS: warming up audio engine")
                try? appState.audioEngine.warmUp()
            }
        }
        .onDisappear {
            appState.audioSettingsOpen = false
            if appState.echoActive {
                appState.audioEngine.stopEcho()
                appState.echoActive = false
            }
            // Grace period before cooling down
            DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak appState] in
                guard let appState else { return }
                guard !appState.audioSettingsOpen,
                      !appState.menuBarPanelOpen,
                      !appState.hotkeyState.isRecording,
                      !appState.activeInputDeviceKeepWarm else { return }
                appState.audioEngine.coolDown()
            }
        }
    }

    private var allDevices: [(device: AppState.InputDeviceInfo, isOnline: Bool)] {
        appState.allDevicesForSettings()
    }

    // Reuse from AdvancedSettingsView
}

private struct AudioSettingsDeviceRow: View {
    let device: AppState.InputDeviceInfo
    let rank: Int
    let isActive: Bool
    let isOnline: Bool
    let isWarm: Bool
    let isHidden: Bool
    let onSelect: (() -> Void)?
    let onToggleWarm: () -> Void
    let onToggleHidden: () -> Void

    var body: some View {
        HStack(spacing: 8) {
            // Priority rank
            Text("\(rank)")
                .font(.system(size: 11, weight: .bold, design: .rounded))
                .foregroundStyle(isActive ? .orange : .secondary)
                .frame(width: 18)

            // Device icon
            DeviceIcon(device: device)
                .font(.title3)
                .foregroundStyle(isActive ? .orange : isOnline ? .secondary : .secondary.opacity(0.3))
                .frame(width: 24)

            // Name + subtitle
            VStack(alignment: .leading, spacing: 1) {
                HStack(spacing: 6) {
                    Text(device.name)
                        .fontWeight(isActive ? .medium : .regular)
                        .foregroundStyle(isOnline ? .primary : .secondary)
                    if isActive {
                        Image(systemName: "checkmark")
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.orange)
                    }
                    if !isOnline {
                        Text("Offline")
                            .font(.system(size: 9, weight: .medium))
                            .foregroundStyle(.secondary.opacity(0.6))
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(
                                RoundedRectangle(cornerRadius: 3, style: .continuous)
                                    .fill(Color.primary.opacity(0.05))
                            )
                    }
                }
                if let subtitle = device.subtitle {
                    Text(subtitle)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }

            Spacer()

            // Icon buttons
            HStack(spacing: 10) {
                Button { onToggleWarm() } label: {
                    Image(systemName: isWarm ? "flame.fill" : "flame")
                        .foregroundStyle(isWarm ? .orange : .secondary.opacity(0.4))
                }
                .buttonStyle(.plain)
                .help("Keep microphone active between sessions")

                Button { onToggleHidden() } label: {
                    Image(systemName: isHidden ? "eye.slash.fill" : "eye")
                        .foregroundStyle(isHidden ? .orange : .secondary.opacity(0.4))
                }
                .buttonStyle(.plain)
                .help("Hide from menu bar device list")
            }
            .font(.body)
        }
        .padding(.vertical, 2)
        .contentShape(Rectangle())
        .onTapGesture { onSelect?() }
        .opacity(isOnline ? 1 : 0.7)
    }
}

// MARK: - Transcription Settings

private struct TranscriptionSettingsView: View {
    @Bindable var appState: AppState
    @State private var tryMeText = ""
    @FocusState private var editorFocused: Bool
    @State private var pipelineExpanded = false
    @State private var mockCPU: Double = 18
    @State private var mockGPU: Double = 61
    @State private var mockMemMB: Double = 387
    @State private var mockRamMB: Double = 512
    @State private var statsTimer: Timer? = nil

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if let errorMsg = appState.modelStatus.errorMessage {
                    HStack(spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.red)
                            .font(.title3)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Model failed to load")
                                .font(.headline)
                            Text(errorMsg)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                        }
                        Spacer()
                        Button("Retry") {
                            appState.loadModelAtStartup()
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.orange)
                    }
                    .padding(12)
                    .background(
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(.red.opacity(0.08))
                            .strokeBorder(.red.opacity(0.2), lineWidth: 1)
                    )
                }

                SettingsCard("Try it out") {
                    VStack(spacing: 10) {
                        TextEditor(text: $tryMeText)
                            .font(.body)
                            .frame(height: 52)
                            .scrollContentBackground(.hidden)
                            .focused($editorFocused)
                            .overlay(alignment: .topLeading) {
                                if tryMeText.isEmpty && !editorFocused {
                                    Text("Dictate here to try your settings…")
                                        .foregroundStyle(.tertiary)
                                        .allowsHitTesting(false)
                                        .padding(.top, 2)
                                        .padding(.leading, 4)
                                }
                            }

                        HStack(spacing: 12) {
                            CombinedStatWidget(
                                label1: "CPU", value1: mockCPU, max1: 100, unit1: "%",
                                color1: statColor(mockCPU, hi: 60, crit: 85),
                                label2: "RAM", value2: mockRamMB, max2: 16384, unit2: "MB",
                                color2: statColor(mockRamMB, hi: 4096, crit: 10240)
                            )
                            CombinedStatWidget(
                                label1: "GPU", value1: mockGPU, max1: 100, unit1: "%",
                                color1: statColor(mockGPU, hi: 60, crit: 85),
                                label2: "VRAM", value2: mockMemMB, max2: 16384, unit2: "MB",
                                color2: statColor(mockMemMB, hi: 8192, crit: 14336)
                            )
                        }
                        .onAppear {
                            updateStats()
                            statsTimer = Timer.scheduledTimer(withTimeInterval: 0.4, repeats: true) { _ in
                                updateStats()
                            }
                        }
                        .onDisappear {
                            statsTimer?.invalidate()
                            statsTimer = nil
                        }
                    }
                }

                SettingsCard("Parameters") {
                    VStack(spacing: 10) {
                        ParamSlider(
                            label: "Chunk size",
                            value: Binding(
                                get: { Double(appState.chunkSizeSec) },
                                set: { appState.chunkSizeSec = Float($0) }
                            ),
                            range: 0.1...2.0, step: 0.05
                        ) { v in
                            let ms = Int(v * 1000)
                            return ms < 1000 ? "\(ms)ms" : String(format: "%.1fs", v)
                        }

                        ParamSlider(
                            label: "Commit tokens",
                            value: Binding(
                                get: { Double(appState.commitTokenCount) },
                                set: { appState.commitTokenCount = UInt32($0) }
                            ),
                            range: 4...48, step: 1
                        ) { "\(Int($0))" }

                        ParamSlider(
                            label: "Rollback tokens",
                            value: Binding(
                                get: { Double(appState.rollbackTokenNum) },
                                set: { appState.rollbackTokenNum = UInt32($0) }
                            ),
                            range: 1...16, step: 1
                        ) { "\(Int($0))" }

                        ParamSlider(
                            label: "Morph speed",
                            value: Binding(
                                get: { Double(appState.animationMorphSpeed) },
                                set: { appState.animationMorphSpeed = Float($0) }
                            ),
                            range: 0...3, step: 0.05
                        ) { v in
                            v < 0.01 ? "Off" : String(format: "%.2g×", v)
                        }

                        ParamSlider(
                            label: "Append speed",
                            value: Binding(
                                get: { Double(appState.animationAppendSpeed) },
                                set: { appState.animationAppendSpeed = Float($0) }
                            ),
                            range: 0...3, step: 0.05
                        ) { v in
                            v < 0.01 ? "Off" : String(format: "%.2g×", v)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                Divider()

                DisclosureGroup("Pipeline", isExpanded: $pipelineExpanded) {
                    VStack(spacing: 0) {
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
                                        Text(AppState.defaultModel.displayName)
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
                        .padding(.top, 8)

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
                }
                .font(.subheadline)
                .foregroundStyle(.secondary)
            }
            .frame(maxWidth: 600, alignment: .leading)
            .padding(24)
        }
    }

    private func updateStats() {
        let s = appState.transcriptionService.getStats()
        withAnimation(.easeInOut(duration: 0.4)) {
            mockCPU = Double(s.cpu_percent)
            mockGPU = Double(s.gpu_percent)
            mockMemMB = Double(s.vram_used_mb)
            mockRamMB = Double(s.ram_used_mb)
        }
    }

    private func statColor(_ val: Double, hi: Double, crit: Double) -> Color {
        val >= crit ? .red : val >= hi ? .orange : .green
    }

    private var modelColor: Color {
        switch appState.modelStatus {
        case .loaded: .green
        case .loading, .downloading: .orange
        case .notLoaded: .gray
        case .error: .red
        }
    }

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: "not loaded"
        case .downloading(let p): "downloading \(Int(p * 100))%"
        case .loading: "loading..."
        case .loaded: AppState.defaultModel.displayName
        case .error(let e): "error: \(e.prefix(30))"
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

private struct GeneralSettingsView: View {
    @Bindable var appState: AppState

    @State private var showDiagSheet = false
    @State private var diagOutput = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                SettingsCard("Behavior") {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle("Sound effects", isOn: Binding(
                            get: { appState.soundEffectsEnabled },
                            set: { appState.soundEffectsEnabled = $0 }
                        ))

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

}

private struct CombinedStatWidget: View {
    let label1: String; let value1: Double; let max1: Double; let unit1: String; let color1: Color
    let label2: String; let value2: Double; let max2: Double; let unit2: String; let color2: Color

    var body: some View {
        VStack(spacing: 6) {
            MockStatRow(label: label1, value: value1, max: max1, unit: unit1, color: color1)
            MockStatRow(label: label2, value: value2, max: max2, unit: unit2, color: color2)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Color.primary.opacity(0.04))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .frame(maxWidth: .infinity)
    }
}

private struct MockStatRow: View {
    let label: String
    let value: Double
    let max: Double
    let unit: String
    let color: Color

    var body: some View {
        VStack(spacing: 3) {
            HStack(spacing: 4) {
                Text(label)
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(value))")
                    .font(.system(size: 11, weight: .medium).monospacedDigit())
                Text(unit)
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color.primary.opacity(0.08))
                    RoundedRectangle(cornerRadius: 2)
                        .fill(color.opacity(0.8))
                        .frame(width: geo.size.width * min(1, value / max))
                }
            }
            .frame(height: 4)
        }
    }
}

private struct ParamSlider: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    let format: (Double) -> String

    init(label: String, value: Binding<Double>, range: ClosedRange<Double>, step: Double, format: @escaping (Double) -> String) {
        self.label = label
        self._value = value
        self.range = range
        self.step = step
        self.format = format
    }

    var body: some View {
        HStack(spacing: 10) {
            Text(label)
                .foregroundStyle(.secondary)
                .frame(width: 140, alignment: .leading)
            Slider(value: $value, in: range, step: step)
            Text(format(value))
                .monospacedDigit()
                .frame(width: 52, alignment: .trailing)
        }
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

private struct DeviceIcon: View {
    let device: AppState.InputDeviceInfo
    var body: some View {
        if let customSymbol = device.customSymbolName {
            Image(customSymbol)
                .symbolRenderingMode(.hierarchical)
        } else if let sfName = device.iconName {
            Image(systemName: sfName)
        }
    }
}

private struct VerticalLevelMeter: View {
    let audioEngine: AudioEngine
    @State private var level: Float = 0
    @State private var levelDb: Float = -60
    @State private var timer: Timer?

    // dB tick marks and target range
    private static let ticks: [Float] = [0, -6, -12, -18, -24, -30, -40, -50, -60]
    private static let targetHigh: Float = -6   // top of ideal range
    private static let targetLow: Float = -24   // bottom of ideal range
    private static let floor: Float = -60

    private func dbToFraction(_ db: Float) -> CGFloat {
        CGFloat(max(0, min(1, (db - Self.floor) / -Self.floor)))
    }

    var body: some View {
        HStack(alignment: .center, spacing: 3) {
            // Tick labels
            GeometryReader { geo in
                ForEach(Self.ticks, id: \.self) { db in
                    let frac = dbToFraction(db)
                    let y = geo.size.height * (1 - frac)
                    Text(db == 0 ? " 0" : "\(Int(db))")
                        .font(.system(size: 7, weight: .medium, design: .monospaced))
                        .foregroundStyle(
                            db >= Self.targetHigh ? Color.orange :
                            db >= Self.targetLow ? Color.secondary :
                            Color.secondary.opacity(0.4)
                        )
                        .position(x: geo.size.width / 2, y: y)
                }
            }
            .frame(width: 20)

            // Meter bar with target range overlay
            GeometryReader { geo in
                ZStack(alignment: .bottom) {
                    // Background
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(Color.primary.opacity(0.06))

                    // Target range indicator
                    let targetTop = geo.size.height * (1 - dbToFraction(Self.targetHigh))
                    let targetBottom = geo.size.height * (1 - dbToFraction(Self.targetLow))
                    RoundedRectangle(cornerRadius: 1, style: .continuous)
                        .fill(Color.green.opacity(0.1))
                        .frame(height: targetBottom - targetTop)
                        .offset(y: -(geo.size.height - targetBottom))

                    // Level fill
                    let fillHeight = geo.size.height * CGFloat(min(1, level))
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(meterColor)
                        .frame(height: fillHeight)

                    // Tick lines
                    ForEach(Self.ticks, id: \.self) { db in
                        let frac = dbToFraction(db)
                        let y = geo.size.height * (1 - frac)
                        Rectangle()
                            .fill(Color.primary.opacity(0.15))
                            .frame(height: 0.5)
                            .offset(y: y - geo.size.height / 2)
                    }
                }
            }
            .frame(width: 6)
        }
        .onAppear {
            timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { _ in
                level = audioEngine.currentLevel
                levelDb = audioEngine.currentLevelDb
            }
        }
        .onDisappear {
            timer?.invalidate()
        }
    }

    private var meterColor: Color {
        if levelDb > -3 { return .red }
        if levelDb > Self.targetHigh { return .orange }
        return .green
    }
}

private struct VerticalVolumeSlider: View {
    let audioEngine: AudioEngine
    let selectedDeviceUID: String?
    @State private var volume: Float = 1.0
    @State private var isSupported = false

    var body: some View {
        GeometryReader { geo in
            if isSupported {
                ZStack(alignment: .bottom) {
                    // Track
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(Color.primary.opacity(0.06))

                    // Fill
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .fill(Color.accentColor.opacity(0.6))
                        .frame(height: geo.size.height * CGFloat(volume))
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let fraction = 1 - Float(value.location.y / geo.size.height)
                            let clamped = max(0, min(1, fraction))
                            volume = clamped
                            if let deviceID = audioEngine.currentDeviceID {
                                AudioEngine.setInputVolume(deviceID: deviceID, volume: clamped)
                            }
                        }
                )
            } else {
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(Color.primary.opacity(0.03))
            }
        }
        .task(id: selectedDeviceUID) { refreshVolume() }
    }

    func refreshVolume() {
        guard let deviceID = audioEngine.currentDeviceID else {
            beeLog("MENUBAR: volume: no current device ID")
            isSupported = false
            return
        }
        if let vol = AudioEngine.getInputVolume(deviceID: deviceID) {
            beeLog("MENUBAR: volume: device \(deviceID) supports input volume = \(vol)")
            volume = vol
            isSupported = true
        } else {
            beeLog("MENUBAR: volume: device \(deviceID) does NOT support input volume")
            isSupported = false
        }
    }
}

/// A dropdown button that looks like a Picker but shows rich device rows in a popover.
private struct DeviceDropdown: View {
    @Bindable var appState: AppState
    @State private var isOpen = false

    var body: some View {
        let activeDevice = appState.availableInputDevices.first { $0.uid == appState.activeInputDeviceUID }

        Button {
            isOpen.toggle()
        } label: {
            HStack(spacing: 8) {
                if let device = activeDevice {
                    DeviceIcon(device: device)
                        .foregroundStyle(.orange)
                        .font(.title3)
                        .frame(width: 20, height: 20)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(device.name)
                        if let sub = device.subtitle {
                            Text(sub)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                    }
                } else {
                    Text("No device")
                        .foregroundStyle(.secondary)
                }
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .fill(Color.primary.opacity(0.04))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .strokeBorder(Color.primary.opacity(0.1), lineWidth: 0.5)
            )
        }
        .buttonStyle(.plain)
        .popover(isPresented: $isOpen, arrowEdge: .bottom) {
            VStack(spacing: 2) {
                ForEach(appState.availableInputDevices, id: \.uid) { device in
                    Button {
                        appState.selectInputDevice(uid: device.uid)
                        isOpen = false
                    } label: {
                        HStack(spacing: 8) {
                            DeviceIcon(device: device)
                                .font(.title3)
                                .foregroundStyle(device.uid == appState.activeInputDeviceUID ? .orange : .secondary)
                                .frame(width: 24)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(device.name)
                                    .fontWeight(device.uid == appState.activeInputDeviceUID ? .medium : .regular)
                                if let sub = device.subtitle {
                                    Text(sub)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            if device.uid == appState.activeInputDeviceUID {
                                Image(systemName: "checkmark")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.orange)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(6)
            .frame(minWidth: 280)
        }
    }
}

private struct InputDeviceList: View {
    @Bindable var appState: AppState

    private var visibleDevices: [AppState.InputDeviceInfo] {
        appState.availableInputDevices.filter { !appState.hiddenDeviceUIDs.contains($0.uid) }
    }

    var body: some View {
        if visibleDevices.isEmpty {
            Text("No input devices")
                .foregroundStyle(.secondary)
        } else {
            VStack(spacing: 2) {
                ForEach(visibleDevices, id: \.uid) { device in
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
                DeviceIcon(device: device)
                    .font(.title3)
                    .foregroundStyle(isActive ? .orange : .secondary)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 1) {
                    Text(device.name)
                        .fontWeight(isActive ? .medium : .regular)
                    if let subtitle = device.subtitle {
                        Text(subtitle)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
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
    var appIcon: NSImage? = nil

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
                if let appIcon {
                    Image(nsImage: appIcon)
                        .resizable()
                        .frame(width: 16, height: 16)
                        .clipShape(RoundedRectangle(cornerRadius: 3, style: .continuous))
                }
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
                DeviceIcon(device: device)
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
