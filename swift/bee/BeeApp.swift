import AppKit
import SwiftUI

final class BeeLifecycleDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        BeeInputClient.switchAwayFromBeeInputIfNeeded()
    }
}

@main
struct BeeApp: App {
    @NSApplicationDelegateAdaptor(BeeLifecycleDelegate.self) private var lifecycleDelegate
    @State private var appState: AppState
    @State private var hotkeyMonitor = HotkeyMonitor()

    init() {
        let audioEngine = AudioEngine()
        let transcriptionService = TranscriptionService()
        let inputClient = BeeInputClient()
        let state = AppState(
            audioEngine: audioEngine,
            transcriptionService: transcriptionService,
            inputClient: inputClient
        )
        _appState = State(initialValue: state)

        let monitor = HotkeyMonitor()
        monitor.appState = state
        monitor.start()
        _hotkeyMonitor = State(initialValue: monitor)

        BeeInputClient.ensureIMERegistered()
        state.loadModelAtStartup()
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(appState: appState)
        } label: {
            MenuBarLabelView(appState: appState)
        }
        .menuBarExtraStyle(.window)

        Window("Bee", id: "bee-main") {
            BeeMainWindowView(appState: appState)
        }
        .defaultSize(width: 640, height: 520)

        Settings {
            BeeSettingsView(appState: appState)
        }
    }
}

private struct BeeMainWindowView: View {
    @Bindable var appState: AppState

    var body: some View {
        NavigationStack {
            List {
                Section("Status") {
                    LabeledContent("State", value: statusLabel)
                    LabeledContent("Model", value: modelLabel)
                    LabeledContent("Input", value: appState.activeInputDeviceName ?? "None")
                }

                Section("Recent Transcripts") {
                    if appState.transcriptionHistory.isEmpty {
                        Text("No transcripts yet")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(appState.transcriptionHistory.prefix(20)) { item in
                            Text(item.text)
                                .textSelection(.enabled)
                                .lineLimit(3)
                        }
                    }
                }
            }
            .navigationTitle("Bee")
        }
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

private struct MenuBarLabelView: View {
    @Bindable var appState: AppState

    var body: some View {
        if isActivelyRecording {
            Image(nsImage: recordingMenuBarImage)
        } else {
            Image("MenuBarIcon")
        }
    }

    private var isActivelyRecording: Bool {
        switch appState.uiState {
        case .pushToTalk, .locked, .lockedOptionHeld:
            return true
        case .idle, .pending:
            return false
        }
    }

    private var recordingMenuBarImage: NSImage {
        let fallbackSize = NSSize(width: 18, height: 18)
        guard let base = NSImage(named: "MenuBarIcon") else {
            let image = NSImage(size: fallbackSize)
            image.isTemplate = false
            return image
        }

        let baseSize = base.size == .zero ? fallbackSize : base.size
        let image = NSImage(size: baseSize)
        image.lockFocus()

        let rect = NSRect(origin: .zero, size: baseSize)
        if let cgMask = base.cgImage(
            forProposedRect: nil,
            context: NSGraphicsContext.current,
            hints: nil
        ) {
            let ctx = NSGraphicsContext.current?.cgContext
            ctx?.saveGState()
            ctx?.clip(to: rect, mask: cgMask)
            NSColor.labelColor.setFill()
            rect.fill()
            ctx?.restoreGState()
        } else {
            base.draw(in: rect)
        }

        NSColor.systemRed.setFill()
        let dotDiameter = max(7.0, floor(min(baseSize.width, baseSize.height) * 0.45))
        let dotMargin = 1.0
        let dotRect = NSRect(
            x: baseSize.width - dotDiameter - dotMargin,
            y: dotMargin,
            width: dotDiameter,
            height: dotDiameter
        )
        NSBezierPath(ovalIn: dotRect).fill()

        image.unlockFocus()
        image.isTemplate = false
        return image
    }
}
