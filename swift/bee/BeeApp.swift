import AppKit
import SwiftUI

extension Notification.Name {
    static let beeOpenMainWindowRequest = Notification.Name("fasterthanlime.bee.openMainWindowRequest")
}

final class BeeLifecycleDelegate: NSObject, NSApplicationDelegate {
    func applicationWillTerminate(_ notification: Notification) {
        BeeInputClient.switchAwayFromBeeInputIfNeeded()
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        NotificationCenter.default.post(name: .beeOpenMainWindowRequest, object: nil)
        return false
    }

    func applicationDidBecomeActive(_ notification: Notification) {
        let hasVisibleNormalWindow = NSApp.windows.contains { window in
            window.isVisible && !(window is NSPanel)
        }
        guard !hasVisibleNormalWindow else { return }
        NotificationCenter.default.post(name: .beeOpenMainWindowRequest, object: nil)
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
        .menuBarExtraStyle(.menu)

        Settings {
            BeeSettingsView(appState: appState)
        }
        .commands {
            CommandGroup(replacing: .appTermination) {
                Button("Close Bee Window") {
                    if let keyWindow = NSApp.keyWindow, !(keyWindow is NSPanel) {
                        keyWindow.performClose(nil)
                        return
                    }

                    let normalWindow = NSApp.orderedWindows.first { window in
                        window.isVisible && !(window is NSPanel)
                    }
                    normalWindow?.performClose(nil)
                }
                .keyboardShortcut("q")
            }
        }
    }
}

private struct MenuBarLabelView: View {
    @Bindable var appState: AppState
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        Group {
            if isActivelyRecording {
                Image(nsImage: recordingMenuBarImage)
            } else {
                Image("MenuBarIcon")
            }
        }
        .onReceive(NotificationCenter.default.publisher(for: .beeOpenMainWindowRequest)) { _ in
            NSApp.activate(ignoringOtherApps: true)
            openSettings()
            DispatchQueue.main.async {
                let normalWindow = NSApp.orderedWindows.first { window in
                    !(window is NSPanel)
                }
                normalWindow?.makeKeyAndOrderFront(nil)
            }
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
