import AppKit
import SwiftUI

extension Notification.Name {
    static let beeOpenMainWindowRequest = Notification.Name("fasterthanlime.bee.openMainWindowRequest")
}

final class BeeLifecycleDelegate: NSObject, NSApplicationDelegate {
    private var windowObservers: [NSObjectProtocol] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        windowObservers.append(
            NotificationCenter.default.addObserver(
                forName: NSWindow.didBecomeKeyNotification,
                object: nil,
                queue: .main
            ) { note in
                guard let window = note.object as? NSWindow, !(window is NSPanel) else { return }
                NSApp.setActivationPolicy(.regular)
            }
        )

        windowObservers.append(
            NotificationCenter.default.addObserver(
                forName: NSWindow.willCloseNotification,
                object: nil,
                queue: .main
            ) { note in
                guard let window = note.object as? NSWindow, !(window is NSPanel) else { return }
                DispatchQueue.main.async {
                    let hasOtherNormalWindows = NSApp.windows.contains { w in
                        w !== window && w.isVisible && !(w is NSPanel)
                    }
                    if !hasOtherNormalWindows {
                        NSApp.setActivationPolicy(.accessory)
                    }
                }
            }
        )
    }

    func applicationWillTerminate(_ notification: Notification) {
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
        state.warmUpIME()
        if state.debugEnabled {
            DebugPanel.shared.show(appState: state)
        }
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(appState: appState)
        } label: {
            MenuBarLabelView(appState: appState)
        }
        .menuBarExtraStyle(.window)

        Settings {
            BeeSettingsView(appState: appState)
        }
        .commands {
            CommandGroup(after: .windowArrangement) {
                Button("Toggle Debug Overlay") {
                    appState.debugEnabled.toggle()
                    if appState.debugEnabled {
                        DebugPanel.shared.show(appState: appState)
                    } else {
                        DebugPanel.shared.hide()
                    }
                }
                .keyboardShortcut("d", modifiers: [.command, .shift])
            }
            CommandGroup(replacing: .appTermination) {
                Button("Close bee window") {
                    if let keyWindow = NSApp.keyWindow, !(keyWindow is NSPanel) {
                        keyWindow.performClose(nil)
                        return
                    }
                    let normalWindow = NSApp.orderedWindows.first { window in
                        window.isVisible && !(window is NSPanel)
                    }
                    normalWindow?.performClose(nil)
                }
                .keyboardShortcut("q", modifiers: [.command])
            }
        }
    }
}

private struct MenuBarLabelView: View {
    @Bindable var appState: AppState
    @Environment(\.openSettings) private var openSettings

    var body: some View {
        Image("MenuBarIcon")
            .renderingMode(.template)
            .opacity(isReady ? 1.0 : 0.4)
            .overlay(alignment: .bottomTrailing) {
                if isSessionActive {
                    Circle()
                        .fill(.red)
                        .frame(width: 6, height: 6)
                        .overlay {
                            Circle()
                                .stroke(Color(nsColor: .windowBackgroundColor), lineWidth: 1)
                        }
                        .offset(x: 1, y: 1)
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

    private var isReady: Bool {
        appState.imeReady && appState.modelStatus == .loaded
    }

    private var isSessionActive: Bool {
        switch appState.hotkeyState {
        case .held, .released, .pushToTalk, .locked, .lockedOptionHeld:
            return true
        case .idle:
            return false
        }
    }
}
