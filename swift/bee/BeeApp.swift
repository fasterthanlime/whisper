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

@MainActor
final class StatusBarController: NSObject {
    private let statusItem: NSStatusItem
    private let popover: NSPopover
    private var eventMonitor: Any?
    private weak var appState: AppState?

    init(appState: AppState) {
        self.appState = appState
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        popover = NSPopover()
        popover.contentSize = NSSize(width: 260, height: 10) // height auto-sizes
        popover.behavior = .transient
        popover.animates = true
        popover.contentViewController = NSHostingController(rootView: MenuBarView(appState: appState))

        super.init()

        if let button = statusItem.button {
            button.image = NSImage(named: "MenuBarIcon")
            button.image?.isTemplate = true
            button.action = #selector(handleClick)
            button.target = self
            button.sendAction(on: [.leftMouseDown, .rightMouseDown])
        }

        // Close popover when clicking outside
        eventMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
            self?.closePopover()
        }

        // Update icon state
        NotificationCenter.default.addObserver(
            self, selector: #selector(updateIcon),
            name: AppState.stateChangedNotification, object: nil
        )

        // Handle open-main-window requests
        NotificationCenter.default.addObserver(
            self, selector: #selector(handleOpenMainWindow),
            name: .beeOpenMainWindowRequest, object: nil
        )

        updateIcon()
    }

    @objc private func handleClick() {
        if popover.isShown {
            closePopover()
        } else {
            showPopover()
        }
    }

    private func showPopover() {
        guard let button = statusItem.button else { return }
        popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        popover.contentViewController?.view.window?.makeKey()
    }

    private func closePopover() {
        popover.performClose(nil)
    }

    @objc func updateIcon() {
        guard let appState, let button = statusItem.button else { return }
        let isReady = appState.imeReady && appState.modelStatus == .loaded
        button.alphaValue = isReady ? 1.0 : 0.4
        button.image = NSImage(named: "MenuBarIcon")
        button.image?.isTemplate = true
    }

    @objc private func handleOpenMainWindow() {
        closePopover()
        NSApp.activate(ignoringOtherApps: true)
        NSApp.sendAction(Selector(("showSettingsWindow:")), to: nil, from: nil)
        DispatchQueue.main.async {
            let normalWindow = NSApp.windows.first { window in
                !(window is NSPanel) && window.isVisible
            }
            normalWindow?.makeKeyAndOrderFront(nil)
        }
    }
}

@main
struct BeeApp: App {
    @NSApplicationDelegateAdaptor(BeeLifecycleDelegate.self) private var lifecycleDelegate
    @State private var appState: AppState
    @State private var hotkeyMonitor = HotkeyMonitor()
    @State private var statusBar: StatusBarController?

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

        _statusBar = State(initialValue: StatusBarController(appState: state))
    }

    var body: some Scene {
        Settings {
            BeeSettingsView(appState: appState)
        }
        .windowStyle(.hiddenTitleBar)
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
