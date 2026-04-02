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
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private weak var appState: AppState?
    private var animationProgress: CGFloat = 0  // 0 = idle, 1 = recording
    private var animationTimer: Timer?

    private static let itemWidth: CGFloat = 26

    init(appState: AppState) {
        self.appState = appState
        statusItem = NSStatusBar.system.statusItem(withLength: Self.itemWidth)
        popover = NSPopover()
        popover.contentSize = NSSize(width: 260, height: 10)
        popover.behavior = .transient
        popover.animates = false
        popover.contentViewController = NSHostingController(rootView: MenuBarView(appState: appState))

        super.init()

        if let button = statusItem.button {
            button.image = NSImage(named: "MenuBarIcon")
            button.image?.isTemplate = true
            button.imagePosition = .imageOnly
            button.contentTintColor = nil
        }

        // Intercept mouseDown on the status item button before drag detection
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] event in
            guard let self, let button = self.statusItem.button else { return event }
            guard event.window == button.window else { return event }

            let locationInButton = button.convert(event.locationInWindow, from: nil)
            if button.bounds.contains(locationInButton) {
                if self.popover.isShown {
                    self.closePopover()
                } else {
                    self.showPopover()
                }
                return nil // consume the event
            }
            return event
        }

        // Close popover when clicking outside
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: [.leftMouseDown, .rightMouseDown]) { [weak self] _ in
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
        let isActive: Bool = switch appState.hotkeyState {
        case .held, .released, .pushToTalk, .locked, .lockedOptionHeld: true
        case .idle: false
        }

        button.alphaValue = isReady ? 1.0 : 0.4

        let target: CGFloat = isActive ? 1 : 0
        guard abs(animationProgress - target) > 0.01 else { return }

        animationTimer?.invalidate()
        let startProgress = animationProgress
        let startTime = CACurrentMediaTime()
        let duration: CFTimeInterval = 0.15

        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60, repeats: true) { [weak self] timer in
            guard let self else { timer.invalidate(); return }
            let elapsed = CACurrentMediaTime() - startTime
            let t = min(1, Float(elapsed / duration))
            let smooth = CGFloat(t * t * (3 - 2 * t))
            self.animationProgress = startProgress + (target - startProgress) * smooth

            self.renderIcon()

            if t >= 1 {
                timer.invalidate()
                self.animationProgress = target
                self.renderIcon()
            }
        }
    }

    private func renderIcon() {
        guard let button = statusItem.button,
              let baseIcon = NSImage(named: "MenuBarIcon") else { return }

        let p = animationProgress
        if p < 0.01 {
            baseIcon.isTemplate = true
            button.image = baseIcon
            return
        }

        let iconSize = baseIcon.size
        let canvasSize = NSSize(width: Self.itemWidth, height: iconSize.height)
        let dotSize: CGFloat = 6
        let shrinkFactor: CGFloat = 1.0 - (0.2 * p)  // shrink to 80% at most
        let beeW = iconSize.width * shrinkFactor
        let beeH = iconSize.height * shrinkFactor

        // Detect menu bar appearance for icon color
        let isDark = button.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        let iconColor: NSColor = isDark ? .white : .black

        // Tint the base icon
        let tinted = baseIcon.copy() as! NSImage
        tinted.lockFocus()
        iconColor.set()
        NSRect(origin: .zero, size: tinted.size).fill(using: .sourceAtop)
        tinted.unlockFocus()

        let composed = NSImage(size: canvasSize, flipped: false) { rect in
            // Bee: centered when idle, shifts left when recording
            let beeCenterX = rect.midX - (5 * p)  // shift left up to 5pt
            let beeRect = NSRect(
                x: beeCenterX - beeW / 2,
                y: (rect.height - beeH) / 2,
                width: beeW,
                height: beeH
            )
            tinted.draw(in: beeRect)

            // Orange dot to the right of the bee
            if p > 0.1 {
                let dotOpacity = min(1, (p - 0.1) / 0.5)
                NSColor.systemOrange.withAlphaComponent(CGFloat(dotOpacity)).setFill()
                let dotRect = NSRect(
                    x: beeRect.maxX + 2,
                    y: (rect.height - dotSize) / 2,
                    width: dotSize,
                    height: dotSize
                )
                NSBezierPath(ovalIn: dotRect).fill()
            }
            return true
        }
        composed.isTemplate = false
        button.image = composed
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
