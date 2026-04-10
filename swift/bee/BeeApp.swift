import AppKit
import SwiftUI

extension Notification.Name {
    static let beeOpenMainWindowRequest = Notification.Name(
        "fasterthanlime.bee.openMainWindowRequest")
}

/// Stores the SwiftUI openSettings action so it can be called from outside SwiftUI.
@MainActor
enum SettingsOpener {
    static var action: (() -> Void)?
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

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        false
    }

    func applicationWillTerminate(_ notification: Notification) {
    }

    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool)
        -> Bool
    {
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

/// Borderless panel that closes when it loses focus — replaces NSPopover
/// for reliable positioning when content resizes.
private class MenuBarPanel: NSPanel {
    override var canBecomeKey: Bool { true }

    init(contentView: NSView) {
        super.init(
            contentRect: .zero,
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: true
        )
        self.contentView = contentView
        isOpaque = false
        backgroundColor = .clear
        level = .popUpMenu
        isMovableByWindowBackground = false
        hidesOnDeactivate = false
    }
}

@MainActor
final class StatusBarController: NSObject {
    private let statusItem: NSStatusItem
    private var panel: MenuBarPanel?
    private var globalMonitor: Any?
    private var localMonitor: Any?
    private weak var appState: AppState?
    private var animationProgress: CGFloat = 0  // 0 = idle, 1 = recording
    private var animationTimer: Timer?
    private var frameObserver: NSObjectProtocol?

    private static let itemWidth: CGFloat = 26
    private static let panelWidth: CGFloat = 420

    init(appState: AppState) {
        self.appState = appState
        statusItem = NSStatusBar.system.statusItem(withLength: Self.itemWidth)

        super.init()

        if let button = statusItem.button {
            button.image = NSImage(named: "MenuBarIcon")
            button.image?.isTemplate = true
            button.imagePosition = .imageOnly
            button.contentTintColor = nil
            button.action = #selector(handleClick)
            button.target = self
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
        if panel?.isVisible == true {
            beeLog("MENUBAR: click → closing panel")
            closePanel()
        } else {
            beeLog("MENUBAR: click → opening panel")
            showPanel()
        }
    }

    private func showPanel() {
        guard let appState, let button = statusItem.button, let buttonWindow = button.window else {
            beeLog(
                "MENUBAR: showPanel guard failed (appState=\(appState != nil), button=\(statusItem.button != nil))"
            )
            return
        }

        let hostingView = NSHostingView(rootView: MenuBarView(appState: appState))
        hostingView.wantsLayer = true
        hostingView.layer?.cornerRadius = 10
        hostingView.layer?.masksToBounds = true

        let panel = MenuBarPanel(contentView: hostingView)
        self.panel = panel

        // Position below the button, right-aligned
        positionPanel(panel, relativeTo: button, in: buttonWindow)

        beeLog("MENUBAR: panel created, making visible")
        panel.makeKeyAndOrderFront(nil)

        // Reposition when content resizes
        hostingView.postsFrameChangedNotifications = true
        frameObserver = NotificationCenter.default.addObserver(
            forName: NSView.frameDidChangeNotification,
            object: hostingView,
            queue: .main
        ) { [weak self, weak panel, weak button, weak buttonWindow] _ in
            guard let self, let panel, let button, let buttonWindow else { return }
            self.positionPanel(panel, relativeTo: button, in: buttonWindow)
        }

        // Close when clicking outside
        globalMonitor = NSEvent.addGlobalMonitorForEvents(matching: [
            .leftMouseDown, .rightMouseDown,
        ]) { [weak self] _ in
            self?.closePanel()
        }

        // Close on Escape
        localMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            if event.keyCode == 53 {  // Escape
                self?.closePanel()
                return nil
            }
            return event
        }
    }

    private func positionPanel(
        _ panel: NSPanel, relativeTo button: NSStatusBarButton, in buttonWindow: NSWindow
    ) {
        let buttonRect = button.convert(button.bounds, to: nil)
        let screenRect = buttonWindow.convertToScreen(buttonRect)
        let contentSize =
            panel.contentView?.fittingSize ?? NSSize(width: Self.panelWidth, height: 200)

        let x = screenRect.midX - contentSize.width / 2
        let y = screenRect.minY - contentSize.height - 4  // 4pt gap

        panel.setContentSize(contentSize)
        panel.setFrameOrigin(NSPoint(x: x, y: y))
    }

    private func closePanel() {
        panel?.orderOut(nil)
        panel = nil

        if let obs = frameObserver {
            NotificationCenter.default.removeObserver(obs)
            frameObserver = nil
        }
        if let monitor = globalMonitor {
            NSEvent.removeMonitor(monitor)
            globalMonitor = nil
        }
        if let monitor = localMonitor {
            NSEvent.removeMonitor(monitor)
            localMonitor = nil
        }
    }

    var isPanelVisible: Bool {
        panel?.isVisible == true
    }

    @objc func updateIcon() {
        guard let appState, let button = statusItem.button else { return }
        let isReady = appState.imeReady && appState.modelStatus == .loaded
        let hasError = appState.modelStatus.hasError
        let isActive: Bool =
            switch appState.hotkeyState {
            case .held, .released, .pushToTalk, .locked, .lockedOptionHeld: true
            case .idle: false
            }

        let isLoadingOrDownloading: Bool =
            switch appState.modelStatus {
            case .loading, .downloading: true
            default: false
            }
        button.alphaValue = isReady ? 1.0 : (hasError || isLoadingOrDownloading ? 1.0 : 0.4)

        let target: CGFloat = isActive ? 1 : 0
        guard abs(animationProgress - target) > 0.01 else { return }

        animationTimer?.invalidate()
        let startProgress = animationProgress
        let startTime = CACurrentMediaTime()
        let duration: CFTimeInterval = 0.15

        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60, repeats: true) {
            [weak self] timer in
            guard let self else {
                timer.invalidate()
                return
            }
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
            let baseIcon = NSImage(named: "MenuBarIcon")
        else { return }

        let hasError = appState?.modelStatus.hasError ?? false
        let isLoadingOrDownloading: Bool =
            switch appState?.modelStatus {
            case .loading, .downloading: true
            default: false
            }
        let p = animationProgress
        if p < 0.01 && !hasError && !isLoadingOrDownloading {
            baseIcon.isTemplate = true
            button.image = baseIcon
            return
        }

        if p < 0.01 && isLoadingOrDownloading {
            // Loading/downloading: show bee + orange dot
            let iconSize = baseIcon.size
            let canvasSize = NSSize(width: Self.itemWidth, height: iconSize.height)
            let dotSize: CGFloat = 6
            let isDark =
                button.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            let iconColor: NSColor = isDark ? .white : .black
            let tinted = baseIcon.copy() as! NSImage
            tinted.lockFocus()
            iconColor.set()
            NSRect(origin: .zero, size: tinted.size).fill(using: .sourceAtop)
            tinted.unlockFocus()
            let composed = NSImage(size: canvasSize, flipped: false) { rect in
                let beeRect = NSRect(
                    x: rect.midX - iconSize.width / 2 - 3,
                    y: (rect.height - iconSize.height) / 2,
                    width: iconSize.width,
                    height: iconSize.height
                )
                tinted.draw(in: beeRect)
                NSColor.systemOrange.setFill()
                let dotRect = NSRect(
                    x: beeRect.maxX - 1,
                    y: beeRect.maxY - dotSize - 1,
                    width: dotSize,
                    height: dotSize
                )
                NSBezierPath(ovalIn: dotRect).fill()
                return true
            }
            composed.isTemplate = false
            button.image = composed
            return
        }

        if p < 0.01 && hasError {
            // Idle with error: show bee + red dot
            let iconSize = baseIcon.size
            let canvasSize = NSSize(width: Self.itemWidth, height: iconSize.height)
            let dotSize: CGFloat = 6
            let isDark =
                button.effectiveAppearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            let iconColor: NSColor = isDark ? .white : .black
            let tinted = baseIcon.copy() as! NSImage
            tinted.lockFocus()
            iconColor.set()
            NSRect(origin: .zero, size: tinted.size).fill(using: .sourceAtop)
            tinted.unlockFocus()
            let composed = NSImage(size: canvasSize, flipped: false) { rect in
                let beeRect = NSRect(
                    x: rect.midX - iconSize.width / 2 - 3,
                    y: (rect.height - iconSize.height) / 2,
                    width: iconSize.width,
                    height: iconSize.height
                )
                tinted.draw(in: beeRect)
                NSColor.systemRed.setFill()
                let dotRect = NSRect(
                    x: beeRect.maxX - 1,
                    y: beeRect.maxY - dotSize - 1,
                    width: dotSize,
                    height: dotSize
                )
                NSBezierPath(ovalIn: dotRect).fill()
                return true
            }
            composed.isTemplate = false
            button.image = composed
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
        closePanel()
        openSettingsWindow()
    }

    func openSettingsWindow() {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        if let action = SettingsOpener.action {
            action()
        }
        DispatchQueue.main.async {
            for window in NSApp.windows where !(window is NSPanel) {
                window.makeKeyAndOrderFront(nil)
                return
            }
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
        let beeEngine = BeeEngine()
        let transcriptionService = TranscriptionService(engine: beeEngine)
        let correctionService = CorrectionService(engine: beeEngine)
        let inputClient = BeeInputClient()
        let state = AppState(
            audioEngine: audioEngine,
            beeEngine: beeEngine,
            transcriptionService: transcriptionService,
            correctionService: correctionService,
            inputClient: inputClient
        )
        _appState = State(initialValue: state)

        let monitor = HotkeyMonitor()
        monitor.appState = state
        monitor.start()
        _hotkeyMonitor = State(initialValue: monitor)

        // Register custom fonts
        if let fontURL = Bundle.main.url(forResource: "bee-symbols", withExtension: "ttf") {
            CTFontManagerRegisterFontsForURL(fontURL as CFURL, .process, nil)
        }

        Task { await BeeInputClient.ensureIMERegistered() }
        state.loadModelAtStartup()
        state.warmUpIME()
        if state.debugEnabled {
            DebugPanel.shared.show(appState: state)
        }

        _statusBar = State(initialValue: StatusBarController(appState: state))
    }

    @Environment(\.openWindow) private var openWindow

    var body: some Scene {
        Window("bee Settings", id: "bee-settings") {
            BeeSettingsView(appState: appState)
                .onAppear {
                    SettingsOpener.action = { [openWindow] in
                        openWindow(id: "bee-settings")
                    }
                }
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 780, height: 520)
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
