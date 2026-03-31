import AppKit
import SwiftUI

/// A floating, always-on-top debug panel showing state machine info.
@MainActor
final class DebugPanel: NSObject, NSWindowDelegate {
    static let shared = DebugPanel()

    private var panel: NSPanel?
    private weak var appState: AppState?

    var isVisible: Bool { panel != nil }

    func show(appState: AppState) {
        guard panel == nil else { return }
        self.appState = appState

        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 260, height: 200),
            styleMask: [.titled, .closable, .resizable, .utilityWindow, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.title = "Bee Debug"
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        panel.isMovableByWindowBackground = true
        panel.hasShadow = true
        panel.backgroundColor = .clear
        panel.titlebarAppearsTransparent = true
        panel.delegate = self

        let hostingView = NSHostingView(rootView: DebugOverlay(appState: appState))
        panel.contentView = hostingView

        // Position in top-right corner
        if let screen = NSScreen.main {
            let screenFrame = screen.visibleFrame
            let x = screenFrame.maxX - 270
            let y = screenFrame.maxY - 210
            panel.setFrameOrigin(NSPoint(x: x, y: y))
        }

        panel.orderFront(nil)
        self.panel = panel
    }

    func hide() {
        panel?.close()
        panel = nil
    }

    func toggle(appState: AppState) {
        if isVisible {
            hide()
        } else {
            show(appState: appState)
        }
    }

    nonisolated func windowWillClose(_ notification: Notification) {
        MainActor.assumeIsolated {
            panel = nil
            appState?.debugEnabled = false
        }
    }
}
