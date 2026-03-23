import AppKit
import SwiftUI

/// Manages the lifecycle of floating recording indicator panels — one per screen.
@MainActor
final class OverlayManager {
    private var panels: [FloatingPanel<AnyView>] = []
    private var isPresented = false
    private var currentAppState: AppState?
    private let overlaySize = CGSize(width: 700, height: 300)
    private var dismissTask: Task<Void, Never>?

    func show(appState: AppState) {
        // Cancel any pending dismiss from a previous recording.
        dismissTask?.cancel()
        dismissTask = nil

        currentAppState = appState
        appState.overlayDismiss = .none

        // Close stale panels.
        if !panels.isEmpty {
            for panel in panels { panel.close() }
            panels.removeAll()
        }

        // Create one panel per screen.
        for screen in NSScreen.screens {
            let binding = Binding<Bool>(
                get: { [weak self] in
                    self?.isPresented ?? false
                },
                set: { [weak self] newValue in
                    guard let self else { return }
                    self.isPresented = newValue
                    if !newValue {
                        self.panels.removeAll()
                    }
                }
            )

            let contentRect = NSRect(origin: .zero, size: overlaySize)
            let panel = FloatingPanel(
                view: {
                    AnyView(
                        RecordingOverlayView(appState: appState)
                    )
                },
                contentRect: contentRect,
                isPresented: binding
            )
            panel.positionTopCenter(on: screen)
            panel.alphaValue = 1.0
            panel.orderFrontRegardless()
            panels.append(panel)
        }

        isPresented = true
    }

    func hide() {
        dismissTask?.cancel()
        dismissTask = nil
        isPresented = false
        for panel in panels { panel.close() }
        panels.removeAll()
        currentAppState?.overlayDismiss = .none
        currentAppState = nil
    }

    /// Trigger dismiss animation and return immediately (non-blocking).
    func hideWithResult(_ result: OverlayResult) {
        guard result != .none, let appState = currentAppState else {
            hide()
            return
        }

        // Tell the SwiftUI views to animate the dismiss (all panels share appState).
        appState.overlayDismiss = result

        // Schedule cleanup after the animation plays.
        dismissTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(280))
            guard !Task.isCancelled else { return }
            self.hide()
        }
    }
}
