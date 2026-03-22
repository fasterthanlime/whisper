import AppKit
import SwiftUI

/// Manages the lifecycle of the floating recording indicator panel.
@MainActor
final class OverlayManager {
    private var panel: FloatingPanel<AnyView>?
    private var isPresented = false
    private var currentResult: OverlayResult = .none
    private var currentAppState: AppState?
    private let overlaySize = CGSize(width: 550, height: 200)

    func show(appState: AppState) {
        currentAppState = appState
        currentResult = .none

        if panel == nil {
            let binding = Binding<Bool>(
                get: { [weak self] in
                    self?.isPresented ?? false
                },
                set: { [weak self] newValue in
                    guard let self else { return }
                    self.isPresented = newValue
                    if !newValue {
                        self.panel = nil
                    }
                }
            )

            let contentRect = NSRect(origin: .zero, size: overlaySize)
            let newPanel = FloatingPanel(
                view: { [weak self] in
                    AnyView(
                        RecordingOverlayView(
                            appState: appState,
                            result: self?.currentResult ?? .none
                        )
                    )
                },
                contentRect: contentRect,
                isPresented: binding
            )
            newPanel.positionTopCenter()
            panel = newPanel
        } else {
            updateView()
        }

        isPresented = true
        panel?.orderFrontRegardless()
    }

    private func updateView() {
        guard let appState = currentAppState else { return }
        panel?.updateView { [weak self] in
            AnyView(
                RecordingOverlayView(
                    appState: appState,
                    result: self?.currentResult ?? .none
                )
            )
        }
    }

    func hide() {
        isPresented = false
        panel?.close()
        panel = nil
        currentAppState = nil
        currentResult = .none
    }

    /// Hide with a success or cancel animation
    func hideWithResult(_ result: OverlayResult) async {
        guard result != .none else {
            hide()
            return
        }

        currentResult = result
        updateView()

        // Wait for animation to play
        try? await Task.sleep(for: .milliseconds(600))

        hide()
    }
}
