import AppKit
import SwiftUI

private final class TransparentHostingView<Content: View>: NSHostingView<Content> {
    override var isOpaque: Bool { false }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        // Disable clipping so scale animations can extend beyond bounds.
        wantsLayer = true
        layer?.masksToBounds = false
        superview?.wantsLayer = true
        superview?.layer?.masksToBounds = false
    }
}

/// A floating, non-activating panel adapted from a blog-style NSPanel implementation.
final class FloatingPanel<Content: View>: NSPanel {
    @Binding private var isPresented: Bool
    private let hostingView: TransparentHostingView<Content>

    init(
        @ViewBuilder view: () -> Content,
        contentRect: NSRect,
        isPresented: Binding<Bool>
    ) {
        self._isPresented = isPresented
        self.hostingView = TransparentHostingView(rootView: view())

        super.init(
            contentRect: contentRect,
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )

        isFloatingPanel = true
        level = .floating
        animationBehavior = .utilityWindow
        isMovableByWindowBackground = false
        hidesOnDeactivate = false

        isOpaque = false
        backgroundColor = .clear
        collectionBehavior = [.canJoinAllSpaces, .stationary]
        ignoresMouseEvents = true
        hasShadow = false

        contentView = hostingView
    }

    func updateView(@ViewBuilder _ view: () -> Content) {
        hostingView.rootView = view()
    }

    override func resignMain() {
        super.resignMain()
        close()
    }

    override func close() {
        super.close()
        isPresented = false
    }

    override var canBecomeKey: Bool {
        false
    }

    override var canBecomeMain: Bool {
        false
    }

    /// Position the panel at the bottom-center of the main screen.
    func positionBottomCenter() {
        guard let screen = NSScreen.main else { return }
        let screenFrame = screen.visibleFrame
        let x = screenFrame.midX - (frame.width / 2)
        let y = screenFrame.minY + 15
        setFrameOrigin(NSPoint(x: x, y: y))
    }

    /// Position the panel at the top-center of the given screen, below the menu bar.
    func positionTopCenter(on screen: NSScreen? = nil) {
        guard let screen = screen ?? NSScreen.main else { return }
        let screenFrame = screen.visibleFrame
        let x = screenFrame.midX - (frame.width / 2)
        let y = screenFrame.maxY - frame.height - 10  // 10px gap below menu bar
        setFrameOrigin(NSPoint(x: x, y: y))
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported")
    }
}
