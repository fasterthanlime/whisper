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

    /// Position the panel near the given cursor location while staying inside the visible frame.
    func positionNearCursor(_ cursor: NSPoint, on screen: NSScreen? = nil) {
        guard let screen = screen ?? NSScreen.main else { return }
        let visible = screen.visibleFrame
        let margin: CGFloat = 10
        let verticalOffset: CGFloat = 22

        var x = cursor.x - (frame.width / 2)
        var y = cursor.y + verticalOffset

        // If there isn't room above the cursor, place the bubble below it.
        if y + frame.height > visible.maxY - margin {
            y = cursor.y - frame.height - verticalOffset
        }

        x = min(max(x, visible.minX + margin), visible.maxX - frame.width - margin)
        y = min(max(y, visible.minY + margin), visible.maxY - frame.height - margin)
        setFrameOrigin(NSPoint(x: x, y: y))
    }

    /// Position the panel near a target frame (input/control) with a small gap.
    func positionNearTargetRect(_ target: CGRect, on screen: NSScreen? = nil) {
        guard let screen = screen ?? NSScreen.main else { return }
        let visible = screen.visibleFrame
        let margin: CGFloat = 10
        let insidePad: CGFloat = 18

        var x = target.midX - (frame.width / 2)

        // If the target is in the lower half, keep it near the bubble bottom so we grow upward.
        // If it's in the upper half, keep it near the bubble top so we grow downward.
        let growUp = target.midY <= visible.midY
        var y: CGFloat
        if growUp {
            y = target.minY - insidePad
        } else {
            y = target.maxY - frame.height + insidePad
        }

        x = min(max(x, visible.minX + margin), visible.maxX - frame.width - margin)
        y = min(max(y, visible.minY + margin), visible.maxY - frame.height - margin)
        setFrameOrigin(NSPoint(x: x, y: y))
    }

    /// Make the overlay body match the input frame exactly, with a footer band
    /// rendered above or below the body.
    func positionOverlayOnInputRect(
        _ target: CGRect,
        footerAbove: Bool,
        on screen: NSScreen? = nil
    ) {
        let screen = screen ?? NSScreen.main
        let visible = screen?.visibleFrame ?? target
        let margin: CGFloat = 10
        let bodyExpandX: CGFloat = 10
        let bodyExpandY: CGFloat = 8
        let footerBandHeight: CGFloat = 34
        let footerGap: CGFloat = 6
        let footerReserve = footerBandHeight + footerGap

        let expandedBody = target.insetBy(dx: -bodyExpandX, dy: -bodyExpandY)
        var frame = CGRect(
            x: expandedBody.origin.x,
            y: footerAbove ? expandedBody.origin.y : (expandedBody.origin.y - footerReserve),
            width: expandedBody.width,
            height: expandedBody.height + footerReserve
        )

        frame.origin.x = min(
            max(frame.origin.x, visible.minX + margin),
            visible.maxX - frame.width - margin
        )
        frame.origin.y = min(
            max(frame.origin.y, visible.minY + margin),
            visible.maxY - frame.height - margin
        )
        setFrame(frame, display: true)
    }

    /// Position for pane-like editors/terminals: stay near the lower input zone,
    /// without pretending the whole pane is a concrete input frame.
    func positionNearEditorRect(_ target: CGRect, on screen: NSScreen? = nil) {
        guard let screen = screen ?? NSScreen.main else { return }
        let visible = screen.visibleFrame
        let margin: CGFloat = 10

        var x = target.midX - (frame.width / 2)
        var y = target.minY + 14

        if y + frame.height > visible.maxY - margin {
            y = visible.maxY - frame.height - margin
        }

        x = min(max(x, visible.minX + margin), visible.maxX - frame.width - margin)
        y = min(max(y, visible.minY + margin), visible.maxY - frame.height - margin)
        setFrameOrigin(NSPoint(x: x, y: y))
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported")
    }
}

private final class FocusHighlightView: NSView {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer?.cornerRadius = 8
        layer?.borderWidth = 1.2
        layer?.borderColor = NSColor.systemTeal.withAlphaComponent(0.45).cgColor
        layer?.backgroundColor = NSColor.systemTeal.withAlphaComponent(0.04).cgColor
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) not supported")
    }
}

/// Lightweight outline that marks where text is expected to land.
final class FocusHighlightPanel: NSPanel {
    private let inset: CGFloat = 5

    init() {
        super.init(
            contentRect: .zero,
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )

        isFloatingPanel = true
        level = .floating
        animationBehavior = .none
        isMovableByWindowBackground = false
        hidesOnDeactivate = false

        isOpaque = false
        backgroundColor = .clear
        collectionBehavior = [.canJoinAllSpaces, .stationary]
        ignoresMouseEvents = true
        hasShadow = false

        contentView = FocusHighlightView(frame: .zero)
    }

    func setHighlightFrame(_ frame: CGRect) {
        let expanded = frame.insetBy(dx: -inset, dy: -inset)
        setFrame(expanded, display: true)
        orderFrontRegardless()
    }

    override var canBecomeKey: Bool {
        false
    }

    override var canBecomeMain: Bool {
        false
    }
}

private final class OverlayConnectorView: NSView {
    private var startPoint = CGPoint.zero
    private var endPoint = CGPoint.zero
    private let strokeColor = NSColor.systemTeal.withAlphaComponent(0.4)
    private let fillColor = NSColor.systemTeal.withAlphaComponent(0.26)

    override var isOpaque: Bool { false }

    func update(start: CGPoint, end: CGPoint) {
        startPoint = start
        endPoint = end
        needsDisplay = true
    }

    override func draw(_ dirtyRect: NSRect) {
        guard startPoint != endPoint else { return }

        let line = NSBezierPath()
        line.move(to: startPoint)
        line.line(to: endPoint)
        line.lineWidth = 1.6
        line.lineCapStyle = .round
        strokeColor.setStroke()
        line.stroke()

        let dx = endPoint.x - startPoint.x
        let dy = endPoint.y - startPoint.y
        let length = hypot(dx, dy)
        guard length > 1 else { return }

        let ux = dx / length
        let uy = dy / length
        let arrowLength: CGFloat = 8
        let arrowWidth: CGFloat = 4
        let base = CGPoint(x: endPoint.x - ux * arrowLength, y: endPoint.y - uy * arrowLength)
        let perp = CGPoint(x: -uy, y: ux)

        let arrow = NSBezierPath()
        arrow.move(to: endPoint)
        arrow.line(to: CGPoint(x: base.x + perp.x * arrowWidth, y: base.y + perp.y * arrowWidth))
        arrow.line(to: CGPoint(x: base.x - perp.x * arrowWidth, y: base.y - perp.y * arrowWidth))
        arrow.close()
        fillColor.setFill()
        arrow.fill()
    }
}

/// A transparent panel that draws a subtle connector line between overlay and target frame.
final class OverlayConnectorPanel: NSPanel {
    private let connectorView = OverlayConnectorView(frame: .zero)
    private let padding: CGFloat = 10

    init() {
        super.init(
            contentRect: .zero,
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: false
        )

        isFloatingPanel = true
        level = .floating
        animationBehavior = .none
        isMovableByWindowBackground = false
        hidesOnDeactivate = false

        isOpaque = false
        backgroundColor = .clear
        collectionBehavior = [.canJoinAllSpaces, .stationary]
        ignoresMouseEvents = true
        hasShadow = false

        contentView = connectorView
    }

    func setConnector(from start: CGPoint, to end: CGPoint) {
        guard start != end else {
            orderOut(nil)
            return
        }

        let minX = min(start.x, end.x) - padding
        let minY = min(start.y, end.y) - padding
        let maxX = max(start.x, end.x) + padding
        let maxY = max(start.y, end.y) + padding
        let frame = CGRect(x: minX, y: minY, width: max(1, maxX - minX), height: max(1, maxY - minY))

        setFrame(frame, display: true)
        connectorView.frame = CGRect(origin: .zero, size: frame.size)
        let localStart = CGPoint(x: start.x - frame.minX, y: start.y - frame.minY)
        let localEnd = CGPoint(x: end.x - frame.minX, y: end.y - frame.minY)
        connectorView.update(start: localStart, end: localEnd)
        orderFrontRegardless()
    }

    override var canBecomeKey: Bool {
        false
    }

    override var canBecomeMain: Bool {
        false
    }
}
