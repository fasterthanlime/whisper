import AppKit
import SwiftUI

// MARK: - Main View

/// Track view for correction review.
/// Renders the sentence in an NSTextView with railway-switch overlays at edit points.
/// Active option sits on the main line; inactive branches off on a curved siding.
struct CorrectionTrackView: View {
    let output: CorrectionService.Output
    @State var resolutions: [String: Bool]
    /// User-added custom corrections: (charStart, charEnd, replacement)
    @State var customEdits: [CustomEdit] = []
    let onApply: ([String: Bool], [CustomEdit]) -> Void
    let onDismiss: () -> Void

    struct CustomEdit: Identifiable {
        let id = UUID()
        var charStart: Int
        var charEnd: Int
        var original: String
        var replacement: String
    }

    init(
        output: CorrectionService.Output,
        onApply: @escaping ([String: Bool], [CustomEdit]) -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.output = output
        self.onApply = onApply
        self.onDismiss = onDismiss
        var initial: [String: Bool] = [:]
        for edit in output.edits {
            initial[edit.editId] = true
        }
        _resolutions = State(initialValue: initial)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Text("\(totalEditCount) correction\(totalEditCount == 1 ? "" : "s")")
                    .font(.headline)
                Spacer()
                Text("ROpt+C")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
            }

            // Track area with switches
            TrackTextView(
                text: output.originalText,
                edits: indexedEdits,
                customEdits: $customEdits,
                resolutions: $resolutions
            )
            .frame(minHeight: 60)

            Divider()

            // Actions
            HStack {
                Button("Dismiss") { onDismiss() }
                    .keyboardShortcut(.escape, modifiers: [])
                Spacer()
                Button("Accept") { onApply(resolutions, customEdits) }
                    .keyboardShortcut(.return, modifiers: [])
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
        .frame(minWidth: 550)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .onKeyPress(characters: .init(charactersIn: "123456789")) { press in
            let digit = Int(String(press.characters))! // safe: filtered above
            let idx = digit - 1
            let edits = indexedEdits
            guard idx < edits.count else { return .ignored }
            let editId = edits[idx].edit.editId
            resolutions[editId] = !(resolutions[editId] ?? true)
            return .handled
        }
    }

    private var totalEditCount: Int {
        output.edits.count + customEdits.count
    }

    /// Edits sorted by position with their display index (1-based).
    var indexedEdits: [IndexedEdit] {
        output.edits
            .sorted { $0.spanStart < $1.spanStart }
            .enumerated()
            .map { IndexedEdit(index: $0.offset + 1, edit: $0.element) }
    }
}

struct IndexedEdit {
    let index: Int
    let edit: CorrectionEdit
}

// MARK: - NSTextView wrapper

/// Wraps an NSTextView that displays the full sentence.
/// Edit spans are replaced with placeholder attachments, and
/// TrackSwitchView overlays are positioned over them.
struct TrackTextView: NSViewRepresentable {
    let text: String
    let edits: [IndexedEdit]
    @Binding var customEdits: [CorrectionTrackView.CustomEdit]
    @Binding var resolutions: [String: Bool]

    func makeNSView(context: Context) -> TrackTextContainer {
        let container = TrackTextContainer()
        container.onTextSelected = { range, selectedText in
            context.coordinator.handleSelection(range: range, selectedText: selectedText)
        }
        context.coordinator.container = container
        return container
    }

    func updateNSView(_ container: TrackTextContainer, context: Context) {
        context.coordinator.edits = edits
        context.coordinator.resolutions = resolutions
        context.coordinator.customEdits = customEdits
        container.updateContent(text: text, edits: edits, resolutions: resolutions)
    }

    func makeCoordinator() -> TrackTextCoordinator {
        TrackTextCoordinator(parent: self)
    }
}

// MARK: - Container view (NSView hosting NSTextView + overlays)

final class TrackTextContainer: NSView {
    let textView: NSTextView
    let scrollView: NSScrollView
    private var switchViews: [String: NSView] = [:]
    var onTextSelected: ((NSRange, String) -> Void)?

    override init(frame: NSRect) {
        let sv = NSScrollView(frame: frame)
        sv.hasVerticalScroller = false
        sv.hasHorizontalScroller = false
        sv.drawsBackground = false

        let tv = NSTextView(frame: frame)
        tv.isEditable = false
        tv.isSelectable = true
        tv.drawsBackground = false
        tv.isRichText = true
        tv.textContainerInset = NSSize(width: 4, height: 4)
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.font = NSFont.systemFont(ofSize: NSFont.systemFontSize)

        sv.documentView = tv
        self.textView = tv
        self.scrollView = sv

        super.init(frame: frame)

        sv.translatesAutoresizingMaskIntoConstraints = false
        addSubview(sv)
        NSLayoutConstraint.activate([
            sv.topAnchor.constraint(equalTo: topAnchor),
            sv.bottomAnchor.constraint(equalTo: bottomAnchor),
            sv.leadingAnchor.constraint(equalTo: leadingAnchor),
            sv.trailingAnchor.constraint(equalTo: trailingAnchor),
        ])
    }

    @available(*, unavailable)
    required init?(coder _: NSCoder) { fatalError() }

    func updateContent(text: String, edits: [IndexedEdit], resolutions: [String: Bool]) {
        let body = NSFont.systemFont(ofSize: NSFont.systemFontSize)
        let attributed = NSMutableAttributedString(
            string: text,
            attributes: [
                .font: body,
                .foregroundColor: NSColor.labelColor,
            ]
        )

        // Highlight edit spans
        for ie in edits {
            let accepted = resolutions[ie.edit.editId] ?? true
            let start = Int(ie.edit.spanStart)
            let end = Int(ie.edit.spanEnd)
            guard start < text.count, end <= text.count, start < end else { continue }

            let range = NSRange(location: start, length: end - start)

            // Replace the span text with the active option
            let activeText = accepted ? ie.edit.replacement : ie.edit.original
            let highlightColor = accepted
                ? NSColor.systemGreen.withAlphaComponent(0.15)
                : NSColor.clear

            let replacement = NSMutableAttributedString(
                string: activeText,
                attributes: [
                    .font: body,
                    .foregroundColor: NSColor.labelColor,
                    .backgroundColor: highlightColor,
                ]
            )

            // We can't replace in-place easily because offsets shift.
            // Instead, highlight the original span and add overlays.
            attributed.addAttributes([
                .backgroundColor: highlightColor,
            ], range: range)
        }

        // Only update if content changed to avoid flicker
        if textView.attributedString() != attributed {
            textView.textStorage?.setAttributedString(attributed)
        }

        // Position overlays after layout
        DispatchQueue.main.async { [weak self] in
            self?.positionOverlays(edits: edits, resolutions: resolutions)
        }
    }

    private func positionOverlays(edits: [IndexedEdit], resolutions: [String: Bool]) {
        guard let layoutManager = textView.layoutManager,
              let textContainer = textView.textContainer else { return }

        // Remove old overlays
        for (_, view) in switchViews {
            view.removeFromSuperview()
        }
        switchViews.removeAll()

        for (idx, ie) in edits.enumerated() {
            let start = Int(ie.edit.spanStart)
            let end = Int(ie.edit.spanEnd)
            let range = NSRange(location: start, length: end - start)

            // Get bounding rect for the span
            var glyphRange = NSRange()
            layoutManager.characterRange(forGlyphRange: range, actualGlyphRange: &glyphRange)
            let boundingRect = layoutManager.boundingRect(forGlyphRange: glyphRange, in: textContainer)

            let accepted = resolutions[ie.edit.editId] ?? true
            let inactiveText = accepted ? ie.edit.original : ie.edit.replacement

            // Determine siding direction: alternate up/down
            let goesUp = idx % 2 == 0

            // Create siding overlay
            let sidingView = TrackSidingView(
                inactiveText: inactiveText,
                number: ie.index,
                goesUp: goesUp,
                spanRect: boundingRect,
                accepted: accepted
            )

            let hostingView = NSHostingView(rootView: sidingView)
            hostingView.translatesAutoresizingMaskIntoConstraints = true

            let sidingHeight: CGFloat = 22
            let curveHeight: CGFloat = 12
            let totalHeight = sidingHeight + curveHeight

            let offsetInTextView = textView.textContainerInset
            let origin: CGPoint
            if goesUp {
                origin = CGPoint(
                    x: boundingRect.minX + offsetInTextView.width - 8,
                    y: boundingRect.minY + offsetInTextView.height - totalHeight
                )
            } else {
                origin = CGPoint(
                    x: boundingRect.minX + offsetInTextView.width - 8,
                    y: boundingRect.maxY + offsetInTextView.height
                )
            }

            let width = max(boundingRect.width + 16, 60)
            hostingView.frame = CGRect(x: origin.x, y: origin.y, width: width, height: totalHeight)

            textView.addSubview(hostingView)
            switchViews[ie.edit.editId] = hostingView
        }
    }

    override var intrinsicContentSize: NSSize {
        guard let layoutManager = textView.layoutManager,
              let textContainer = textView.textContainer else {
            return NSSize(width: NSView.noIntrinsicMetric, height: 80)
        }
        layoutManager.ensureLayout(for: textContainer)
        let usedRect = layoutManager.usedRect(for: textContainer)
        // Add space for sidings above and below
        return NSSize(
            width: NSView.noIntrinsicMetric,
            height: usedRect.height + textView.textContainerInset.height * 2 + 80
        )
    }
}

// MARK: - Coordinator

final class TrackTextCoordinator {
    let parent: TrackTextView
    weak var container: TrackTextContainer?
    var edits: [IndexedEdit] = []
    var resolutions: [String: Bool] = [:]
    var customEdits: [CorrectionTrackView.CustomEdit] = []

    init(parent: TrackTextView) {
        self.parent = parent
    }

    func handleSelection(range: NSRange, selectedText: String) {
        guard !selectedText.isEmpty else { return }
        // Check this range doesn't overlap with existing edits
        for ie in edits {
            let eStart = Int(ie.edit.spanStart)
            let eEnd = Int(ie.edit.spanEnd)
            if range.location < eEnd && range.location + range.length > eStart {
                return // overlaps existing edit
            }
        }
        // TODO: show popover for custom correction
    }
}

// MARK: - Siding view (the curved branch)

struct TrackSidingView: View {
    let inactiveText: String
    let number: Int
    let goesUp: Bool
    let spanRect: CGRect
    let accepted: Bool

    private let curveHeight: CGFloat = 12
    private let sidingHeight: CGFloat = 22

    var body: some View {
        ZStack(alignment: goesUp ? .bottom : .top) {
            // The curve
            SidingCurve(goesUp: goesUp, curveHeight: curveHeight)
                .stroke(Color.secondary.opacity(0.3), lineWidth: 1.5)
                .frame(height: curveHeight)
                .frame(maxWidth: .infinity, alignment: .leading)

            // The siding content
            HStack(spacing: 4) {
                // Number badge
                Text(circledNumber(number))
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)

                Text(inactiveText)
                    .font(.system(size: NSFont.systemFontSize - 1))
                    .foregroundStyle(.secondary.opacity(0.6))
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.secondary.opacity(0.08))
            )
            .frame(maxWidth: .infinity, alignment: .center)
            .offset(y: goesUp ? -curveHeight : curveHeight)
        }
    }

    private func circledNumber(_ n: Int) -> String {
        let circled = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
        return n >= 1 && n <= 9 ? circled[n - 1] : "(\(n))"
    }
}

// MARK: - Bézier siding curve shape

struct SidingCurve: Shape {
    let goesUp: Bool
    let curveHeight: CGFloat

    var animatableData: CGFloat {
        get { curveHeight }
        set { }
    }

    func path(in rect: CGRect) -> Path {
        var p = Path()
        let midX = rect.midX
        let w = rect.width * 0.3

        if goesUp {
            // Start from bottom-left, curve up to top-center, curve back down to bottom-right
            p.move(to: CGPoint(x: midX - w, y: rect.maxY))
            p.addCurve(
                to: CGPoint(x: midX, y: rect.minY),
                control1: CGPoint(x: midX - w, y: rect.minY),
                control2: CGPoint(x: midX - w * 0.3, y: rect.minY)
            )
            p.addCurve(
                to: CGPoint(x: midX + w, y: rect.maxY),
                control1: CGPoint(x: midX + w * 0.3, y: rect.minY),
                control2: CGPoint(x: midX + w, y: rect.minY)
            )
        } else {
            // Start from top-left, curve down to bottom-center, curve back up to top-right
            p.move(to: CGPoint(x: midX - w, y: rect.minY))
            p.addCurve(
                to: CGPoint(x: midX, y: rect.maxY),
                control1: CGPoint(x: midX - w, y: rect.maxY),
                control2: CGPoint(x: midX - w * 0.3, y: rect.maxY)
            )
            p.addCurve(
                to: CGPoint(x: midX + w, y: rect.minY),
                control1: CGPoint(x: midX + w * 0.3, y: rect.maxY),
                control2: CGPoint(x: midX + w, y: rect.maxY)
            )
        }
        return p
    }
}
