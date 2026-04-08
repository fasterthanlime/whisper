import SwiftUI

// MARK: - View Model

@MainActor
@Observable
final class CorrectionViewModel {
    let output: CorrectionService.Output
    var resolutions: [String: Bool]
    var isEditing = false
    var editableText: String = ""

    init(output: CorrectionService.Output) {
        self.output = output
        var initial: [String: Bool] = [:]
        for edit in output.edits {
            initial[edit.editId] = true
        }
        self.resolutions = initial
    }

    var sortedEdits: [IndexedEdit] {
        output.edits
            .sorted { $0.spanStart < $1.spanStart }
            .enumerated()
            .map { IndexedEdit(index: $0.offset + 1, edit: $0.element) }
    }

    func toggleEdit(at index: Int) {
        let edits = sortedEdits
        guard index < edits.count else { return }
        let editId = edits[index].edit.editId
        resolutions[editId] = !(resolutions[editId] ?? true)
    }

    func enterEditMode() {
        editableText = currentText
        isEditing = true
    }

    /// Build the current text from resolutions.
    var currentText: String {
        var text = output.originalText
        let edits = output.edits.sorted { $0.spanStart > $1.spanStart }
        for edit in edits {
            let accepted = resolutions[edit.editId] ?? true
            let replacement = accepted ? edit.replacement : edit.original
            let start = text.index(text.startIndex, offsetBy: Int(edit.spanStart))
            let end = text.index(text.startIndex, offsetBy: Int(edit.spanEnd))
            text.replaceSubrange(start..<end, with: replacement)
        }
        return text
    }
}

// MARK: - Main View

struct CorrectionTrackView: View {
    @Bindable var viewModel: CorrectionViewModel
    let onApply: () -> Void
    let onDismiss: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Text("\(viewModel.output.edits.count) correction\(viewModel.output.edits.count == 1 ? "" : "s")")
                    .font(.headline)
                Spacer()
                if !viewModel.isEditing {
                    Text("E to edit")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
                }
            }

            if viewModel.isEditing {
                TextEditor(text: $viewModel.editableText)
                    .font(.body)
                    .frame(minHeight: 60, maxHeight: 200)
                    .scrollContentBackground(.hidden)
                    .padding(4)
                    .background(
                        RoundedRectangle(cornerRadius: 6)
                            .fill(Color.primary.opacity(0.05))
                    )
            } else {
                FlowLayout(spacing: 0) {
                    ForEach(segments) { segment in
                        switch segment.kind {
                        case .plain(let text):
                            Text(text)
                                .font(.body)

                        case .edit(let ie):
                            let accepted = viewModel.resolutions[ie.edit.editId] ?? true
                            TrackSwitchView(
                                active: accepted ? ie.edit.replacement : ie.edit.original,
                                inactive: accepted ? ie.edit.original : ie.edit.replacement,
                                number: ie.index,
                                sidingUp: ie.index % 2 != 0,
                                accepted: accepted
                            ) {
                                withAnimation(.spring(response: 0.3)) {
                                    viewModel.resolutions[ie.edit.editId] = !accepted
                                }
                            }
                        }
                    }
                }
                .padding(.vertical, 24)
            }

            Divider()

            HStack {
                Button("Dismiss") { onDismiss() }
                Spacer()
                Button("Accept") { onApply() }
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
        .frame(minWidth: 550)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .onReceive(NotificationCenter.default.publisher(for: .correctionPanelAccept)) { _ in
            onApply()
        }
    }

    // MARK: - Segments

    private var segments: [Segment] {
        let text = viewModel.output.originalText
        let edits = viewModel.sortedEdits
        var result: [Segment] = []
        var cursor = text.startIndex

        for ie in edits {
            let spanStart = text.index(text.startIndex, offsetBy: Int(ie.edit.spanStart), limitedBy: text.endIndex)
                ?? text.endIndex
            let spanEnd = text.index(text.startIndex, offsetBy: Int(ie.edit.spanEnd), limitedBy: text.endIndex)
                ?? text.endIndex

            if cursor < spanStart {
                result.append(Segment(kind: .plain(String(text[cursor..<spanStart]))))
            }
            result.append(Segment(kind: .edit(ie)))
            cursor = spanEnd
        }

        if cursor < text.endIndex {
            result.append(Segment(kind: .plain(String(text[cursor...]))))
        }
        return result
    }
}

// MARK: - Data types

struct IndexedEdit {
    let index: Int
    let edit: CorrectionEdit
}

struct Segment: Identifiable {
    let id = UUID()
    let kind: Kind
    enum Kind {
        case plain(String)
        case edit(IndexedEdit)
    }
}

// MARK: - Track Switch

struct TrackSwitchView: View {
    let active: String
    let inactive: String
    let number: Int
    let sidingUp: Bool
    let accepted: Bool
    let toggle: () -> Void

    private let curveHeight: CGFloat = 14

    var body: some View {
        Button(action: toggle) {
            Text(active)
                .font(.body)
                .padding(.horizontal, 2)
                .overlay(alignment: .bottom) {
                    DashedUnderline()
                        .stroke(style: StrokeStyle(lineWidth: 1, dash: [3, 2]))
                        .foregroundStyle(.secondary.opacity(0.4))
                        .frame(height: 1)
                        .offset(y: 2)
                }
        }
        .buttonStyle(.plain)
        .overlay(alignment: sidingUp ? .top : .bottom) {
            VStack(spacing: 0) {
                if sidingUp {
                    sidingContent
                    sidingCurve
                } else {
                    sidingCurve
                    sidingContent
                }
            }
            .fixedSize()
            .offset(y: sidingUp ? -(curveHeight + 20) : (curveHeight + 20))
            .allowsHitTesting(false)
        }
    }

    private var sidingContent: some View {
        HStack(spacing: 5) {
            Text(circledNumber(number))
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(inactive)
                .font(.system(size: NSFont.systemFontSize))
                .foregroundStyle(.primary.opacity(0.6))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(
            RoundedRectangle(cornerRadius: 5)
                .fill(Color.secondary.opacity(0.15))
        )
    }

    @ViewBuilder
    private var sidingCurve: some View {
        SidingCurve(goesUp: sidingUp)
            .stroke(Color.secondary.opacity(0.3), lineWidth: 1.5)
            .frame(height: curveHeight)
    }

    private func circledNumber(_ n: Int) -> String {
        let circled = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨"]
        return n >= 1 && n <= 9 ? circled[n - 1] : "(\(n))"
    }
}

// MARK: - Dashed underline

struct DashedUnderline: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        p.move(to: CGPoint(x: rect.minX, y: rect.midY))
        p.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        return p
    }
}

// MARK: - Bézier siding curve

struct SidingCurve: Shape {
    let goesUp: Bool

    func path(in rect: CGRect) -> Path {
        var p = Path()
        if goesUp {
            p.move(to: CGPoint(x: rect.midX - 8, y: rect.maxY))
            p.addCurve(
                to: CGPoint(x: rect.midX, y: rect.minY),
                control1: CGPoint(x: rect.midX - 8, y: rect.midY),
                control2: CGPoint(x: rect.midX, y: rect.midY)
            )
        } else {
            p.move(to: CGPoint(x: rect.midX - 8, y: rect.minY))
            p.addCurve(
                to: CGPoint(x: rect.midX, y: rect.maxY),
                control1: CGPoint(x: rect.midX - 8, y: rect.midY),
                control2: CGPoint(x: rect.midX, y: rect.midY)
            )
        }
        return p
    }
}

// MARK: - Flow Layout

struct FlowLayout: Layout {
    var spacing: CGFloat = 4

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let maxWidth = proposal.width ?? .infinity
        var x: CGFloat = 0
        var y: CGFloat = 0
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > maxWidth && x > 0 {
                x = 0
                y += rowHeight + spacing
                rowHeight = 0
            }
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }

        return CGSize(width: maxWidth, height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var rowHeight: CGFloat = 0

        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x + size.width > bounds.maxX && x > bounds.minX {
                x = bounds.minX
                y += rowHeight + spacing
                rowHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: .unspecified)
            x += size.width + spacing
            rowHeight = max(rowHeight, size.height)
        }
    }
}
