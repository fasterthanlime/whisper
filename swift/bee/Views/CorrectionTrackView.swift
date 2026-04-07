import SwiftUI

/// Track view for correction review.
/// Shows the sentence split at correction boundaries.
/// At each edit: a two-lane toggle (original on top, correction on bottom).
/// Between edits: plain text.
struct CorrectionTrackView: View {
    let output: CorrectionService.Output
    @State var resolutions: [String: Bool]
    let onApply: ([String: Bool]) -> Void
    let onDismiss: () -> Void

    init(
        output: CorrectionService.Output,
        onApply: @escaping ([String: Bool]) -> Void,
        onDismiss: @escaping () -> Void
    ) {
        self.output = output
        self.onApply = onApply
        self.onDismiss = onDismiss
        // Default: all corrections accepted
        var initial: [String: Bool] = [:]
        for edit in output.edits {
            initial[edit.edit_id] = true
        }
        _resolutions = State(initialValue: initial)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Text("\(output.edits.count) correction\(output.edits.count == 1 ? "" : "s")")
                    .font(.headline)
                Spacer()
                Text("ROpt+C")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 4))
            }

            // Track: segments of plain text and edits
            FlowLayout(spacing: 2) {
                ForEach(segments, id: \.id) { segment in
                    switch segment {
                    case .plain(let id, let text):
                        Text(text)
                            .font(.body.monospaced())
                            .id(id)

                    case .edit(let id, let edit):
                        let accepted = resolutions[edit.edit_id] ?? true
                        EditLaneView(
                            original: edit.original,
                            replacement: edit.replacement,
                            accepted: accepted
                        ) {
                            resolutions[edit.edit_id] = !accepted
                        }
                        .id(id)
                    }
                }
            }

            Divider()

            // Actions
            HStack {
                Button("Dismiss") { onDismiss() }
                    .keyboardShortcut(.escape, modifiers: [])
                Spacer()
                Button("Reject All") {
                    for edit in output.edits {
                        resolutions[edit.edit_id] = false
                    }
                }
                Button("Accept All") {
                    for edit in output.edits {
                        resolutions[edit.edit_id] = true
                    }
                }
                Button("Apply") { onApply(resolutions) }
                    .keyboardShortcut(.return, modifiers: [])
                    .buttonStyle(.borderedProminent)
            }
        }
        .padding(16)
        .frame(minWidth: 400)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Segment computation

    enum Segment {
        case plain(id: String, text: String)
        case edit(id: String, edit: CorrectionService.Edit)

        var id: String {
            switch self {
            case .plain(let id, _): id
            case .edit(let id, _): id
            }
        }
    }

    var segments: [Segment] {
        let text = output.originalText
        let sortedEdits = output.edits.sorted { $0.span_start < $1.span_start }
        var result: [Segment] = []
        var cursor = text.startIndex

        for (i, edit) in sortedEdits.enumerated() {
            let spanStart = text.index(text.startIndex, offsetBy: edit.span_start, limitedBy: text.endIndex)
                ?? text.endIndex
            let spanEnd = text.index(text.startIndex, offsetBy: edit.span_end, limitedBy: text.endIndex)
                ?? text.endIndex

            // Plain text before this edit
            if cursor < spanStart {
                let plain = String(text[cursor..<spanStart])
                result.append(.plain(id: "p\(i)", text: plain))
            }

            result.append(.edit(id: "e\(i)", edit: edit))
            cursor = spanEnd
        }

        // Trailing plain text
        if cursor < text.endIndex {
            let plain = String(text[cursor...])
            result.append(.plain(id: "pEnd", text: plain))
        }

        return result
    }
}

/// A toggleable lane showing original (top) and replacement (bottom).
struct EditLaneView: View {
    let original: String
    let replacement: String
    let accepted: Bool
    let toggle: () -> Void

    var body: some View {
        Button(action: toggle) {
            VStack(spacing: 1) {
                // Original (top lane)
                Text(original)
                    .font(.body.monospaced())
                    .strikethrough(accepted)
                    .foregroundStyle(accepted ? .secondary : .primary)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(
                        accepted ? Color.clear : Color.red.opacity(0.15),
                        in: RoundedRectangle(cornerRadius: 3)
                    )

                // Replacement (bottom lane)
                Text(replacement)
                    .font(.body.monospaced())
                    .strikethrough(!accepted)
                    .foregroundStyle(accepted ? .primary : .secondary)
                    .padding(.horizontal, 4)
                    .padding(.vertical, 2)
                    .background(
                        accepted ? Color.green.opacity(0.15) : Color.clear,
                        in: RoundedRectangle(cornerRadius: 3)
                    )
            }
        }
        .buttonStyle(.plain)
    }
}

/// Simple flow layout for wrapping text segments.
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
