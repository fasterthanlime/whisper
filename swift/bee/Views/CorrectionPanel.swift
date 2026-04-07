import AppKit
import SwiftUI

@MainActor
final class CorrectionPanel {
    static let shared = CorrectionPanel()

    private var panel: NSPanel?
    private var eventMonitor: Any?

    private init() {}

    func show(
        output: CorrectionService.Output,
        correctionService: CorrectionService,
        inputClient: BeeInputClient
    ) {
        dismiss()

        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 200),
            styleMask: [.nonactivatingPanel, .fullSizeContentView, .titled, .closable],
            backing: .buffered,
            defer: false
        )
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.titlebarAppearsTransparent = true
        panel.titleVisibility = .hidden
        panel.isMovableByWindowBackground = true
        panel.backgroundColor = .clear
        panel.hasShadow = true

        let view = CorrectionTrackView(
            output: output,
            onApply: { [weak self] resolutions, customEdits in
                // Send teaching signals for model edits
                let teachData = resolutions.map { (editId: $0.key, accepted: $0.value) }
                Task {
                    await correctionService.teach(sessionId: output.sessionId, resolutions: teachData)
                    await correctionService.save()
                }

                // Rebuild final text from resolutions + custom edits
                let finalText = Self.rebuildText(output: output, resolutions: resolutions, customEdits: customEdits)
                if finalText != output.bestText {
                    inputClient.replaceText(
                        sessionId: output.sessionId,
                        oldText: output.bestText,
                        newText: finalText
                    )
                }

                self?.dismiss()
            },
            onDismiss: { [weak self] in
                self?.dismiss()
            }
        )

        panel.contentView = NSHostingView(rootView: view)
        panel.center()
        panel.makeKeyAndOrderFront(nil)

        // Close on click outside
        eventMonitor = NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown) { [weak self, weak panel] event in
            guard let panel, let self else { return event }
            if !NSMouseInRect(NSEvent.mouseLocation, panel.frame, false) {
                self.dismiss()
            }
            return event
        }

        self.panel = panel
    }

    func dismiss() {
        if let monitor = eventMonitor {
            NSEvent.removeMonitor(monitor)
            eventMonitor = nil
        }
        panel?.close()
        panel = nil
    }

    /// Rebuild the output text applying the user's resolutions and custom edits.
    static func rebuildText(
        output: CorrectionService.Output,
        resolutions: [String: Bool],
        customEdits: [CorrectionTrackView.CustomEdit] = []
    ) -> String {
        // Collect all edits (model + custom) as (start, end, replacement), sorted in reverse
        struct EditRange: Comparable {
            let start: Int
            let end: Int
            let replacement: String
            static func < (lhs: EditRange, rhs: EditRange) -> Bool { lhs.start > rhs.start }
        }

        var allEdits: [EditRange] = []

        for edit in output.edits {
            let accepted = resolutions[edit.editId] ?? true
            let replacement = accepted ? edit.replacement : edit.original
            allEdits.append(EditRange(start: Int(edit.spanStart), end: Int(edit.spanEnd), replacement: replacement))
        }

        for custom in customEdits {
            allEdits.append(EditRange(start: custom.charStart, end: custom.charEnd, replacement: custom.replacement))
        }

        allEdits.sort()

        var chars = Array(output.originalText)
        for edit in allEdits {
            let start = chars.index(chars.startIndex, offsetBy: edit.start)
            let end = chars.index(chars.startIndex, offsetBy: edit.end)
            chars.replaceSubrange(start..<end, with: edit.replacement)
        }

        return String(chars)
    }
}
