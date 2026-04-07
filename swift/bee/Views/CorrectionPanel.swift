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
            contentRect: NSRect(x: 0, y: 0, width: 480, height: 200),
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
            onApply: { [weak self] resolutions in
                // Send teaching signals
                let teachData = resolutions.map { (editId: $0.key, accepted: $0.value) }
                Task {
                    await correctionService.teach(sessionId: output.sessionId, resolutions: teachData)
                    await correctionService.save()
                }

                // Rebuild final text from resolutions
                let finalText = Self.rebuildText(output: output, resolutions: resolutions)
                if finalText != output.bestText {
                    // Text needs replacing — send via IPC
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

    /// Rebuild the output text applying the user's resolutions.
    /// If a resolution is `true` (accepted), use the replacement; otherwise keep original.
    static func rebuildText(
        output: CorrectionService.Output,
        resolutions: [String: Bool]
    ) -> String {
        // Start with the original text, apply edits in reverse order
        var chars = Array(output.originalText)
        let sortedEdits = output.edits.sorted { $0.spanStart > $1.spanStart }

        for edit in sortedEdits {
            let accepted = resolutions[edit.editId] ?? true
            let replacement = accepted ? edit.replacement : edit.original
            let start = chars.index(chars.startIndex, offsetBy: Int(edit.spanStart))
            let end = chars.index(chars.startIndex, offsetBy: Int(edit.spanEnd))
            chars.replaceSubrange(start..<end, with: replacement)
        }

        return String(chars)
    }
}
