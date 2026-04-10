import AppKit
import SwiftUI

@MainActor
final class CorrectionPanel {
    static let shared = CorrectionPanel()

    private var panel: NSPanel?
    private var clickMonitor: Any?
    private var keyMonitor: Any?
    private var viewModel: CorrectionViewModel?

    private init() {}

    func show(
        output: CorrectionService.Output,
        correctionService: CorrectionService,
        inputClient: BeeInputClient
    ) {
        dismiss()

        let vm = CorrectionViewModel(output: output)
        self.viewModel = vm

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
            viewModel: vm,
            onApply: { [weak self] in
                guard let vm = self?.viewModel else { return }

                // Send teaching signals for model edits
                let teachData = vm.resolutions.map { (editId: $0.key, accepted: $0.value) }
                Task {
                    await correctionService.teach(sessionId: output.sessionId, resolutions: teachData)
                    await correctionService.save()
                }

                // Use manual text if user edited, otherwise rebuild from resolutions
                let finalText = vm.isEditing
                    ? vm.editableText
                    : Self.rebuildText(output: output, resolutions: vm.resolutions)
                if finalText != output.bestText {
                    inputClient.replaceText(
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
        clickMonitor = NSEvent.addLocalMonitorForEvents(matching: .leftMouseDown) { [weak self, weak panel] event in
            guard let panel, let self else { return event }
            if !NSMouseInRect(NSEvent.mouseLocation, panel.frame, false) {
                self.dismiss()
            }
            return event
        }

        // Handle keyboard events (panel is non-activating so .onKeyPress won't work)
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, let vm = self.viewModel, !vm.isEditing else { return event }

            // Escape → dismiss
            if event.keyCode == 53 {
                self.dismiss()
                return nil
            }

            // Enter → accept
            if event.keyCode == 36 {
                // Trigger onApply via notification
                NotificationCenter.default.post(name: .correctionPanelAccept, object: nil)
                return nil
            }

            // 'e' → edit mode
            if event.charactersIgnoringModifiers == "e" {
                vm.enterEditMode()
                return nil
            }

            // 1-9 → toggle edit
            if let chars = event.charactersIgnoringModifiers,
               let digit = Int(chars), digit >= 1, digit <= 9 {
                withAnimation(.spring(response: 0.3)) {
                    vm.toggleEdit(at: digit - 1)
                }
                return nil
            }

            return event
        }

        self.panel = panel
    }

    func dismiss() {
        if let monitor = clickMonitor {
            NSEvent.removeMonitor(monitor)
            clickMonitor = nil
        }
        if let monitor = keyMonitor {
            NSEvent.removeMonitor(monitor)
            keyMonitor = nil
        }
        viewModel = nil
        panel?.close()
        panel = nil
    }

    /// Rebuild the output text applying the user's resolutions.
    static func rebuildText(
        output: CorrectionService.Output,
        resolutions: [String: Bool]
    ) -> String {
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

extension Notification.Name {
    static let correctionPanelAccept = Notification.Name("correctionPanelAccept")
}
