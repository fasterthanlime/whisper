import Cocoa
import InputMethodKit
import os

/// IMK input controller for Hark dictation.
/// Receives text from the main Hark app via XPC and inserts it into
/// the client application using setMarkedText / insertText.
@objc(HarkInputController)
class HarkInputController: IMKInputController {
    private static let logger = Logger(
        subsystem: "fasterthanlime.hark.input-method",
        category: "InputController"
    )

    /// The current marked (provisional) text, if any.
    private var currentMarkedText: String = ""

    // MARK: - Lifecycle

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        HarkXPCService.shared.activeController = self
        HarkXPCService.shared.lastController = self
        Self.logger.warning("Server activated")
    }

    override func deactivateServer(_ sender: Any!) {
        // Don't nil out the controller or auto-commit during an active
        // dictation session — focus changes cause transient deactivations.
        if currentMarkedText.isEmpty {
            if HarkXPCService.shared.activeController === self {
                HarkXPCService.shared.activeController = nil
            }
        }
        super.deactivateServer(sender)
        let hasMarked = !self.currentMarkedText.isEmpty
        Self.logger.warning("Server deactivated (hasMarkedText=\(hasMarked ? "yes" : "no"))")
    }

    // MARK: - Input handling

    /// We don't handle regular key input — just pass it through to the app.
    override func inputText(_ string: String!, client sender: Any!) -> Bool {
        return false
    }

    /// Handle events — pass everything through.
    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        return false
    }

    // MARK: - Commands from Hark via XPC

    /// Set provisional text (streaming transcription updates).
    func handleSetMarkedText(_ text: String) {
        guard let client = self.client() else {
            Self.logger.warning("No client for setMarkedText")
            return
        }

        currentMarkedText = text

        // Create attributed string with underline to indicate provisional text
        let attrs: [NSAttributedString.Key: Any] = [
            .underlineStyle: NSUnderlineStyle.single.rawValue,
            .underlineColor: NSColor.systemBlue.withAlphaComponent(0.5),
        ]
        let attributed = NSAttributedString(string: text, attributes: attrs)

        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: text.utf16.count, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    /// Commit final text, replacing marked text.
    func handleCommitText(_ text: String) {
        guard let client = self.client() else {
            Self.logger.warning("No client for commitText")
            return
        }

        currentMarkedText = ""
        client.insertText(
            text + " ",
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }

    /// Cancel — clear marked text without committing.
    func handleCancelInput() {
        guard let client = self.client() else {
            Self.logger.warning("No client for cancelInput")
            return
        }

        currentMarkedText = ""
        // Setting empty marked text clears it
        client.setMarkedText(
            "",
            selectionRange: NSRange(location: 0, length: 0),
            replacementRange: NSRange(location: NSNotFound, length: 0)
        )
    }
}
