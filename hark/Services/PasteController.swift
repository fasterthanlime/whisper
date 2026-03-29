import AppKit
import Carbon.HIToolbox

/// Pastes text into the frontmost application by writing to the pasteboard
/// and simulating Cmd+V.
@MainActor
struct PasteController {

    /// Paste text into the active application.
    /// Saves and restores the original clipboard contents.
    /// - Parameters:
    ///   - text: The text to paste
    ///   - submit: If true, simulate Enter after pasting (for submitting). If false, append a trailing space.
    static func paste(_ text: String, submit: Bool = false) async throws {
        guard hasAccessibilityPermission else {
            throw PasteError.accessibilityPermissionRequired
        }

        guard !text.isEmpty else { return }

        let pasteboard = NSPasteboard.general
        let snapshot = capturePasteboardSnapshot(from: pasteboard)

        // Append a trailing space unless we're submitting
        var finalText = text
        if !submit {
            finalText += " "
        }

        guard writeText(finalText, to: pasteboard) else {
            throw PasteError.pasteboardWriteFailed
        }

        let stagedChangeCount = pasteboard.changeCount

        // Small delay to ensure pasteboard is updated
        try? await Task.sleep(for: .milliseconds(50))

        // Simulate Cmd+V keystroke
        do {
            try await simulateCmdV()
        } catch {
            restorePasteboard(
                snapshot,
                to: pasteboard,
                expectedChangeCount: stagedChangeCount
            )
            throw error
        }

        // If submitting, simulate Enter after paste
        if submit {
            try? await Task.sleep(for: .milliseconds(300))
            try await simulateReturn()
        }

        // Restore original pasteboard after a delay (don't block the caller)
        let changeCount = stagedChangeCount
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(300))
            restorePasteboard(snapshot, to: pasteboard, expectedChangeCount: changeCount)
        }
    }

    /// Check if the Return/Enter key is currently pressed.
    static func isReturnKeyPressed() -> Bool {
        guard let source = CGEventSource(stateID: .hidSystemState) else {
            return false
        }
        return CGEventSource.keyState(.hidSystemState, key: CGKeyCode(kVK_Return))
    }

    /// Check if the app has Accessibility permission (required for CGEvent posting).
    static var hasAccessibilityPermission: Bool {
        AXIsProcessTrusted()
    }

    /// Prompt the user to grant Accessibility permission.
    static func requestAccessibilityPermission() {
        // Use string literal to avoid Swift 6 concurrency warning on global kAXTrustedCheckOptionPrompt
        let options = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
        AXIsProcessTrustedWithOptions(options)
    }

    // MARK: - Direct AX input

    /// Captures the focused AX text element so we can write to it repeatedly
    /// during streaming transcription.
    static func captureFocusedTextField() -> AXUIElement? {
        let systemWide = AXUIElementCreateSystemWide()
        var focusedRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedRef
        ) == .success,
              let focusedRef,
              CFGetTypeID(focusedRef) == AXUIElementGetTypeID()
        else {
            return nil
        }
        let element = unsafeBitCast(focusedRef, to: AXUIElement.self)

        // Verify it supports kAXValueAttribute (i.e. it's a text field)
        var valueRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXValueAttribute as CFString,
            &valueRef
        ) == .success, valueRef is String else {
            return nil
        }
        return element
    }

    /// Write text directly into an AX text element, replacing whatever is there
    /// from position `replaceFrom` onward. Returns the UTF-16 length written
    /// (for tracking on the next update), or nil on failure.
    @discardableResult
    static func setDirectText(
        _ text: String,
        on element: AXUIElement,
        replaceFrom: Int = 0
    ) -> Bool {
        // Read current value to splice our text after replaceFrom
        var valueRef: AnyObject?
        let currentText: String
        if AXUIElementCopyAttributeValue(
            element,
            kAXValueAttribute as CFString,
            &valueRef
        ) == .success, let existing = valueRef as? String {
            currentText = existing
        } else {
            currentText = ""
        }

        // Build new value: preserve everything before replaceFrom, append new text
        let ns = currentText as NSString
        let clampedFrom = min(max(0, replaceFrom), ns.length)
        let prefix = ns.substring(to: clampedFrom)
        let newValue = prefix + text

        // Set the value
        let setResult = AXUIElementSetAttributeValue(
            element,
            kAXValueAttribute as CFString,
            newValue as CFTypeRef
        )
        guard setResult == .success else { return false }

        // Move cursor to end
        let endPos = (newValue as NSString).length
        var range = CFRange(location: endPos, length: 0)
        if let rangeValue = AXValueCreate(.cfRange, &range) {
            AXUIElementSetAttributeValue(
                element,
                kAXSelectedTextRangeAttribute as CFString,
                rangeValue
            )
        }

        return true
    }

    // MARK: - Private

    /// Check whether the focused text field already has non-whitespace text
    /// immediately before the cursor, meaning we should prepend a space to the
    /// transcribed text so it doesn't jam against existing content.
    private static func shouldPrependSpace() -> Bool {
        let systemWide = AXUIElementCreateSystemWide()

        // Get the currently focused UI element
        var focusedRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedRef
        ) == .success,
              let focusedRef
        else {
            return false
        }
        guard CFGetTypeID(focusedRef) == AXUIElementGetTypeID() else {
            return false
        }
        let element = unsafeBitCast(focusedRef, to: AXUIElement.self)

        // Read the element's text value
        var valueRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXValueAttribute as CFString,
            &valueRef
        ) == .success,
              let text = valueRef as? String,
              !text.isEmpty
        else {
            return false
        }

        // Read the selected text range (cursor position)
        var rangeRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            &rangeRef
        ) == .success,
              let rangeRef
        else {
            return false
        }
        guard CFGetTypeID(rangeRef) == AXValueGetTypeID() else {
            return false
        }
        let rangeValue = unsafeBitCast(rangeRef, to: AXValue.self)
        guard AXValueGetType(rangeValue) == .cfRange else {
            return false
        }
        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(rangeValue, .cfRange, &range) else {
            return false
        }

        let cursorPosition = range.location
        guard cursorPosition > 0, cursorPosition <= text.utf16.count else {
            return false
        }

        // Get the character just before the cursor
        let utf16 = text.utf16
        let idx = utf16.index(utf16.startIndex, offsetBy: cursorPosition - 1)
        guard let scalar = Unicode.Scalar(utf16[idx]) else { return false }
        let char = Character(scalar)

        return !char.isWhitespace
    }

    private static func writeText(_ text: String, to pasteboard: NSPasteboard) -> Bool {
        pasteboard.clearContents()
        return pasteboard.setString(text, forType: .string)
    }

    private static func capturePasteboardSnapshot(from pasteboard: NSPasteboard) -> PasteboardSnapshot {
        let snapshots = (pasteboard.pasteboardItems ?? []).map { item in
            let payloads = item.types.compactMap { type -> PasteboardPayload? in
                guard let data = item.data(forType: type) else { return nil }
                return PasteboardPayload(type: type, data: data)
            }
            return PasteboardItemSnapshot(payloads: payloads)
        }

        return PasteboardSnapshot(items: snapshots)
    }

    private static func restorePasteboard(
        _ snapshot: PasteboardSnapshot,
        to pasteboard: NSPasteboard,
        expectedChangeCount: Int
    ) {
        guard pasteboard.changeCount == expectedChangeCount else {
            return
        }

        pasteboard.clearContents()
        guard !snapshot.items.isEmpty else {
            return
        }

        let restoredItems: [NSPasteboardItem] = snapshot.items.compactMap { snapshotItem in
            guard !snapshotItem.payloads.isEmpty else { return nil }
            let item = NSPasteboardItem()
            for payload in snapshotItem.payloads {
                item.setData(payload.data, forType: payload.type)
            }
            return item
        }

        if !restoredItems.isEmpty {
            pasteboard.writeObjects(restoredItems)
        }
    }

    private static func simulateCmdV() async throws {
        let vKeyCode: CGKeyCode = CGKeyCode(kVK_ANSI_V)

        // Key down
        guard let keyDown = CGEvent(
            keyboardEventSource: nil,
            virtualKey: vKeyCode,
            keyDown: true
        ) else {
            throw PasteError.keyEventCreationFailed
        }
        keyDown.flags = .maskCommand
        keyDown.post(tap: .cghidEventTap)

        // Brief delay between down and up
        try? await Task.sleep(for: .milliseconds(10))

        // Key up
        guard let keyUp = CGEvent(
            keyboardEventSource: nil,
            virtualKey: vKeyCode,
            keyDown: false
        ) else {
            throw PasteError.keyEventCreationFailed
        }
        keyUp.flags = .maskCommand
        keyUp.post(tap: .cghidEventTap)
    }

    private static func simulateReturn() async throws {
        let returnKeyCode: CGKeyCode = CGKeyCode(kVK_Return)

        guard let keyDown = CGEvent(
            keyboardEventSource: nil,
            virtualKey: returnKeyCode,
            keyDown: true
        ) else {
            throw PasteError.keyEventCreationFailed
        }
        keyDown.flags = []  // Clear all modifiers — bare Return only.
        keyDown.post(tap: .cghidEventTap)

        try? await Task.sleep(for: .milliseconds(10))

        guard let keyUp = CGEvent(
            keyboardEventSource: nil,
            virtualKey: returnKeyCode,
            keyDown: false
        ) else {
            throw PasteError.keyEventCreationFailed
        }
        keyUp.flags = []
        keyUp.post(tap: .cghidEventTap)
    }

    private struct PasteboardSnapshot {
        let items: [PasteboardItemSnapshot]
    }

    private struct PasteboardItemSnapshot {
        let payloads: [PasteboardPayload]
    }

    private struct PasteboardPayload {
        let type: NSPasteboard.PasteboardType
        let data: Data
    }
}

enum PasteError: LocalizedError {
    case accessibilityPermissionRequired
    case pasteboardWriteFailed
    case keyEventCreationFailed

    var errorDescription: String? {
        switch self {
        case .accessibilityPermissionRequired:
            return "Accessibility permission is required to paste text"
        case .pasteboardWriteFailed:
            return "Unable to write text to the pasteboard"
        case .keyEventCreationFailed:
            return "Unable to synthesize Cmd+V key events"
        }
    }
}
