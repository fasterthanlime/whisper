import Carbon
import Foundation

/// Communicates with the bee-input IME process via distributed notifications.
final class BeeInputClient: Sendable {
    private static let dnc = DistributedNotificationCenter.default()

    private static let setMarkedTextName = NSNotification.Name("fasterthanlime.bee.setMarkedText")
    private static let commitTextName = NSNotification.Name("fasterthanlime.bee.commitText")
    private static let cancelInputName = NSNotification.Name("fasterthanlime.bee.cancelInput")
    private static let stopDictatingName = NSNotification.Name("fasterthanlime.bee.stopDictating")

    nonisolated(unsafe) private static var previousInputSource: TISInputSource?

    // MARK: - Input Source Switching

    func activate() {
        guard let beeSource = Self.findBeeInputSource() else {
            beeLog("IME ACTIVATE: bee input source NOT FOUND")
            return
        }
        Self.previousInputSource = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue()
        let result = TISSelectInputSource(beeSource)
        beeLog("IME ACTIVATE: TISSelectInputSource result=\(result)")
    }

    func deactivate() {
        if let previous = Self.previousInputSource {
            TISSelectInputSource(previous)
            Self.previousInputSource = nil
        }
    }

    // MARK: - Distributed Notifications

    func setMarkedText(_ text: String) {
        Self.dnc.postNotificationName(
            Self.setMarkedTextName,
            object: nil,
            userInfo: ["text": text],
            deliverImmediately: true
        )
    }

    func logSetMarkedText(_ text: String) {
        beeLog("IME setMarkedText: \(text.prefix(60).debugDescription)")
        setMarkedText(text)
    }

    func commitText(_ text: String) {
        Self.dnc.postNotificationName(
            Self.stopDictatingName,
            object: nil,
            userInfo: nil,
            deliverImmediately: true
        )
        Self.dnc.postNotificationName(
            Self.commitTextName,
            object: nil,
            userInfo: ["text": text],
            deliverImmediately: true
        )
    }

    func clearMarkedText() {
        Self.dnc.postNotificationName(
            Self.cancelInputName,
            object: nil,
            userInfo: nil,
            deliverImmediately: true
        )
    }

    func simulateReturn() {
        let src = CGEventSource(stateID: .hidSystemState)
        if let down = CGEvent(keyboardEventSource: src, virtualKey: 0x24, keyDown: true),
           let up = CGEvent(keyboardEventSource: src, virtualKey: 0x24, keyDown: false) {
            down.post(tap: .cghidEventTap)
            usleep(10_000) // 10ms
            up.post(tap: .cghidEventTap)
        }
    }

    // MARK: - IME Registration

    @discardableResult
    static func ensureIMERegistered() -> Bool {
        // Check if already registered
        if findBeeInputSource() != nil { return true }

        // Look for ~/Library/Input Methods/bee-input.app
        let inputMethodsDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Input Methods/beeInput.app")
        guard FileManager.default.fileExists(atPath: inputMethodsDir.path) else {
            return false
        }

        let status = TISRegisterInputSource(inputMethodsDir as CFURL)
        guard status == noErr else { return false }

        // Enable it
        if let source = findBeeInputSource() {
            TISEnableInputSource(source)
            return true
        }
        return false
    }

    static func restoreInputSourceIfNeeded() {
        if let previous = previousInputSource {
            TISSelectInputSource(previous)
            previousInputSource = nil
        }
    }

    private static func findBeeInputSource() -> TISInputSource? {
        let properties: [CFString: Any] = [
            kTISPropertyBundleID: "fasterthanlime.inputmethod.bee" as CFString,
        ]
        guard let sources = TISCreateInputSourceList(properties as CFDictionary, false)?.takeRetainedValue() as? [TISInputSource],
              let source = sources.first else {
            return nil
        }
        return source
    }
}
