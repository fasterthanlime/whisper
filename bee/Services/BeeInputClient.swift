import Carbon
import Foundation

// h[impl ime.communication]
/// Communicates with the bee-input IME process via distributed notifications.
final class BeeInputClient: Sendable {
    private static let dnc = DistributedNotificationCenter.default()

    private static let setMarkedTextName = NSNotification.Name("fasterthanlime.bee.setMarkedText")
    private static let commitTextName = NSNotification.Name("fasterthanlime.bee.commitText")
    private static let cancelInputName = NSNotification.Name("fasterthanlime.bee.cancelInput")
    private static let stopDictatingName = NSNotification.Name("fasterthanlime.bee.stopDictating")

    nonisolated(unsafe) private static var previousInputSource: TISInputSource?

    // MARK: - Input Source Switching

    // h[impl ime.activate]
    func activate() {
        guard let beeSource = Self.findBeeInputSource() else { return }
        Self.previousInputSource = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue()
        TISSelectInputSource(beeSource)
    }

    // h[impl ime.deactivate]
    func deactivate() {
        if let previous = Self.previousInputSource {
            TISSelectInputSource(previous)
            Self.previousInputSource = nil
        }
    }

    // MARK: - Distributed Notifications

    // h[impl ime.marked-text]
    func setMarkedText(_ text: String) {
        Self.dnc.postNotificationName(
            Self.setMarkedTextName,
            object: nil,
            userInfo: ["text": text],
            deliverImmediately: true
        )
    }

    // h[impl ime.commit]
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

    // h[impl ime.clear-on-cancel]
    func clearMarkedText() {
        Self.dnc.postNotificationName(
            Self.cancelInputName,
            object: nil,
            userInfo: nil,
            deliverImmediately: true
        )
    }

    // h[impl ime.submit]
    func simulateReturn() {
        // TODO: create CGEvent for Return keyDown + keyUp, post to HID
    }

    // MARK: - IME Discovery

    // h[impl ime.safety.restore-on-quit]
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
