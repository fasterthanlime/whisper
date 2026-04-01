import AppKit
import Carbon
import Foundation

/// Communicates with the bee-input IME process via distributed notifications.
final class BeeInputClient: Sendable {
    private static let dnc = DistributedNotificationCenter.default()
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"

    private static let setMarkedTextName = NSNotification.Name("fasterthanlime.bee.setMarkedText")
    private static let commitTextName = NSNotification.Name("fasterthanlime.bee.commitText")
    private static let cancelInputName = NSNotification.Name("fasterthanlime.bee.cancelInput")
    private static let stopDictatingName = NSNotification.Name("fasterthanlime.bee.stopDictating")
    private static let setSessionContextName = NSNotification.Name("fasterthanlime.bee.setSessionContext")

    nonisolated(unsafe) private static var previousInputSource: TISInputSource?

    // MARK: - Input Source Switching

    @discardableResult
    func activate(sessionID: UUID) -> Bool {
        let userInfo: [AnyHashable: Any] = [
            "sessionID": sessionID.uuidString,
        ]

        Self.dnc.postNotificationName(
            Self.setSessionContextName,
            object: nil,
            userInfo: userInfo,
            deliverImmediately: true
        )

        guard let beeSource = Self.findBeeInputSource() else {
            beeLog("IME ACTIVATE: bee input source NOT FOUND")
            return false
        }

        if let current = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
           !Self.isBeeInputSource(current) {
            Self.previousInputSource = current
        }

        let result = TISSelectInputSource(beeSource)
        beeLog("IME ACTIVATE: TISSelectInputSource result=\(result)")
        guard result == noErr else {
            return false
        }

        return true
    }

    func deactivate() {
        Self.switchAwayFromBeeInputIfNeeded()
    }

    // MARK: - Distributed Notifications

    func setMarkedText(_ text: String, sessionID: UUID) {
        Self.dnc.postNotificationName(
            Self.setMarkedTextName,
            object: nil,
            userInfo: [
                "sessionID": sessionID.uuidString,
                "text": text,
            ],
            deliverImmediately: true
        )
    }

    func logSetMarkedText(_ text: String, sessionID: UUID) {
        beeLog("IME setMarkedText: \(text.prefix(60).debugDescription)")
        setMarkedText(text, sessionID: sessionID)
    }

    func commitText(_ text: String, sessionID: UUID) {
        Self.dnc.postNotificationName(
            Self.commitTextName,
            object: nil,
            userInfo: [
                "sessionID": sessionID.uuidString,
                "text": text,
            ],
            deliverImmediately: true
        )
    }

    func clearMarkedText(sessionID: UUID) {
        Self.dnc.postNotificationName(
            Self.cancelInputName,
            object: nil,
            userInfo: ["sessionID": sessionID.uuidString],
            deliverImmediately: true
        )
    }

    func stopDictating(sessionID: UUID) {
        Self.dnc.postNotificationName(
            Self.stopDictatingName,
            object: nil,
            userInfo: ["sessionID": sessionID.uuidString],
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
        switchAwayFromBeeInputIfNeeded()
    }

    static func switchAwayFromBeeInputIfNeeded() {
        if let previous = previousInputSource, !isBeeInputSource(previous) {
            let result = TISSelectInputSource(previous)
            beeLog("IME DEACTIVATE: restore previous result=\(result)")
            previousInputSource = nil
            if result == noErr { return }
        }
        previousInputSource = nil

        guard let current = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
              isBeeInputSource(current) else {
            return
        }

        guard let fallback = fallbackInputSource(current: current) else {
            beeLog("IME DEACTIVATE: no fallback input source available")
            return
        }

        let result = TISSelectInputSource(fallback)
        beeLog("IME DEACTIVATE: fallback select result=\(result)")
    }

    private static func fallbackInputSource(current: TISInputSource) -> TISInputSource? {
        if let next = nextInputSource(after: current) {
            return next
        }

        if let ascii = TISCopyCurrentASCIICapableKeyboardLayoutInputSource()?.takeRetainedValue(),
           !isBeeInputSource(ascii) {
            return ascii
        }

        return selectCapableInputSources().first(where: { !isBeeInputSource($0) })
    }

    private static func nextInputSource(after current: TISInputSource) -> TISInputSource? {
        let sources = selectCapableInputSources()
        guard !sources.isEmpty else { return nil }

        guard let currentIndex = sources.firstIndex(where: { CFEqual($0, current) }) else {
            return sources.first(where: { !isBeeInputSource($0) })
        }

        for offset in 1...sources.count {
            let index = (currentIndex + offset) % sources.count
            let candidate = sources[index]
            if !isBeeInputSource(candidate) {
                return candidate
            }
        }
        return nil
    }

    private static func selectCapableInputSources() -> [TISInputSource] {
        let properties: [CFString: Any] = [
            kTISPropertyInputSourceIsSelectCapable: true,
        ]
        return (TISCreateInputSourceList(properties as CFDictionary, false)?
            .takeRetainedValue() as? [TISInputSource]) ?? []
    }

    private static func isBeeInputSource(_ source: TISInputSource?) -> Bool {
        guard let source, let beeSource = findBeeInputSource() else {
            return false
        }
        return CFEqual(source, beeSource)
    }

    private static func findBeeInputSource() -> TISInputSource? {
        let properties: [CFString: Any] = [
            kTISPropertyBundleID: beeBundleID as CFString,
        ]
        guard let sources = TISCreateInputSourceList(properties as CFDictionary, false)?.takeRetainedValue() as? [TISInputSource],
              let source = sources.first else {
            return nil
        }
        return source
    }
}
