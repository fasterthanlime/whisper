import Carbon
import Foundation

class BeeXPCService: NSObject {
    static let shared = BeeXPCService()
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"

    weak var activeController: BeeInputController?
    var isDictating = false
    var pendingText: String?

    var controller: BeeInputController? {
        activeController
    }

    /// Called from BeeInputController.activateServer to flush any pending text.
    func flushPending() {
        if let text = pendingText, let ctrl = controller {
            beeInputLog("flushPending: delivering \(text.prefix(40).debugDescription)")
            pendingText = nil
            ctrl.handleSetMarkedText(text)
        }
    }

    func setMarkedText(_ text: String) {
        isDictating = true
        DispatchQueue.main.async {
            if let ctrl = self.controller {
                ctrl.handleSetMarkedText(text)
            } else {
                beeInputLog("setMarkedText: no controller, queuing \(text.prefix(40).debugDescription)")
                self.pendingText = text
            }
        }
    }

    func commitText(_ text: String, submit: Bool) {
        isDictating = false
        pendingText = nil
        DispatchQueue.main.async {
            self.controller?.handleCommitText(text, submit: submit)
        }
    }

    func cancelInput() {
        isDictating = false
        pendingText = nil
        DispatchQueue.main.async {
            self.controller?.handleCancelInput()
        }
    }

    func switchAwayFromBeeInput() {
        guard let current = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
              Self.isBeeInputSource(current) else {
            return
        }

        guard let fallback = Self.fallbackInputSource(current: current) else {
            beeInputLog("switchAwayFromBeeInput: no fallback available")
            return
        }

        let result = TISSelectInputSource(fallback)
        beeInputLog("switchAwayFromBeeInput: fallback select result=\(result)")
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
        guard let sources = TISCreateInputSourceList(properties as CFDictionary, false)?
            .takeRetainedValue() as? [TISInputSource],
              let source = sources.first else {
            return nil
        }
        return source
    }
}
