import AppKit
import Carbon
import Foundation

final class BeeIMEBridgeState: NSObject {
    static let shared = BeeIMEBridgeState()
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"

    weak var activeController: BeeInputController?
    private(set) var activeControllerPID: pid_t?
    private(set) var activeClientIdentity: String?

    /// Survives deactivateServer — the controller/client may still be usable
    /// for direct session claims when activateServer doesn't fire.
    weak var lastKnownController: BeeInputController?
    private(set) var lastKnownControllerPID: pid_t?
    private(set) var lastKnownClientIdentity: String?
    private(set) var activeSessionID: UUID?
    private var boundClientIdentity: String?
    var pendingText: String?

    var controller: BeeInputController? {
        activeController
    }

    var isDictating: Bool {
        activeSessionID != nil
    }

    var hasActiveSession: Bool {
        activeSessionID != nil
    }

    func attachSession(sessionID: UUID, clientIdentity: String?) {
        if activeSessionID != sessionID {
            pendingText = nil
        }
        activeSessionID = sessionID
        boundClientIdentity = clientIdentity
        beeInputLog(
            "attachSession: session=\(sessionID.uuidString.prefix(8)) clientID=\(clientIdentity ?? "nil")"
        )
    }

    func clearSessionIfMatching(sessionID: UUID) {
        guard activeSessionID == sessionID else { return }
        beeInputLog("clearSession: session=\(sessionID.uuidString.prefix(8))")
        clearSessionState()
    }

    func flushPending() {
        guard let text = pendingText else { return }
        guard canRouteToCurrentController() else {
            beeInputLog("flushPending: route not ready, keeping pending (\(routingDebugInfo()))")
            return
        }
        guard let ctrl = controller else {
            beeInputLog("flushPending: no controller, keeping pending")
            return
        }

        beeInputLog("flushPending: delivering \(text.prefix(40).debugDescription)")
        pendingText = nil
        ctrl.handleSetMarkedText(text)
    }

    func setMarkedText(_ text: String, sessionID: UUID) {
        DispatchQueue.main.async {
            guard self.activeSessionID == sessionID else {
                beeInputLog(
                    "setMarkedText: stale session=\(sessionID.uuidString.prefix(8)) current=\(self.activeSessionID?.uuidString.prefix(8) ?? "nil"), dropping"
                )
                return
            }

            guard self.canRouteToCurrentController() else {
                beeInputLog(
                    "setMarkedText: route not ready, queuing \(text.prefix(40).debugDescription) (\(self.routingDebugInfo()))"
                )
                self.pendingText = text
                return
            }

            if let ctrl = self.controller {
                ctrl.handleSetMarkedText(text)
            } else {
                beeInputLog("setMarkedText: no controller, queuing \(text.prefix(40).debugDescription)")
                self.pendingText = text
            }
        }
    }

    func commitText(_ text: String, submit: Bool, sessionID: UUID) {
        DispatchQueue.main.async {
            guard self.activeSessionID == sessionID else {
                beeInputLog(
                    "commitText: stale session=\(sessionID.uuidString.prefix(8)) current=\(self.activeSessionID?.uuidString.prefix(8) ?? "nil"), dropping"
                )
                return
            }

            let ctrl = self.controller
            self.clearSessionState()
            ctrl?.handleCommitText(text, submit: submit)
        }
    }

    func cancelInput(sessionID: UUID) {
        DispatchQueue.main.async {
            guard self.activeSessionID == sessionID else {
                beeInputLog(
                    "cancelInput: stale session=\(sessionID.uuidString.prefix(8)) current=\(self.activeSessionID?.uuidString.prefix(8) ?? "nil"), dropping"
                )
                return
            }

            let ctrl = self.controller
            self.clearSessionState()
            ctrl?.handleCancelInput()
        }
    }

    func stopDictating(sessionID: UUID) {
        DispatchQueue.main.async {
            guard self.activeSessionID == sessionID else {
                beeInputLog(
                    "stopDictating: stale session=\(sessionID.uuidString.prefix(8)) current=\(self.activeSessionID?.uuidString.prefix(8) ?? "nil"), dropping"
                )
                return
            }
            self.clearSessionState()
        }
    }

    private func clearSessionState() {
        activeSessionID = nil
        pendingText = nil
        boundClientIdentity = nil
    }

    func registerActiveController(_ controller: BeeInputController, clientPID: pid_t?, clientIdentity: String?) {
        activeController = controller
        activeControllerPID = clientPID
        activeClientIdentity = clientIdentity
        lastKnownController = controller
        lastKnownControllerPID = clientPID
        lastKnownClientIdentity = clientIdentity
        beeInputLog(
            "registerActiveController: pid=\(clientPID.map(String.init) ?? "nil") clientID=\(clientIdentity ?? "nil")"
        )
    }

    func unregisterActiveController(_ controller: BeeInputController) {
        guard activeController === controller else { return }
        activeController = nil
        activeControllerPID = nil
        activeClientIdentity = nil
    }

    private func canRouteToCurrentController() -> Bool {
        guard controller != nil else { return false }
        guard let boundClientIdentity else { return true }
        return activeClientIdentity == boundClientIdentity
    }

    private func routingDebugInfo() -> String {
        let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
        return "frontmost=\(frontmostPID.map(String.init) ?? "nil") controllerPID=\(activeControllerPID.map(String.init) ?? "nil") clientID=\(activeClientIdentity ?? "nil") boundClientID=\(boundClientIdentity ?? "nil") hasController=\(activeController != nil)"
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
