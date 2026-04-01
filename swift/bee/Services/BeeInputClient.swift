import AppKit
import Carbon
import Foundation

@objc
private protocol BeeIMEControlXPC {
    func prepareSession(_ sessionID: String, targetPID: Int32, activationID: String, withReply reply: @escaping (Bool) -> Void)
    func sessionStatus(_ sessionID: String, withReply reply: @escaping (Bool, Int32, String) -> Void)
    func clearSession(_ sessionID: String, withReply reply: @escaping () -> Void)
}

/// Communicates with the bee-input IME process via distributed notifications.
final class BeeInputClient: Sendable {
    private static let dnc = DistributedNotificationCenter.default()
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"
    private static let xpcServiceName = "fasterthanlime.inputmethod.bee.xpc"
    private static let readyPollIntervalMs: UInt64 = 20
    private static let readyTimeoutMs: UInt64 = 1200

    private static let setMarkedTextName = NSNotification.Name("fasterthanlime.bee.setMarkedText")
    private static let commitTextName = NSNotification.Name("fasterthanlime.bee.commitText")
    private static let cancelInputName = NSNotification.Name("fasterthanlime.bee.cancelInput")
    private static let stopDictatingName = NSNotification.Name("fasterthanlime.bee.stopDictating")

    nonisolated(unsafe) private static var previousInputSource: TISInputSource?
    nonisolated(unsafe) private static var xpcConnection: NSXPCConnection?
    private static let xpcLock = NSLock()

    // MARK: - Input Source Switching

    @discardableResult
    func activate(sessionID: UUID, targetPID: pid_t?) async -> Bool {
        let activationID = UUID().uuidString
        let prepared = await Self.prepareSessionXPC(
            sessionID: sessionID,
            targetPID: targetPID,
            activationID: activationID
        )
        guard prepared else {
            beeLog("IME ACTIVATE: prepareSession failed for session=\(sessionID.uuidString.prefix(8))")
            return false
        }

        let selected = await MainActor.run { Self.selectBeeInputSource() }
        guard selected else {
            await Self.clearSessionXPC(sessionID: sessionID)
            return false
        }

        let ready = await Self.awaitSessionReadyXPC(
            sessionID: sessionID,
            targetPID: targetPID,
            activationID: activationID
        )
        if !ready {
            beeLog("IME ACTIVATE: session ready timeout id=\(sessionID.uuidString.prefix(8))")
            await Self.clearSessionXPC(sessionID: sessionID)
            Self.switchAwayFromBeeInputIfNeeded()
        }
        return ready
    }

    @MainActor
    private static func selectBeeInputSource() -> Bool {
        guard let beeSource = findBeeInputSource() else {
            beeLog("IME ACTIVATE: bee input source NOT FOUND")
            return false
        }

        if let current = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
           !isBeeInputSource(current) {
            previousInputSource = current
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

    private static func getXPCConnection() -> NSXPCConnection {
        xpcLock.lock()
        defer { xpcLock.unlock() }
        if let connection = xpcConnection {
            return connection
        }
        let connection = NSXPCConnection(machServiceName: xpcServiceName, options: [])
        connection.remoteObjectInterface = NSXPCInterface(with: BeeIMEControlXPC.self)
        connection.resume()
        xpcConnection = connection
        return connection
    }

    private static func invalidateXPCConnection() {
        xpcLock.lock()
        defer { xpcLock.unlock() }
        xpcConnection?.invalidate()
        xpcConnection = nil
    }

    private static func prepareSessionXPC(sessionID: UUID, targetPID: pid_t?, activationID: String) async -> Bool {
        await withCheckedContinuation { continuation in
            let connection = getXPCConnection()
            let proxy = connection.remoteObjectProxyWithErrorHandler { error in
                beeLog("IME XPC prepareSession error: \(error.localizedDescription)")
                invalidateXPCConnection()
                continuation.resume(returning: false)
            } as? BeeIMEControlXPC

            guard let proxy else {
                continuation.resume(returning: false)
                return
            }

            proxy.prepareSession(
                sessionID.uuidString,
                targetPID: targetPID ?? -1,
                activationID: activationID
            ) { ok in
                continuation.resume(returning: ok)
            }
        }
    }

    private static func sessionStatusXPC(sessionID: UUID) async -> (ready: Bool, clientPID: pid_t?, clientID: String?) {
        await withCheckedContinuation { continuation in
            let connection = getXPCConnection()
            let proxy = connection.remoteObjectProxyWithErrorHandler { error in
                beeLog("IME XPC sessionStatus error: \(error.localizedDescription)")
                invalidateXPCConnection()
                continuation.resume(returning: (false, nil, nil))
            } as? BeeIMEControlXPC

            guard let proxy else {
                continuation.resume(returning: (false, nil, nil))
                return
            }

            proxy.sessionStatus(sessionID.uuidString) { ready, clientPID, clientID in
                let pid: pid_t? = clientPID >= 0 ? pid_t(clientPID) : nil
                let id: String? = clientID.isEmpty ? nil : clientID
                continuation.resume(returning: (ready, pid, id))
            }
        }
    }

    private static func awaitSessionReadyXPC(sessionID: UUID, targetPID: pid_t?, activationID: String) async -> Bool {
        let deadline = ProcessInfo.processInfo.systemUptime + (Double(readyTimeoutMs) / 1000.0)
        while ProcessInfo.processInfo.systemUptime < deadline {
            let status = await sessionStatusXPC(sessionID: sessionID)
            if status.ready {
                beeLog(
                    "IME ACTIVATE: session ready id=\(sessionID.uuidString.prefix(8)) activationID=\(activationID.prefix(8)) clientPID=\(status.clientPID.map(String.init) ?? "nil") clientID=\(status.clientID ?? "nil")"
                )
                return true
            }
            try? await Task.sleep(nanoseconds: readyPollIntervalMs * 1_000_000)
        }
        beeLog(
            "IME ACTIVATE: session not ready id=\(sessionID.uuidString.prefix(8)) activationID=\(activationID.prefix(8)) targetPID=\(targetPID.map(String.init) ?? "nil")"
        )
        return false
    }

    private static func clearSessionXPC(sessionID: UUID) async {
        await withCheckedContinuation { continuation in
            let connection = getXPCConnection()
            let proxy = connection.remoteObjectProxyWithErrorHandler { error in
                beeLog("IME XPC clearSession error: \(error.localizedDescription)")
                invalidateXPCConnection()
                continuation.resume()
            } as? BeeIMEControlXPC

            guard let proxy else {
                continuation.resume()
                return
            }

            proxy.clearSession(sessionID.uuidString) {
                continuation.resume()
            }
        }
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
