import AppKit
import Carbon
import Foundation

private final class BeeAppControlSink: NSObject, BeeBrokerPeerXPC {
    private static let imeSubmitName = NSNotification.Name("fasterthanlime.bee.imeSubmit")
    private static let imeCancelName = NSNotification.Name("fasterthanlime.bee.imeCancel")
    private static let imeUserTypedName = NSNotification.Name("fasterthanlime.bee.imeUserTyped")
    private static let imeContextLostName = NSNotification.Name("fasterthanlime.bee.imeContextLost")
    private static let imeSessionStartedName = NSNotification.Name(
        "fasterthanlime.bee.imeSessionStarted")

    private func post(_ name: NSNotification.Name, userInfo: [AnyHashable: Any]) {
        NotificationCenter.default.post(name: name, object: nil, userInfo: userInfo)
    }

    func handleIMESessionStarted(_ sessionID: String, clientPID: Int32, clientID: String) {
        var userInfo: [AnyHashable: Any] = ["sessionID": sessionID]
        if clientPID >= 0 {
            userInfo["clientPID"] = Int(clientPID)
        }
        if !clientID.isEmpty {
            userInfo["clientID"] = clientID
        }
        post(Self.imeSessionStartedName, userInfo: userInfo)
    }

    func handleIMESubmit(_ sessionID: String) {
        post(Self.imeSubmitName, userInfo: ["sessionID": sessionID])
    }

    func handleIMECancel(_ sessionID: String) {
        post(Self.imeCancelName, userInfo: ["sessionID": sessionID])
    }

    func handleIMEUserTyped(_ sessionID: String, keyCode: Int32, characters: String) {
        post(
            Self.imeUserTypedName,
            userInfo: [
                "sessionID": sessionID,
                "keyCode": Int(keyCode),
                "characters": characters,
            ]
        )
    }

    func handleIMEContextLost(_ sessionID: String, hadMarkedText: Bool) {
        post(
            Self.imeContextLostName,
            userInfo: [
                "sessionID": sessionID,
                "hadMarkedText": hadMarkedText,
            ]
        )
    }

    func handleClearSession(_ sessionID: String) {}
    func handleSetMarkedText(_ sessionID: String, text: String) {}
    func handleCommitText(_ sessionID: String, text: String, submit: Bool) {}
    func handleCancelInput(_ sessionID: String) {}
    func handleStopDictating(_ sessionID: String) {}
}

/// Communicates with the helper broker process via XPC.
final class BeeInputClient: Sendable {
    private static let brokerServiceName = "fasterthanlime.bee.broker"
    private static let brokerLaunchLabel = "fasterthanlime.bee.broker"
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"
    private static let appInstanceID = UUID().uuidString

    nonisolated(unsafe) private static var previousInputSource: TISInputSource?
    nonisolated(unsafe) private static var xpcConnection: NSXPCConnection?
    nonisolated(unsafe) private static var appControlSink = BeeAppControlSink()
    nonisolated(unsafe) private static var helloSent = false
    nonisolated(unsafe) private static var brokerLaunchAttempted = false
    private static let xpcLock = NSLock()

    init() {
        Self.ensureBrokerLaunchdService()
        Self.sendHelloIfNeeded()
    }

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
            beeLog(
                "IME ACTIVATE: prepareSession failed for session=\(sessionID.uuidString.prefix(8))")
            return false
        }

        // Select the bee input source (on cold start this also launches the
        // IME process), then wait for the IME to be connected to the broker.
        let selected = await Self.selectBeeInputSource()
        guard selected else {
            await Self.clearSessionXPC(sessionID: sessionID)
            return false
        }

        let imeReady = await Self.waitForIMEXPC()
        if !imeReady {
            beeLog("IME ACTIVATE: waitForIME failed")
        }

        beeLog(
            "IME ACTIVATE: selection done id=\(sessionID.uuidString.prefix(8)) activationID=\(activationID.prefix(8)) targetPID=\(targetPID.map(String.init) ?? "nil"), waiting for IME confirm event"
        )
        return true
    }

    @MainActor
    private static func selectBeeInputSource() async -> Bool {
        guard let beeSource = findBeeInputSource() else {
            beeLog("IME ACTIVATE: bee input source NOT FOUND")
            return false
        }

        if let current = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue(),
            !isBeeInputSource(current)
        {
            previousInputSource = current
        }

        // Deferred Selection: give the target app time to finish its
        // responder chain update before we switch input sources.
        try? await Task.sleep(for: .milliseconds(20))

        let result = TISSelectInputSource(beeSource)
        beeLog("IME ACTIVATE: TISSelectInputSource result=\(result)")
        guard result == noErr else {
            return false
        }

        // Simulated Event: post a Shift key-up to force the target app's
        // NSTextInputContext to notice the new input source.
        let src = CGEventSource(stateID: .hidSystemState)
        if let shiftUp = CGEvent(keyboardEventSource: src, virtualKey: UInt16(kVK_Shift), keyDown: false) {
            shiftUp.post(tap: .cghidEventTap)
        }

        return true
    }

    func deactivate() {
        Self.switchAwayFromBeeInputIfNeeded()
    }

    // MARK: - IME Commands

    func setMarkedText(_ text: String, sessionID: UUID) {
        Self.setMarkedTextXPC(text, sessionID: sessionID)
    }

    func logSetMarkedText(_ text: String, sessionID: UUID) {
        beeLog("IME setMarkedText: \(text.prefix(60).debugDescription)")
        setMarkedText(text, sessionID: sessionID)
    }

    func commitText(_ text: String, sessionID: UUID) {
        Self.commitTextXPC(text, submit: false, sessionID: sessionID)
    }

    func clearMarkedText(sessionID: UUID) {
        Self.cancelInputXPC(sessionID: sessionID)
    }

    func stopDictating(sessionID: UUID) {
        Self.stopDictatingXPC(sessionID: sessionID)
    }

    func simulateReturn() {
        let src = CGEventSource(stateID: .hidSystemState)
        if let down = CGEvent(keyboardEventSource: src, virtualKey: 0x24, keyDown: true),
            let up = CGEvent(keyboardEventSource: src, virtualKey: 0x24, keyDown: false)
        {
            down.post(tap: .cghidEventTap)
            usleep(10_000)  // 10ms
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
        return xpcLock.withLock {
            if let connection = xpcConnection {
                return connection
            }
            let connection = NSXPCConnection(machServiceName: brokerServiceName, options: [])
            connection.remoteObjectInterface = NSXPCInterface(with: BeeBrokerXPC.self)
            connection.exportedInterface = NSXPCInterface(with: BeeBrokerPeerXPC.self)
            connection.exportedObject = appControlSink
            connection.resume()
            xpcConnection = connection
            return connection
        }
    }

    private static func invalidateXPCConnection() {
        xpcLock.withLock {
            xpcConnection?.invalidate()
            xpcConnection = nil
            helloSent = false
        }
    }

    private static func ensureBrokerLaunchdService() {
        let shouldAttempt = xpcLock.withLock { () -> Bool in
            if brokerLaunchAttempted {
                return false
            }
            brokerLaunchAttempted = true
            return true
        }
        guard shouldAttempt else { return }

        let uid = getuid()
        let domain = "gui/\(uid)"
        let service = "\(domain)/\(brokerLaunchLabel)"

        // // First try to kickstart an already-bootstrapped service.
        // let kickStatus = runLaunchctl(args: ["kickstart", "-k", service])
        // if kickStatus == 0 {
        //     beeLog("BROKER launchd: kickstart ok service=\(service)")
        //     return
        // }

        // // If not bootstrapped yet, bootstrap from the per-user LaunchAgent plist.
        // let plistPath = NSHomeDirectory() + "/Library/LaunchAgents/\(brokerLaunchLabel).plist"
        // if FileManager.default.fileExists(atPath: plistPath) {
        //     _ = runLaunchctl(args: ["bootstrap", domain, plistPath])
        //     let retryStatus = runLaunchctl(args: ["kickstart", "-k", service])
        //     if retryStatus == 0 {
        //         beeLog("BROKER launchd: bootstrap+kickstart ok service=\(service)")
        //         return
        //     }
        // }

        beeLog("BROKER launchd: unable to start service=\(service)")
    }

    @discardableResult
    private static func runLaunchctl(args: [String]) -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/bin/launchctl")
        process.arguments = args
        do {
            try process.run()
            process.waitUntilExit()
            return process.terminationStatus
        } catch {
            beeLog(
                "BROKER launchd: launchctl failed args=\(args.joined(separator: " ")) error=\(error.localizedDescription)"
            )
            return -1
        }
    }

    private static func prepareSessionXPC(sessionID: UUID, targetPID: pid_t?, activationID: String)
        async -> Bool
    {
        sendHelloIfNeeded()
        let connection = getXPCConnection()
        return await withCheckedContinuation { continuation in
            let proxy =
                connection.remoteObjectProxyWithErrorHandler { error in
                    beeLog("BROKER XPC prepareSession error: \(error.localizedDescription)")
                    invalidateXPCConnection()
                    continuation.resume(returning: false)
                } as? BeeBrokerXPC

            guard let proxy else {
                continuation.resume(returning: false)
                return
            }

            proxy.prepareSession(
                sessionID.uuidString,
                targetPID: targetPID ?? -1,
                activationID: activationID,
                appInstanceID: appInstanceID
            ) { ok in
                continuation.resume(returning: ok)
            }
        }
    }

    private static func waitForIMEXPC() async -> Bool {
        let connection = getXPCConnection()
        return await withCheckedContinuation { continuation in
            let proxy =
                connection.remoteObjectProxyWithErrorHandler { error in
                    beeLog("BROKER XPC waitForIME error: \(error.localizedDescription)")
                    invalidateXPCConnection()
                    continuation.resume(returning: false)
                } as? BeeBrokerXPC

            guard let proxy else {
                continuation.resume(returning: false)
                return
            }

            proxy.waitForIME(appInstanceID: appInstanceID) { ok in
                continuation.resume(returning: ok)
            }
        }
    }

    private static func clearSessionXPC(sessionID: UUID) async {
        let connection = getXPCConnection()
        await withCheckedContinuation { continuation in
            let proxy =
                connection.remoteObjectProxyWithErrorHandler { error in
                    beeLog("BROKER XPC clearSession error: \(error.localizedDescription)")
                    invalidateXPCConnection()
                    continuation.resume()
                } as? BeeBrokerXPC

            guard let proxy else {
                continuation.resume()
                return
            }

            proxy.clearSession(sessionID.uuidString, appInstanceID: appInstanceID) {
                continuation.resume()
            }
        }
    }

    private static func sendHelloIfNeeded() {
        let shouldSend = xpcLock.withLock { () -> Bool in
            if helloSent {
                return false
            }
            helloSent = true
            return true
        }
        guard shouldSend else { return }

        let connection = getXPCConnection()
        let proxy =
            connection.remoteObjectProxyWithErrorHandler { error in
                beeLog("BROKER XPC appHello error: \(error.localizedDescription)")
                invalidateXPCConnection()
            } as? BeeBrokerXPC
        proxy?.appHello(appInstanceID) { ok in
            if !ok {
                beeLog("BROKER XPC appHello rejected")
            }
        }
    }

    private static func setMarkedTextXPC(_ text: String, sessionID: UUID) {
        Task.detached {
            await withCheckedContinuation { continuation in
                let connection = getXPCConnection()
                let proxy =
                    connection.remoteObjectProxyWithErrorHandler { error in
                        beeLog("BROKER XPC setMarkedText error: \(error.localizedDescription)")
                        invalidateXPCConnection()
                        continuation.resume()
                    } as? BeeBrokerXPC

                guard let proxy else {
                    continuation.resume()
                    return
                }

                proxy.setMarkedText(sessionID.uuidString, text: text, appInstanceID: appInstanceID)
                { _ in
                    continuation.resume()
                }
            }
        }
    }

    private static func commitTextXPC(_ text: String, submit: Bool, sessionID: UUID) {
        Task.detached {
            await withCheckedContinuation { continuation in
                let connection = getXPCConnection()
                let proxy =
                    connection.remoteObjectProxyWithErrorHandler { error in
                        beeLog("BROKER XPC commitText error: \(error.localizedDescription)")
                        invalidateXPCConnection()
                        continuation.resume()
                    } as? BeeBrokerXPC

                guard let proxy else {
                    continuation.resume()
                    return
                }

                proxy.commitText(
                    sessionID.uuidString, text: text, submit: submit, appInstanceID: appInstanceID
                ) { _ in
                    continuation.resume()
                }
            }
        }
    }

    private static func cancelInputXPC(sessionID: UUID) {
        Task.detached {
            await withCheckedContinuation { continuation in
                let connection = getXPCConnection()
                let proxy =
                    connection.remoteObjectProxyWithErrorHandler { error in
                        beeLog("BROKER XPC cancelInput error: \(error.localizedDescription)")
                        invalidateXPCConnection()
                        continuation.resume()
                    } as? BeeBrokerXPC

                guard let proxy else {
                    continuation.resume()
                    return
                }

                proxy.cancelInput(sessionID.uuidString, appInstanceID: appInstanceID) { _ in
                    continuation.resume()
                }
            }
        }
    }

    private static func stopDictatingXPC(sessionID: UUID) {
        Task.detached {
            await withCheckedContinuation { continuation in
                let connection = getXPCConnection()
                let proxy =
                    connection.remoteObjectProxyWithErrorHandler { error in
                        beeLog("BROKER XPC stopDictating error: \(error.localizedDescription)")
                        invalidateXPCConnection()
                        continuation.resume()
                    } as? BeeBrokerXPC

                guard let proxy else {
                    continuation.resume()
                    return
                }

                proxy.stopDictating(sessionID.uuidString, appInstanceID: appInstanceID) { _ in
                    continuation.resume()
                }
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
            isBeeInputSource(current)
        else {
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
            !isBeeInputSource(ascii)
        {
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
            kTISPropertyInputSourceIsSelectCapable: true
        ]
        return
            (TISCreateInputSourceList(properties as CFDictionary, false)?
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
            kTISPropertyBundleID: beeBundleID as CFString
        ]
        guard
            let sources = TISCreateInputSourceList(properties as CFDictionary, false)?
                .takeRetainedValue() as? [TISInputSource],
            let source = sources.first
        else {
            return nil
        }
        return source
    }
}
