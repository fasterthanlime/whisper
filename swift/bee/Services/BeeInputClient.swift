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

    func handleIMESessionStarted(_ sessionID: String) {
        post(Self.imeSessionStartedName, userInfo: ["sessionID": sessionID])
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

    func handleNewPreparedSession(_ sessionID: String, targetPID: Int32) {}
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
            activationID: activationID,
            targetPID: Int32(targetPID ?? 0)
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

        beeLog(
            "IME ACTIVATE: selection done id=\(sessionID.uuidString.prefix(8)) activationID=\(activationID.prefix(8)), waiting for IME confirm event"
        )
        return true
    }

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
        beeLog("TIS SELECT: \(inputSourceID(beeSource)) (activate) result=\(result)")
        guard result == noErr else {
            return false
        }

        return true
    }

    /// Force a focus cycle: hide the target app briefly, then reactivate it.
    /// This creates a real focus loss/gain that makes the OS call activateServer.
    @MainActor
    static func forceFocusCycle() {
        guard let targetApp = NSWorkspace.shared.frontmostApplication else { return }

        let env = ProcessInfo.processInfo.environment
        let hideDelayMS = UInt32(env["BEE_FOCUS_CYCLE_HIDE_DELAY_MS"] ?? "500") ?? 0
        let unhideDelayMS = UInt32(env["BEE_FOCUS_CYCLE_UNHIDE_DELAY_MS"] ?? "500") ?? 0
        let activateDelayMS = UInt32(env["BEE_FOCUS_CYCLE_ACTIVATE_DELAY_MS"] ?? "500") ?? 500

        beeLog("IME ACTIVATE: focus cycle — hiding \(targetApp.localizedName ?? "?")")
        targetApp.hide()
        if hideDelayMS > 0 {
            usleep(hideDelayMS * 1_000)
        }

        beeLog("IME ACTIVATE: focus cycle — reactivating \(targetApp.localizedName ?? "?")")
        targetApp.unhide()
        if unhideDelayMS > 0 {
            usleep(unhideDelayMS * 1_000)
        }

        targetApp.activate()
        if activateDelayMS > 0 {
            usleep(activateDelayMS * 1_000)
        }
    }

    @MainActor
    static func stealthFocusCycle() {
        let previousApp = NSWorkspace.shared.frontmostApplication
        let appName = previousApp?.localizedName ?? "?"
        let appPID = previousApp?.processIdentifier ?? 0
        beeLog("IME ACTIVATE: stealth focus cycle start frontmost=\(appName) pid=\(appPID)")

        let panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 1, height: 1),
            styleMask: [],
            backing: .buffered,
            defer: false
        )
        panel.isFloatingPanel = true
        panel.level = .floating
        panel.alphaValue = 0.0

        // Add a text field so the text input system fully engages
        let textField = NSTextField(frame: NSRect(x: 0, y: 0, width: 1, height: 1))
        panel.contentView?.addSubview(textField)

        beeLog("IME ACTIVATE: stealth focus cycle activating bee + panel")
        NSApp.activate()
        panel.makeKeyAndOrderFront(nil)
        panel.makeFirstResponder(textField)

        if let beeSource = findBeeInputSource() {
            let r = TISSelectInputSource(beeSource)
            beeLog("TIS SELECT: \(inputSourceID(beeSource)) (stealth panel activate) result=\(r)")
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            beeLog("IME ACTIVATE: stealth focus cycle ordering panel out")
            panel.orderOut(nil)

            if let previousApp {
                beeLog(
                    "IME ACTIVATE: stealth focus cycle reactivating app name=\(previousApp.localizedName ?? "?") pid=\(previousApp.processIdentifier)"
                )
                previousApp.activate()
                if let beeSource = Self.findBeeInputSource() {
                    let r = TISSelectInputSource(beeSource)
                    beeLog("TIS SELECT: \(Self.inputSourceID(beeSource)) (stealth panel reactivate) result=\(r)")
                }
            } else {
                beeLog("IME ACTIVATE: stealth focus cycle no previous app to reactivate")
            }
        }
    }

    /// Cycle input sources: select a non-bee source, then re-select bee.
    /// Forces the Text Input Management system to tear down and re-create the
    /// IME connection, triggering activateServer.
    @MainActor
    static func tisToggleCycle() {
        guard let beeSource = findBeeInputSource() else {
            beeLog("IME ACTIVATE: TIS toggle — bee source not found")
            return
        }
        guard let other = findKeyboardLayout() else {
            beeLog("IME ACTIVATE: TIS toggle — no keyboard layout found")
            return
        }

        let otherID = inputSourceID(other)
        let beeID = inputSourceID(beeSource)
        let awayResult = TISSelectInputSource(other)
        beeLog("TIS SELECT: \(otherID) (toggle away) result=\(awayResult)")
        usleep(200_000)
        let backResult = TISSelectInputSource(beeSource)
        beeLog("TIS SELECT: \(beeID) (toggle back) result=\(backResult)")
    }

    /// Find an actual keyboard layout (not an IME or palette) to use for TIS toggling.
    private static func findKeyboardLayout() -> TISInputSource? {
        // Prefer the current ASCII-capable keyboard layout
        if let ascii = TISCopyCurrentASCIICapableKeyboardLayoutInputSource()?.takeRetainedValue(),
            !isBeeInputSource(ascii)
        {
            return ascii
        }

        // Fall back to any keyboard layout that isn't bee
        let props: [CFString: Any] = [
            kTISPropertyInputSourceCategory: kTISCategoryKeyboardInputSource,
            kTISPropertyInputSourceIsSelectCapable: true,
        ]
        guard
            let sources = TISCreateInputSourceList(props as CFDictionary, false)?
                .takeRetainedValue() as? [TISInputSource]
        else { return nil }
        return sources.first(where: { !isBeeInputSource($0) })
    }

    /// Nudge the focused UI element via Accessibility to trigger IME re-activation.
    /// Clears focus then restores it, forcing the input context to reconnect.
    @MainActor
    static func axNudgeFocus() {
        let systemWide = AXUIElementCreateSystemWide()
        var focusedRef: AnyObject?
        let err = AXUIElementCopyAttributeValue(
            systemWide, kAXFocusedUIElementAttribute as CFString, &focusedRef)
        guard err == .success, let focused = focusedRef else {
            beeLog("IME ACTIVATE: AX nudge — no focused element (err=\(err.rawValue))")
            return
        }

        let element = focused as! AXUIElement

        // Get the app that owns this element
        var pidValue: pid_t = 0
        AXUIElementGetPid(element, &pidValue)

        guard let app = NSRunningApplication(processIdentifier: pidValue) else {
            beeLog("IME ACTIVATE: AX nudge — can't find app for pid=\(pidValue)")
            return
        }

        let appElement = AXUIElementCreateApplication(pidValue)

        // Clear focused element, brief pause, restore it
        beeLog("IME ACTIVATE: AX nudge — clearing focus pid=\(pidValue)")
        AXUIElementSetAttributeValue(appElement, kAXFocusedUIElementAttribute as CFString, kCFNull)
        usleep(30_000)  // 30ms
        beeLog("IME ACTIVATE: AX nudge — restoring focus")
        AXUIElementSetAttributeValue(appElement, kAXFocusedUIElementAttribute as CFString, element)
    }

    func deactivate(caller: String = #function, file: String = #fileID, line: Int = #line) {
        beeLog("IME DEACTIVATE called from \(file):\(line) \(caller)")
        Self.switchAwayFromBeeInputIfNeeded()
    }

    /// Wait for the IME to connect to the broker.
    static func waitForIMEReady() async -> Bool {
        await waitForIMEXPC()
    }

    // MARK: - IME Commands

    func setMarkedText(_ text: String, sessionID: UUID) {
        beeLog("setMarkedText → broker session=\(sessionID.uuidString.prefix(8))")
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

    static func restoreInputSourceIfNeeded(
        caller: String = #function, file: String = #fileID, line: Int = #line
    ) {
        beeLog("IME RESTORE called from \(file):\(line) \(caller)")
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

        // beeLog("BROKER launchd: unable to start service=\(service)")
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

    private static func prepareSessionXPC(sessionID: UUID, activationID: String, targetPID: Int32)
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
                activationID: activationID,
                targetPID: targetPID,
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
            let id = inputSourceID(previous)
            let result = TISSelectInputSource(previous)
            beeLog("TIS SELECT: \(id) (restore previous) result=\(result)")
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
            beeLog("TIS SELECT: no fallback input source available")
            return
        }

        let id = inputSourceID(fallback)
        let result = TISSelectInputSource(fallback)
        beeLog("TIS SELECT: \(id) (fallback) result=\(result)")
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

    private static func inputSourceID(_ source: TISInputSource) -> String {
        guard let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else {
            return "<unknown>"
        }
        return Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String
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
