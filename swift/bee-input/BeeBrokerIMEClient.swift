import Foundation

final class BeeBrokerIMEClient {
    static let shared = BeeBrokerIMEClient()

    private let brokerServiceName = "fasterthanlime.bee.broker"
    private let imeInstanceID = UUID().uuidString
    private let lock = NSLock()
    private var connection: NSXPCConnection?
    private var started = false
    private let callbackSink = BeeIMEPeerSink()

    private init() {}

    func start() {
        let shouldStart = lock.withLock { () -> Bool in
            if started { return false }
            started = true
            return true
        }
        guard shouldStart else { return }
        sendHello()
    }

    private func getConnection() -> NSXPCConnection {
        lock.withLock {
            if let connection {
                return connection
            }
            let conn = NSXPCConnection(machServiceName: brokerServiceName, options: [])
            conn.remoteObjectInterface = NSXPCInterface(with: BeeBrokerXPC.self)
            conn.exportedInterface = NSXPCInterface(with: BeeBrokerPeerXPC.self)
            conn.exportedObject = callbackSink
            conn.resume()
            connection = conn
            return conn
        }
    }

    private func invalidateConnection() {
        lock.withLock {
            connection?.invalidate()
            connection = nil
        }
    }

    private func sendHello() {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { error in
            beeInputLog("BROKER imeHello error: \(error.localizedDescription)")
            self.invalidateConnection()
        } as? BeeBrokerXPC
        proxy?.imeHello(imeInstanceID) { ok in
            if !ok {
                beeInputLog("BROKER imeHello rejected")
            } else {
                beeInputLog("BROKER imeHello ok id=\(self.imeInstanceID.prefix(8))")
            }
        }
    }

    func imeAttach(sessionID: UUID, clientPID: pid_t?, clientID: String?) {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { error in
            beeInputLog("BROKER imeAttach error: \(error.localizedDescription)")
            self.invalidateConnection()
        } as? BeeBrokerXPC
        proxy?.imeAttach(
            sessionID.uuidString,
            clientPID: clientPID.map { Int32($0) } ?? -1,
            clientID: clientID ?? "",
            imeInstanceID: imeInstanceID
        ) { ok in
            if !ok {
                beeInputLog("BROKER imeAttach rejected session=\(sessionID.uuidString.prefix(8))")
            }
        }
    }

    /// Synchronous (blocking) claim — use from activateServer to prevent
    /// deactivateServer from racing during the XPC round-trip.
    func claimPreparedSessionSync(
        clientPID: pid_t?,
        clientID: String?
    ) -> (found: Bool, sessionID: UUID?, targetPID: pid_t?, activationID: String?) {
        let conn = getConnection()
        let proxy = conn.synchronousRemoteObjectProxyWithErrorHandler { error in
            beeInputLog("BROKER claimPreparedSession error: \(error.localizedDescription)")
            self.invalidateConnection()
        } as? BeeBrokerXPC

        guard let proxy else {
            return (false, nil, nil, nil)
        }

        var resultFound = false
        var resultSessionID: UUID?
        var resultTargetPID: pid_t?
        var resultActivationID: String?

        proxy.claimPreparedSession(
            clientPID: clientPID.map { Int32($0) } ?? -1,
            clientID: clientID ?? "",
            imeInstanceID: imeInstanceID
        ) { found, sessionIDRaw, targetPIDRaw, activationID in
            guard found, let sessionID = UUID(uuidString: sessionIDRaw) else {
                return
            }
            resultFound = true
            resultSessionID = sessionID
            resultTargetPID = targetPIDRaw >= 0 ? pid_t(targetPIDRaw) : nil
            resultActivationID = activationID.isEmpty ? nil : activationID
        }

        return (resultFound, resultSessionID, resultTargetPID, resultActivationID)
    }

    func imeSubmit(sessionID: UUID) {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { _ in } as? BeeBrokerXPC
        proxy?.imeSubmit(sessionID.uuidString, imeInstanceID: imeInstanceID) {}
    }

    func imeCancel(sessionID: UUID) {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { _ in } as? BeeBrokerXPC
        proxy?.imeCancel(sessionID.uuidString, imeInstanceID: imeInstanceID) {}
    }

    func imeUserTyped(sessionID: UUID, keyCode: UInt16, characters: String) {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { _ in } as? BeeBrokerXPC
        proxy?.imeUserTyped(
            sessionID.uuidString,
            keyCode: Int32(keyCode),
            characters: characters,
            imeInstanceID: imeInstanceID
        ) {}
    }

    func imeContextLost(sessionID: UUID, hadMarkedText: Bool) {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { _ in } as? BeeBrokerXPC
        proxy?.imeContextLost(
            sessionID.uuidString,
            hadMarkedText: hadMarkedText,
            imeInstanceID: imeInstanceID
        ) {}
    }
}

private final class BeeIMEPeerSink: NSObject, BeeBrokerPeerXPC {
    func handleNewPreparedSession(_ sessionID: String, targetPID: Int32) {
        guard let uuid = UUID(uuidString: sessionID) else { return }
        let pid = targetPID >= 0 ? pid_t(targetPID) : nil
        DispatchQueue.main.async {
            let bridge = BeeIMEBridgeState.shared

            // If we have a live active controller for the right PID, claim now.
            if let controller = bridge.activeController {
                let controllerPID = bridge.activeControllerPID
                if let pid, let controllerPID, pid != controllerPID {
                    beeInputLog("handleNewPreparedSession: PID mismatch controller=\(controllerPID) target=\(pid), storing for activateServer")
                    bridge.storePreparedSession(sessionID: uuid, targetPID: pid)
                    return
                }
                let clientIdentity = bridge.activeClientIdentity
                beeInputLog("handleNewPreparedSession: active controller alive, claiming session=\(sessionID.prefix(8)) directly")
                let result = BeeBrokerIMEClient.shared.claimPreparedSessionSync(
                    clientPID: controllerPID,
                    clientID: clientIdentity
                )
                guard result.found, let claimedSessionID = result.sessionID else {
                    beeInputLog("handleNewPreparedSession: claim failed, storing for activateServer")
                    bridge.storePreparedSession(sessionID: uuid, targetPID: pid)
                    return
                }
                beeInputLog("handleNewPreparedSession: attached session=\(claimedSessionID.uuidString.prefix(8))")
                bridge.attachSession(sessionID: claimedSessionID, clientIdentity: clientIdentity)
                bridge.flushPending()
                BeeBrokerIMEClient.shared.imeAttach(
                    sessionID: claimedSessionID,
                    clientPID: controllerPID,
                    clientID: clientIdentity
                )
                return
            }

            // No live controller — store session for activateServer to pick up.
            beeInputLog("handleNewPreparedSession: no active controller, storing session=\(sessionID.prefix(8)) for activateServer")
            bridge.storePreparedSession(sessionID: uuid, targetPID: pid)
        }
    }

    func handleClearSession(_ sessionID: String) {
        guard let id = UUID(uuidString: sessionID) else { return }
        BeeIMEBridgeState.shared.clearSessionIfMatching(sessionID: id)
    }

    func handleSetMarkedText(_ sessionID: String, text: String) {
        guard let id = UUID(uuidString: sessionID) else { return }
        BeeIMEBridgeState.shared.setMarkedText(text, sessionID: id)
    }

    func handleCommitText(_ sessionID: String, text: String, submit: Bool) {
        guard let id = UUID(uuidString: sessionID) else { return }
        BeeIMEBridgeState.shared.commitText(text, submit: submit, sessionID: id)
    }

    func handleCancelInput(_ sessionID: String) {
        guard let id = UUID(uuidString: sessionID) else { return }
        BeeIMEBridgeState.shared.cancelInput(sessionID: id)
    }

    func handleStopDictating(_ sessionID: String) {
        guard let id = UUID(uuidString: sessionID) else { return }
        BeeIMEBridgeState.shared.stopDictating(sessionID: id)
    }

    func handleIMESessionStarted(_ sessionID: String, clientPID: Int32, clientID: String) {}
    func handleIMESubmit(_ sessionID: String) {}
    func handleIMECancel(_ sessionID: String) {}
    func handleIMEUserTyped(_ sessionID: String, keyCode: Int32, characters: String) {}
    func handleIMEContextLost(_ sessionID: String, hadMarkedText: Bool) {}
}
