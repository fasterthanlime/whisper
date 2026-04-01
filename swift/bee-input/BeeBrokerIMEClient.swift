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

    func claimPreparedSession(
        clientPID: pid_t?,
        clientID: String?,
        completion: @escaping (_ found: Bool, _ sessionID: UUID?, _ targetPID: pid_t?, _ activationID: String?) -> Void
    ) {
        let conn = getConnection()
        let proxy = conn.remoteObjectProxyWithErrorHandler { error in
            beeInputLog("BROKER claimPreparedSession error: \(error.localizedDescription)")
            self.invalidateConnection()
            completion(false, nil, nil, nil)
        } as? BeeBrokerXPC
        proxy?.claimPreparedSession(
            clientPID: clientPID.map { Int32($0) } ?? -1,
            clientID: clientID ?? "",
            imeInstanceID: imeInstanceID
        ) { found, sessionIDRaw, targetPIDRaw, activationID in
            guard found, let sessionID = UUID(uuidString: sessionIDRaw) else {
                completion(false, nil, nil, nil)
                return
            }
            let targetPID: pid_t? = targetPIDRaw >= 0 ? pid_t(targetPIDRaw) : nil
            completion(true, sessionID, targetPID, activationID.isEmpty ? nil : activationID)
        }
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
        DispatchQueue.main.async {
            let bridge = BeeIMEBridgeState.shared
            guard let controller = bridge.activeController else {
                beeInputLog("handleNewPreparedSession: no active controller, waiting for activateServer")
                return
            }
            let controllerPID = bridge.activeControllerPID
            let pid = targetPID >= 0 ? pid_t(targetPID) : nil
            if let pid, let controllerPID, pid != controllerPID {
                beeInputLog("handleNewPreparedSession: PID mismatch controller=\(controllerPID) target=\(pid), waiting for activateServer")
                return
            }
            let clientIdentity = bridge.activeClientIdentity
            beeInputLog("handleNewPreparedSession: controller still active, claiming session=\(sessionID.prefix(8)) directly")
            BeeBrokerIMEClient.shared.claimPreparedSession(
                clientPID: controllerPID,
                clientID: clientIdentity
            ) { found, claimedSessionID, _, _ in
                DispatchQueue.main.async {
                    guard found, let claimedSessionID else {
                        beeInputLog("handleNewPreparedSession: claim failed")
                        return
                    }
                    guard bridge.activeController === controller else {
                        beeInputLog("handleNewPreparedSession: controller changed during claim")
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
                }
            }
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
