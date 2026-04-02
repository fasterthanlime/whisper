import Foundation

private func brokerLog(_ msg: String) {
    let ts = ProcessInfo.processInfo.systemUptime
    let line = String(format: "[%.3f] BROKER: %@\n", ts, msg)
    if let data = line.data(using: .utf8),
        let fh = FileHandle(forWritingAtPath: "/tmp/bee.log")
    {
        fh.seekToEndOfFile()
        fh.write(data)
        fh.closeFile()
    } else if let data = line.data(using: .utf8) {
        try? data.write(to: URL(fileURLWithPath: "/tmp/bee.log"))
    }
}

private final class BeeBrokerService: NSObject, BeeBrokerXPC {
    struct SessionInfo {
        var id: String
        var appInstanceID: String
    }

    enum SessionState {
        case idle
        case prepared(SessionInfo)
        case claimed(SessionInfo)
    }

    private let queue = DispatchQueue(label: "fasterthanlime.bee.broker.state")
    private var appConnections: [String: NSXPCConnection] = [:]
    private var imeConnections: [String: NSXPCConnection] = [:]
    private var session: SessionState = .idle
    private var activeIMEInstanceID: String?
    private var imeWaiters: [(Bool) -> Void] = []

    private func appProxy(_ appInstanceID: String) -> BeeBrokerPeerXPC? {
        guard let conn = appConnections[appInstanceID] else { return nil }
        return conn.remoteObjectProxyWithErrorHandler { error in
            brokerLog("app callback error: \(error.localizedDescription)")
        } as? BeeBrokerPeerXPC
    }

    private func imeProxy() -> BeeBrokerPeerXPC? {
        guard let imeID = activeIMEInstanceID, let conn = imeConnections[imeID] else { return nil }
        return conn.remoteObjectProxyWithErrorHandler { error in
            brokerLog("ime callback error: \(error.localizedDescription)")
        } as? BeeBrokerPeerXPC
    }

    func appHello(_ appInstanceID: String, withReply reply: @escaping (Bool) -> Void) {
        guard let conn = NSXPCConnection.current() else {
            reply(false)
            return
        }
        queue.async {
            self.appConnections[appInstanceID] = conn
            brokerLog("appHello: id=\(appInstanceID.prefix(8))")
            reply(true)
        }
    }

    func imeHello(_ imeInstanceID: String, withReply reply: @escaping (Bool) -> Void) {
        guard let conn = NSXPCConnection.current() else {
            reply(false)
            return
        }
        queue.async {
            self.imeConnections[imeInstanceID] = conn
            self.activeIMEInstanceID = imeInstanceID
            brokerLog(
                "imeHello: id=\(imeInstanceID.prefix(8)) flushing \(self.imeWaiters.count) waiter(s)"
            )
            let waiters = self.imeWaiters
            self.imeWaiters.removeAll()
            for waiter in waiters {
                waiter(true)
            }
            reply(true)
        }
    }

    func waitForIME(appInstanceID: String, withReply reply: @escaping (Bool) -> Void) {
        queue.async {
            if self.activeIMEInstanceID != nil {
                brokerLog("waitForIME: IME already connected")
                reply(true)
            } else {
                brokerLog("waitForIME: IME not connected, queuing waiter")
                self.imeWaiters.append(reply)
            }
        }
    }

    func prepareSession(
        _ sessionID: String,
        activationID: String,
        appInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    ) {
        guard let conn = NSXPCConnection.current() else {
            reply(false)
            return
        }
        queue.async {
            self.appConnections[appInstanceID] = conn
            let info = SessionInfo(id: sessionID, appInstanceID: appInstanceID)
            self.session = .prepared(info)
            brokerLog("prepareSession: id=\(sessionID.prefix(8))")
            if let ime = self.imeProxy() {
                ime.handleNewPreparedSession(sessionID)
            }
            reply(true)
        }
    }

    func claimPreparedSession(
        imeInstanceID: String,
        withReply reply: @escaping (Bool, String) -> Void
    ) {
        guard let conn = NSXPCConnection.current() else {
            reply(false, "")
            return
        }
        queue.async {
            self.imeConnections[imeInstanceID] = conn
            self.activeIMEInstanceID = imeInstanceID

            guard case .prepared(let info) = self.session else {
                reply(false, "")
                return
            }
            self.session = .claimed(info)
            brokerLog("claimPreparedSession: session=\(info.id.prefix(8))")
            reply(true, info.id)
        }
    }

    /// Get the app proxy for the current session, if any.
    private func sessionAppProxy() -> (SessionInfo, BeeBrokerPeerXPC)? {
        switch session {
        case .claimed(let info):
            guard let app = appProxy(info.appInstanceID) else { return nil }
            return (info, app)
        default:
            return nil
        }
    }

    func clearSession(
        _ sessionID: String, appInstanceID: String, withReply reply: @escaping () -> Void
    ) {
        queue.async {
            if let ime = self.imeProxy() {
                ime.handleClearSession(sessionID)
            }
            self.session = .idle
            reply()
        }
    }

    func setMarkedText(
        _ sessionID: String,
        text: String,
        appInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    ) {
        queue.async {
            guard let ime = self.imeProxy() else {
                reply(false)
                return
            }
            ime.handleSetMarkedText(sessionID, text: text)
            reply(true)
        }
    }

    func commitText(
        _ sessionID: String,
        text: String,
        submit: Bool,
        appInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    ) {
        queue.async {
            guard let ime = self.imeProxy() else {
                reply(false)
                return
            }
            ime.handleCommitText(sessionID, text: text, submit: submit)
            self.session = .idle
            reply(true)
        }
    }

    func cancelInput(
        _ sessionID: String, appInstanceID: String, withReply reply: @escaping (Bool) -> Void
    ) {
        queue.async {
            guard let ime = self.imeProxy() else {
                reply(false)
                return
            }
            ime.handleCancelInput(sessionID)
            self.session = .idle
            reply(true)
        }
    }

    func stopDictating(
        _ sessionID: String, appInstanceID: String, withReply reply: @escaping (Bool) -> Void
    ) {
        queue.async {
            guard let ime = self.imeProxy() else {
                reply(false)
                return
            }
            ime.handleStopDictating(sessionID)
            self.session = .idle
            reply(true)
        }
    }

    func imeAttach(
        _ sessionID: String,
        imeInstanceID: String,
        withReply reply: @escaping (Bool) -> Void
    ) {
        guard let conn = NSXPCConnection.current() else {
            reply(false)
            return
        }
        queue.async {
            self.imeConnections[imeInstanceID] = conn
            self.activeIMEInstanceID = imeInstanceID
            guard case .claimed(let info) = self.session,
                  info.id == sessionID,
                  let app = self.appProxy(info.appInstanceID)
            else {
                reply(false)
                return
            }
            brokerLog("imeAttach: session=\(sessionID.prefix(8))")
            app.handleIMESessionStarted(sessionID)
            reply(true)
        }
    }

    func imeSubmit(
        _ sessionID: String, imeInstanceID: String, withReply reply: @escaping () -> Void
    ) {
        queue.async {
            if let (_, app) = self.sessionAppProxy() { app.handleIMESubmit(sessionID) }
            reply()
        }
    }

    func imeCancel(
        _ sessionID: String, imeInstanceID: String, withReply reply: @escaping () -> Void
    ) {
        queue.async {
            if let (_, app) = self.sessionAppProxy() { app.handleIMECancel(sessionID) }
            reply()
        }
    }

    func imeUserTyped(
        _ sessionID: String,
        keyCode: Int32,
        characters: String,
        imeInstanceID: String,
        withReply reply: @escaping () -> Void
    ) {
        queue.async {
            if let (_, app) = self.sessionAppProxy() {
                app.handleIMEUserTyped(sessionID, keyCode: keyCode, characters: characters)
            }
            reply()
        }
    }

    func imeContextLost(
        _ sessionID: String,
        hadMarkedText: Bool,
        imeInstanceID: String,
        withReply reply: @escaping () -> Void
    ) {
        queue.async {
            if let (_, app) = self.sessionAppProxy() {
                app.handleIMEContextLost(sessionID, hadMarkedText: hadMarkedText)
            }
            reply()
        }
    }
}

private final class BeeBrokerDelegate: NSObject, NSXPCListenerDelegate {
    private let service = BeeBrokerService()

    func listener(
        _ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection
    ) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: BeeBrokerXPC.self)
        newConnection.exportedObject = service
        newConnection.remoteObjectInterface = NSXPCInterface(with: BeeBrokerPeerXPC.self)
        newConnection.resume()
        brokerLog("accepted new connection")
        return true
    }
}

let machService = "fasterthanlime.bee.broker"
let listener = NSXPCListener(machServiceName: machService)
private let delegate = BeeBrokerDelegate()
listener.delegate = delegate
listener.resume()
brokerLog("broker listening machService=\(machService)")
RunLoop.main.run()
