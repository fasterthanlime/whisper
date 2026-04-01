import Cocoa
import InputMethodKit

@objc
private protocol BeeIMEControlXPC {
    func prepareSession(_ sessionID: String, targetPID: Int32, activationID: String, withReply reply: @escaping (Bool) -> Void)
    func sessionStatus(_ sessionID: String, withReply reply: @escaping (Bool, Int32, String) -> Void)
    func clearSession(_ sessionID: String, withReply reply: @escaping () -> Void)
}

private final class BeeIMEControlService: NSObject, BeeIMEControlXPC {
    func prepareSession(_ sessionID: String, targetPID: Int32, activationID: String, withReply reply: @escaping (Bool) -> Void) {
        DispatchQueue.main.async {
            guard let sessionID = UUID(uuidString: sessionID) else {
                reply(false)
                return
            }
            let pid: pid_t? = targetPID >= 0 ? pid_t(targetPID) : nil
            BeeIMEBridgeState.shared.prepareSession(
                sessionID: sessionID,
                targetPID: pid,
                activationID: activationID
            )
            reply(true)
        }
    }

    func sessionStatus(_ sessionID: String, withReply reply: @escaping (Bool, Int32, String) -> Void) {
        DispatchQueue.main.async {
            guard let sessionID = UUID(uuidString: sessionID) else {
                reply(false, -1, "")
                return
            }
            let status = BeeIMEBridgeState.shared.sessionStatus(sessionID: sessionID)
            let clientPID: Int32
            if let pid = status.clientPID {
                clientPID = Int32(pid)
            } else {
                clientPID = -1
            }
            reply(status.ready, clientPID, status.clientIdentity ?? "")
        }
    }

    func clearSession(_ sessionID: String, withReply reply: @escaping () -> Void) {
        DispatchQueue.main.async {
            if let sessionID = UUID(uuidString: sessionID) {
                BeeIMEBridgeState.shared.clearSessionIfMatching(sessionID: sessionID)
            }
            reply()
        }
    }
}

func beeInputLog(_ msg: String) {
    let ts = ProcessInfo.processInfo.systemUptime
    let line = String(format: "[%.3f] IME: %@\n", ts, msg)
    if let data = line.data(using: .utf8),
       let fh = FileHandle(forWritingAtPath: "/tmp/bee.log") {
        fh.seekToEndOfFile()
        fh.write(data)
        fh.closeFile()
    } else if let data = line.data(using: .utf8) {
        try? data.write(to: URL(fileURLWithPath: "/tmp/bee.log"))
    }
}

class BeeInputAppDelegate: NSObject, NSApplicationDelegate {
    var server: IMKServer?
    var xpcListener: NSXPCListener?
    private let xpcService = BeeIMEControlService()
    private static let imeSessionStartedName = NSNotification.Name("fasterthanlime.bee.imeSessionStarted")
    private static let xpcMachServiceName = "fasterthanlime.inputmethod.bee.xpc"

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let connectionName = Bundle.main.infoDictionary?["InputMethodConnectionName"] as? String,
              let bundleIdentifier = Bundle.main.bundleIdentifier else {
            beeInputLog("failed to initialize IMKServer: missing bundle metadata")
            return
        }

        server = IMKServer(
            name: connectionName,
            bundleIdentifier: bundleIdentifier
        )

        let listener = NSXPCListener(machServiceName: Self.xpcMachServiceName)
        listener.delegate = self
        listener.resume()
        xpcListener = listener
        beeInputLog("XPC listener started: \(Self.xpcMachServiceName)")

        let dnc = DistributedNotificationCenter.default()

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.setMarkedText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String,
                  let sessionID = Self.sessionID(from: notification) else { return }
            let hasCtrl = BeeIMEBridgeState.shared.controller != nil
            beeInputLog(
                "RECV setMarkedText: \(text.prefix(40).debugDescription) hasController=\(hasCtrl) session=\(sessionID.uuidString.prefix(8))"
            )
            BeeIMEBridgeState.shared.setMarkedText(text, sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.commitText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String,
                  let sessionID = Self.sessionID(from: notification) else { return }
            let submit = notification.userInfo?["submit"] as? Bool ?? false
            BeeIMEBridgeState.shared.commitText(text, submit: submit, sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.cancelInput"),
            object: nil, queue: .main
        ) { notification in
            guard let sessionID = Self.sessionID(from: notification) else { return }
            BeeIMEBridgeState.shared.cancelInput(sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.stopDictating"),
            object: nil, queue: .main
        ) { notification in
            guard let sessionID = Self.sessionID(from: notification) else { return }
            BeeIMEBridgeState.shared.stopDictating(sessionID: sessionID)
        }

    }

    private static func sessionID(from notification: Notification) -> UUID? {
        guard let raw = notification.userInfo?["sessionID"] as? String,
              let sessionID = UUID(uuidString: raw) else {
            beeInputLog("notification missing/invalid sessionID")
            return nil
        }
        return sessionID
    }

    private static func postSessionStartedIfReady() {
        guard let ack = BeeIMEBridgeState.shared.consumeSessionStartAcknowledgementIfReady() else {
            return
        }
        var userInfo: [AnyHashable: Any] = ["sessionID": ack.sessionID.uuidString]
        if let clientPID = ack.clientPID {
            userInfo["clientPID"] = clientPID
        }
        if let clientID = ack.clientIdentity {
            userInfo["clientID"] = clientID
        }
        DistributedNotificationCenter.default().postNotificationName(
            imeSessionStartedName,
            object: nil,
            userInfo: userInfo,
            deliverImmediately: true
        )
        beeInputLog(
            "imeSessionStarted: session=\(ack.sessionID.uuidString.prefix(8)) clientPID=\(ack.clientPID.map(String.init) ?? "nil") clientID=\(ack.clientIdentity ?? "nil")"
        )
    }
}

extension BeeInputAppDelegate: NSXPCListenerDelegate {
    func listener(_ listener: NSXPCListener, shouldAcceptNewConnection newConnection: NSXPCConnection) -> Bool {
        newConnection.exportedInterface = NSXPCInterface(with: BeeIMEControlXPC.self)
        newConnection.exportedObject = xpcService
        newConnection.resume()
        beeInputLog("XPC accepted new connection")
        return true
    }
}
