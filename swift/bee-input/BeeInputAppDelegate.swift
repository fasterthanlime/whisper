import Cocoa
import InputMethodKit

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
    private static let imeSessionStartedName = NSNotification.Name("fasterthanlime.bee.imeSessionStarted")

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

        let dnc = DistributedNotificationCenter.default()

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.setMarkedText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String,
                  let sessionID = Self.sessionID(from: notification) else { return }
            let hasCtrl = BeeXPCService.shared.controller != nil
            beeInputLog(
                "RECV setMarkedText: \(text.prefix(40).debugDescription) hasController=\(hasCtrl) session=\(sessionID.uuidString.prefix(8))"
            )
            BeeXPCService.shared.setMarkedText(text, sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.commitText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String,
                  let sessionID = Self.sessionID(from: notification) else { return }
            let submit = notification.userInfo?["submit"] as? Bool ?? false
            BeeXPCService.shared.commitText(text, submit: submit, sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.cancelInput"),
            object: nil, queue: .main
        ) { notification in
            guard let sessionID = Self.sessionID(from: notification) else { return }
            BeeXPCService.shared.cancelInput(sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.stopDictating"),
            object: nil, queue: .main
        ) { notification in
            guard let sessionID = Self.sessionID(from: notification) else { return }
            BeeXPCService.shared.stopDictating(sessionID: sessionID)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.setSessionContext"),
            object: nil, queue: .main
        ) { notification in
            guard let sessionID = Self.sessionID(from: notification) else { return }
            BeeXPCService.shared.setSessionContext(sessionID: sessionID)
            Self.postSessionStartedIfReady()
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
        guard let sessionID = BeeXPCService.shared.consumeSessionStartAcknowledgementIfReady() else {
            return
        }
        DistributedNotificationCenter.default().postNotificationName(
            imeSessionStartedName,
            object: nil,
            userInfo: ["sessionID": sessionID.uuidString],
            deliverImmediately: true
        )
        beeInputLog("imeSessionStarted: session=\(sessionID.uuidString.prefix(8))")
    }
}
