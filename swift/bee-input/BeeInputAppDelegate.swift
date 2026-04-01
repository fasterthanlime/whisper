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

    func applicationDidFinishLaunching(_ notification: Notification) {
        server = IMKServer(
            name: Bundle.main.infoDictionary!["InputMethodConnectionName"] as! String,
            bundleIdentifier: Bundle.main.bundleIdentifier!
        )

        let dnc = DistributedNotificationCenter.default()

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.setMarkedText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String else { return }
            let hasCtrl = BeeXPCService.shared.controller != nil
            beeInputLog("RECV setMarkedText: \(text.prefix(40).debugDescription) hasController=\(hasCtrl)")
            BeeXPCService.shared.setMarkedText(text)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.commitText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String else { return }
            let submit = notification.userInfo?["submit"] as? Bool ?? false
            BeeXPCService.shared.commitText(text, submit: submit)
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.cancelInput"),
            object: nil, queue: .main
        ) { _ in
            BeeXPCService.shared.cancelInput()
        }

        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.bee.stopDictating"),
            object: nil, queue: .main
        ) { _ in
            BeeXPCService.shared.isDictating = false
            BeeXPCService.shared.pendingText = nil
        }
    }
}
