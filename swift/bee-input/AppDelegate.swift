import Cocoa
import InputMethodKit

class BeeInputAppDelegate: NSObject, NSApplicationDelegate {
    var server: IMKServer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard
            let connectionName = Bundle.main.infoDictionary?["InputMethodConnectionName"]
                as? String,
            let bundleIdentifier = Bundle.main.bundleIdentifier
        else {
            beeInputLog("failed to initialize IMKServer: missing bundle metadata")
            return
        }

        server = IMKServer(
            name: connectionName,
            bundleIdentifier: bundleIdentifier
        )
        beeInputLog("IMK server started")
        AppClientFactory.shared.start()
    }
}

private func beeLogPath() -> String {
    FileManager.default.containerURL(
        forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee"
    )?.appendingPathComponent("bee.log").path ?? "/tmp/bee.log"
}

private func beeTimestamp() -> String {
    let now = Date()
    let cal = Calendar.current
    let dc = cal.dateComponents(
        [.year, .month, .day, .hour, .minute, .second, .nanosecond], from: now)
    return String(
        format: "%04d-%02d-%02dT%02d:%02d:%02d.%06dZ",
        dc.year!, dc.month!, dc.day!, dc.hour!, dc.minute!, dc.second!,
        dc.nanosecond! / 1000)
}

func beeInputLog(_ msg: String) {
    let path = beeLogPath()
    let line = "\(beeTimestamp())  INFO IME: \(msg)\n"
    if let data = line.data(using: .utf8),
        let fh = FileHandle(forWritingAtPath: path)
    {
        fh.seekToEndOfFile()
        fh.write(data)
        fh.closeFile()
    } else if let data = line.data(using: .utf8) {
        try? data.write(to: URL(fileURLWithPath: path))
    }
}
