import Cocoa
import InputMethodKit
import os

/// App delegate for the HarkInput input method.
/// Initializes the IMKServer and starts the XPC listener.
class HarkInputAppDelegate: NSObject, NSApplicationDelegate {
    private static let logger = Logger(
        subsystem: "fasterthanlime.hark.input-method",
        category: "AppDelegate"
    )

    var server: IMKServer?
    var xpcListener: NSXPCListener?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let connectionName = Bundle.main.infoDictionary?["InputMethodConnectionName"] as? String else {
            Self.logger.error("Missing InputMethodConnectionName in Info.plist")
            return
        }
        guard let bundleID = Bundle.main.bundleIdentifier else {
            Self.logger.error("Missing bundle identifier")
            return
        }

        server = IMKServer(name: connectionName, bundleIdentifier: bundleID)
        Self.logger.info("IMKServer started: \(connectionName, privacy: .public)")

        // Start XPC listener for communication with the main Hark app.
        let listener = NSXPCListener(machServiceName: "fasterthanlime.hark.input-method.xpc")
        listener.delegate = HarkXPCDelegate.shared
        listener.resume()
        xpcListener = listener
        Self.logger.info("XPC listener started")
    }
}
