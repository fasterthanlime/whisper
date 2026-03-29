import Carbon
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

        // Register this app as an input source with macOS.
        let bundleURL = Bundle.main.bundleURL as CFURL
        let registerStatus = TISRegisterInputSource(bundleURL)
        Self.logger.warning("TISRegisterInputSource status=\(registerStatus, privacy: .public)")

        Self.logger.warning("Starting IMKServer name=\(connectionName, privacy: .public) bundle=\(bundleID, privacy: .public)")
        server = IMKServer(name: connectionName, bundleIdentifier: bundleID)
        Self.logger.warning("IMKServer init completed")

        // Listen for commands from the main Hark app via distributed notifications.
        let dnc = DistributedNotificationCenter.default()
        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.hark.setMarkedText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String else { return }
            HarkXPCService.shared.setMarkedText(text)
        }
        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.hark.commitText"),
            object: nil, queue: .main
        ) { notification in
            guard let text = notification.userInfo?["text"] as? String else { return }
            let submit = notification.userInfo?["submit"] as? Bool ?? false
            HarkXPCService.shared.commitText(text, submit: submit)
        }
        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.hark.cancelInput"),
            object: nil, queue: .main
        ) { _ in
            HarkXPCService.shared.cancelInput()
        }
        dnc.addObserver(
            forName: NSNotification.Name("fasterthanlime.hark.stopDictating"),
            object: nil, queue: .main
        ) { _ in
            HarkXPCService.shared.isDictating = false
        }
        Self.logger.warning("Distributed notification listeners registered")
    }
}
