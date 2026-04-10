import Carbon
import Cocoa
import InputMethodKit

/// Pure pass-through layer. All state lives in Bridge (on the bridge).
@objc(BeeInputController)
class BeeInputController: IMKInputController {
    nonisolated override init!(server: IMKServer!, delegate: Any!, client: Any!) {
        MainActor.assumeIsolated { beeInputLog("init! with a server, delegate, client") }
        super.init(server: server, delegate: delegate, client: client)
    }

    nonisolated override init() {
        MainActor.assumeIsolated { beeInputLog("init! without anything") }
        super.init()
    }

    nonisolated override func activateServer(_ sender: Any!) {
        let senderId = describeClient(sender)

        MainActor.assumeIsolated {
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let currClientIdentity = currentClientIdentity()
            let bridge = Bridge.shared
            beeInputLog(
                "activateServer: senderID=\(senderId) frontmostPID=\(frontmostPID.map(String.init) ?? "nil") clientID=\(currClientIdentity)"
            )

            let stickyRouteBecameAvailable = bridge.activate(
                self, pid: frontmostPID, clientID: currClientIdentity)

            if stickyRouteBecameAvailable {
                AppClientFactory.shared.imeAttach()
            }
        }

        super.activateServer(sender)
    }

    nonisolated override func deactivateServer(_ sender: Any!) {
        let senderID = describeClient(sender)

        MainActor.assumeIsolated {
            let bridge = Bridge.shared

            beeInputLog(
                "deactivateServer: senderID=\(senderID) clientID=\(currentClientIdentity())"
            )

            let stickyRouteBecameUnavailable = bridge.deactivate(self, clientID: senderID)
            if stickyRouteBecameUnavailable {
                AppClientFactory.shared.imeContextLost(hadMarkedText: true)
            }
        }

        super.deactivateServer(sender)
    }

    // MARK: - Utilities

    private func currentClientIdentity() -> String {
        self.client()?.bundleIdentifier() ?? "-"
    }
}

nonisolated func describeClient(_ obj: Any!) -> String {
    guard let client = obj as? (any IMKTextInput & NSObjectProtocol) else {
        return "?"
    }
    let bundleID = client.bundleIdentifier() ?? "-"
    return bundleID
}
