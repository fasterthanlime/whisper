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

            let activationEvent = bridge.activate(
                self, pid: frontmostPID, clientID: currClientIdentity)

            switch activationEvent {
            case .none, .delayedTerminalFlushed:
                break
            case .stickyRouteRestored:
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

            let deactivationEvent = bridge.deactivate(self, clientID: senderID)
            switch deactivationEvent {
            case .none:
                break
            case .stickyRouteUnavailable(let hadMarkedText):
                AppClientFactory.shared.imeContextLost(hadMarkedText: hadMarkedText)
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
