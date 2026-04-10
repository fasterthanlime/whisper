import Carbon
import Cocoa
import InputMethodKit

/// Pure pass-through layer. All state lives in BeeIMEBridgeState (on the bridge).
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
            let bridge = BeeIMEBridgeState.shared
            beeInputLog(
                "activateServer: senderID=\(senderId) frontmostPID=\(frontmostPID.map(String.init) ?? "nil") clientID=\(currClientIdentity)"
            )

            // On every activateServer: if proxy reports stale marked text
            // and we're not actively dictating, clear it.
            if let client = self.client() as? (any IMKTextInput & NSObjectProtocol) {
                let marked = client.markedRange()
                if marked.location != NSNotFound && marked.length > 0 && !bridge.isDictating {
                    let markedText = client.attributedSubstring(from: marked)?.string
                    beeInputLog(
                        "activateServer: stale marked text markedRange=\(NSStringFromRange(marked)) text=\(markedText?.prefix(80).debugDescription ?? "nil"), clearing"
                    )
                    let empty = NSAttributedString(string: "", attributes: [.markedClauseSegment: 0])
                    client.setMarkedText(
                        empty,
                        selectionRange: NSRange(location: 0, length: 0),
                        replacementRange: NSRange(location: NSNotFound, length: 0)
                    )
                    let markedAfter = client.markedRange()
                    beeInputLog(
                        "activateServer: cleanup result markedRange=\(NSStringFromRange(markedAfter))"
                    )
                }
            }

            let isNew = bridge.activate(self, pid: frontmostPID, clientID: currClientIdentity)
            if isNew {
                BeeVoxIMEClient.shared.imeAttach()
            }
        }
        super.activateServer(sender)
    }

    nonisolated override func deactivateServer(_ sender: Any!) {
        let senderId = describeClient(sender)

        MainActor.assumeIsolated {
            let bridge = BeeIMEBridgeState.shared
            let session = bridge.currentSession
            let isDictating = bridge.isDictating
            let hadMarkedText = !(session?.currentMarkedText.isEmpty ?? true)

            beeInputLog(
                "deactivateServer: hadMarkedText=\(hadMarkedText) senderID=\(senderId) clientID=\(currentClientIdentity())"
            )

            bridge.deactivate(self)

            if isDictating {
                BeeVoxIMEClient.shared.imeContextLost(hadMarkedText: hadMarkedText)
            }
        }
        // cancelComposition triggers IMK's internal flow that sends
        // setMarkedText("") to the actual client app.
        super.cancelComposition()
        super.deactivateServer(sender)
    }

    // MARK: - Utilities

    private func currentClientIdentity() -> String {
        describeClient(self.client())
    }

    nonisolated private func describeClient(_ obj: Any!) -> String {
        guard let client = obj as? (any IMKTextInput & NSObjectProtocol) else {
            return "nil(\(obj.map { String(describing: type(of: $0)) } ?? "nil"))"
        }
        var parts: [String] = []
        parts.append("id=\(client.uniqueClientIdentifierString() ?? "?")")
        parts.append("bundle=\(client.bundleIdentifier() ?? "?")")
        let marked = client.markedRange()
        parts.append("markedRange=\(NSStringFromRange(marked))")
        parts.append("selectedRange=\(NSStringFromRange(client.selectedRange()))")
        if marked.location != NSNotFound && marked.length > 0 {
            let markedContent = client.attributedSubstring(from: marked)?.string ?? "?"
            parts.append("markedText=\(markedContent.prefix(80).debugDescription)")
        }
        return parts.joined(separator: " ")
    }
}
