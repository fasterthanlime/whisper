import Carbon
import Cocoa
import InputMethodKit

/// Pure pass-through layer. All state lives in BeeIMEBridgeState (on the bridge).
@objc(BeeInputController)
class BeeInputController: IMKInputController {

    nonisolated override init!(server: IMKServer!, delegate: Any!, client: Any!) {
        super.init(server: server, delegate: delegate, client: client)
    }

    nonisolated override init() {
        super.init()
    }

    nonisolated override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        MainActor.assumeIsolated {
            beeInputLog("activateServer: entry")
            let bridge = BeeIMEBridgeState.shared
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let clientIdentity = currentClientIdentity()
            bridge.activate(self, pid: frontmostPID, clientID: clientIdentity)
        }
    }

    nonisolated override func deactivateServer(_ sender: Any!) {
        MainActor.assumeIsolated {
            let bridge = BeeIMEBridgeState.shared

            guard bridge.activeController === self else {
                beeInputLog("deactivateServer: stale controller, ignoring")
                return
            }

            let session = bridge.currentSession
            let isDictating = bridge.isDictating
            let sessionID = bridge.activeSessionID

            let hadMarkedText = !(session?.currentMarkedText.isEmpty ?? true)
            beeInputLog(
                "deactivateServer: session=\(sessionID?.uuidString.prefix(8) ?? "none") hadMarkedText=\(hadMarkedText)"
            )

            bridge.deactivate(self)

            if isDictating {
                BeeVoxIMEClient.shared.imeContextLost(hadMarkedText: hadMarkedText)
            }
        }
        super.deactivateServer(sender)
    }

    nonisolated override func cancelComposition() {
        MainActor.assumeIsolated {
            beeInputLog("cancelComposition: entry")
        }
        super.cancelComposition()
        MainActor.assumeIsolated {
            BeeIMEBridgeState.shared.didCancelComposition(on: self)
        }
    }

    nonisolated override func commitComposition(_ sender: Any!) {
        MainActor.assumeIsolated {
            beeInputLog("commitComposition: entry -> cancel")
        }
        cancelComposition()
    }

    nonisolated override func recognizedEvents(_ sender: Any!) -> Int {
        Int(
            NSEvent.EventTypeMask([
                .keyDown,
                .leftMouseDown,
                .rightMouseDown,
                .otherMouseDown,
            ]).rawValue
        )
    }

    nonisolated override func mouseDown(
        onCharacterIndex index: Int,
        coordinate point: NSPoint,
        withModifier flags: Int,
        continueTracking keepTracking: UnsafeMutablePointer<ObjCBool>!,
        client sender: Any!
    ) -> Bool {
        keepTracking?.pointee = false
        return MainActor.assumeIsolated {
            let bridge = BeeIMEBridgeState.shared
            guard bridge.activeController === self else { return false }
            guard let session = bridge.currentSession else { return false }
            guard let client = self.client() else { return false }

            let markedRange = client.markedRange()
            guard markedRange.location != NSNotFound else { return false }

            let clickedInsideComposition = NSLocationInRange(index, markedRange)
            beeInputLog(
                "mouseDown: index=\(index) markedRange=\(markedRange) inside=\(clickedInsideComposition)"
            )

            if !clickedInsideComposition && !session.currentMarkedText.isEmpty {
                beeInputLog("mouseDown: outside composition -> cancel")
                cancelComposition()
            }
            return false
        }
    }

    nonisolated override func composedString(_ sender: Any!) -> Any! {
        let text = MainActor.assumeIsolated {
            BeeIMEBridgeState.shared.currentSession?.currentMarkedText
        } ?? ""
        return text as NSString
    }

    nonisolated override func originalString(_ sender: Any!) -> NSAttributedString! {
        NSAttributedString(string: "")
    }

    nonisolated override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event, event.type == .keyDown else {
            return false
        }
        let keyCode = event.keyCode
        let characters = event.characters
        return MainActor.assumeIsolated {
            let bridge = BeeIMEBridgeState.shared
            guard let sessionID = bridge.activeSessionID else {
                return false
            }
            let sessionIdStr = sessionID.uuidString

            switch Int(keyCode) {
            case kVK_Return, kVK_ANSI_KeypadEnter:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    sessionId: sessionIdStr, eventType: "submit",
                    keyCode: UInt32(keyCode), characters: "")
                return true

            case kVK_Escape:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    sessionId: sessionIdStr, eventType: "cancel",
                    keyCode: UInt32(keyCode), characters: "")
                return true

            default:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    sessionId: sessionIdStr, eventType: "typed",
                    keyCode: UInt32(keyCode), characters: characters ?? "")
                return false
            }
        }
    }

    // MARK: - Utilities

    private func currentClientIdentity() -> String? {
        guard let client = self.client() else { return nil }
        let opaque = Unmanaged.passUnretained(client as AnyObject).toOpaque()
        return String(UInt(bitPattern: opaque), radix: 16, uppercase: true)
    }
}
