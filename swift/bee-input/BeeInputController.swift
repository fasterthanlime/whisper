import Carbon
import Cocoa
import InputMethodKit

/// Pure pass-through layer. All state lives in BeeIMEBridgeState (on the bridge).
@objc(BeeInputController)
class BeeInputController: IMKInputController {
    @MainActor
    private var isHandlingCancelComposition = false

    nonisolated override init!(server: IMKServer!, delegate: Any!, client: Any!) {
        super.init(server: server, delegate: delegate, client: client)
    }

    nonisolated override init() {
        super.init()
    }

    nonisolated override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        MainActor.assumeIsolated {
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let clientIdentity = currentClientIdentity()
            let bridge = BeeIMEBridgeState.shared
            beeInputLog(
                "activateServer: entry frontmostPID=\(frontmostPID.map(String.init) ?? "nil") clientID=\(clientIdentity ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) client=\(describeClient())"
            )
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

            let clientObj = self.client()
            let clientMarkedRange = clientObj?.markedRange() ?? NSRange(location: NSNotFound, length: 0)
            let clientHasMarked = clientMarkedRange.location != NSNotFound && clientMarkedRange.length > 0

            let hadMarkedText = !(session?.currentMarkedText.isEmpty ?? true)
            beeInputLog(
                "deactivateServer: session=\(sessionID?.uuidString.prefix(8) ?? "none") hadMarkedText=\(hadMarkedText) clientID=\(currentClientIdentity() ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) client=\(describeClient())"
            )

            if hadMarkedText || clientHasMarked {
                beeInputLog("deactivateServer: discarding marked text before detach")
                bridge.discardMarkedText(on: self.client() as AnyObject, reason: "deactivate")
                bridge.didCancelComposition(on: self)
            }

            bridge.deactivate(self)

            if isDictating {
                BeeVoxIMEClient.shared.imeContextLost(hadMarkedText: hadMarkedText)
            }
        }
        super.deactivateServer(sender)
    }

    nonisolated override func cancelComposition() {
        MainActor.assumeIsolated {
            beeInputLog(
                "cancelComposition: entry clientID=\(currentClientIdentity() ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) reentrant=\(isHandlingCancelComposition)"
            )

            guard !isHandlingCancelComposition else {
                BeeIMEBridgeState.shared.didCancelComposition(on: self)
                return
            }

            isHandlingCancelComposition = true
            defer { isHandlingCancelComposition = false }

            BeeIMEBridgeState.shared.discardMarkedText(
                on: self.client() as AnyObject,
                reason: "cancelComposition"
            )
            BeeIMEBridgeState.shared.didCancelComposition(on: self)
        }
    }

    nonisolated override func commitComposition(_ sender: Any!) {
        MainActor.assumeIsolated {
            beeInputLog(
                "commitComposition: entry clientID=\(currentClientIdentity() ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) -> cancel"
            )
        }
        cancelComposition()
    }

    nonisolated override func updateComposition() {
        MainActor.assumeIsolated {
            let composed = (self.composedString(nil) as? String) ?? String(describing: self.composedString(nil) ?? "nil")
            beeInputLog(
                "updateComposition: entry clientID=\(currentClientIdentity() ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) composed=\(composed.prefix(80).debugDescription)"
            )
        }
        super.updateComposition()
        MainActor.assumeIsolated {
            beeInputLog(
                "updateComposition: exit clientID=\(currentClientIdentity() ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription())"
            )
        }
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
        MainActor.assumeIsolated {
            beeInputLog(
                "composedString: returning text=\(text.prefix(80).debugDescription) clientID=\(self.currentClientIdentity() ?? "nil") markedRange=\(self.currentMarkedRangeDescription()) selectedRange=\(self.currentSelectedRangeDescription())"
            )
        }
        return text as NSString
    }

    nonisolated override func originalString(_ sender: Any!) -> NSAttributedString! {
        let attr = NSAttributedString(string: "")
        MainActor.assumeIsolated {
            beeInputLog(
                "originalString: returning empty attributed string clientID=\(self.currentClientIdentity() ?? "nil") markedRange=\(self.currentMarkedRangeDescription()) selectedRange=\(self.currentSelectedRangeDescription())"
            )
        }
        return attr
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
                beeInputLog("handle(keyDown): submit session=\(sessionIdStr.prefix(8))")
                return true

            case kVK_Escape:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    sessionId: sessionIdStr, eventType: "cancel",
                    keyCode: UInt32(keyCode), characters: "")
                beeInputLog("handle(keyDown): cancel session=\(sessionIdStr.prefix(8))")
                return true

            default:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    sessionId: sessionIdStr, eventType: "typed",
                    keyCode: UInt32(keyCode), characters: characters ?? "")
                beeInputLog(
                    "handle(keyDown): typed session=\(sessionIdStr.prefix(8)) keyCode=\(keyCode) chars=\((characters ?? "").debugDescription)"
                )
                return false
            }
        }
    }

    // MARK: - Utilities

    private func currentClientIdentity() -> String? {
        guard let client = self.client() as? NSObject else { return nil }
        let selector = NSSelectorFromString("uniqueClientIdentifierString")
        guard client.responds(to: selector) else { return nil }
        return client.perform(selector)?.takeUnretainedValue() as? String
    }

    private func currentMarkedRangeDescription() -> String {
        guard let client = self.client() else { return "nil" }
        return NSStringFromRange(client.markedRange())
    }

    private func currentSelectedRangeDescription() -> String {
        guard let client = self.client() else { return "nil" }
        return NSStringFromRange(client.selectedRange())
    }

    private func describeClient() -> String {
        guard let client = self.client() else { return "nil" }
        return String(describing: type(of: client))
    }
}
