import Carbon
import Cocoa
import InputMethodKit

/// Pure pass-through layer. All state lives in BeeIMEBridgeState (on the bridge).
@objc(BeeInputController)
class BeeInputController: IMKInputController {
    @MainActor
    private var isHandlingCancelComposition = false
    @MainActor
    private weak var activeInputContext: NSTextInputContext?

    nonisolated override init!(server: IMKServer!, delegate: Any!, client: Any!) {
        MainActor.assumeIsolated { beeInputLog("init! with a server, delegate, client") }
        super.init(server: server, delegate: delegate, client: client)
    }

    nonisolated override init() {
        MainActor.assumeIsolated { beeInputLog("init! without anything") }
        super.init()
    }

    nonisolated override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        MainActor.assumeIsolated {
            activeInputContext = NSTextInputContext.current
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier
            let clientIdentity = currentClientIdentity()
            let bridge = BeeIMEBridgeState.shared
            beeInputLog(
                "activateServer: entry frontmostPID=\(frontmostPID.map(String.init) ?? "nil") clientID=\(clientIdentity ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) inputContextCaptured=\(activeInputContext != nil) client=\(describeClient())"
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

            let client = self.client()
            let clientMarkedRange =
                client?.markedRange() ?? NSRange(location: NSNotFound, length: 0)
            let clientHasMarked =
                clientMarkedRange.location != NSNotFound && clientMarkedRange.length > 0

            let hadMarkedText = !(session?.currentMarkedText.isEmpty ?? true)
            beeInputLog(
                "deactivateServer: session=\(sessionID?.uuidString.prefix(8) ?? "none") hadMarkedText=\(hadMarkedText) clientID=\(currentClientIdentity() ?? "nil") markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) client=\(describeClient())"
            )

            if hadMarkedText || clientHasMarked {
                bridge.setMarkedText("", sessionID: sessionID!)
                bridge.didCancelComposition(on: self)
            }

            bridge.deactivate(self)
            activeInputContext = nil

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

            discardMarkedTextFromCapturedContext(reason: "cancelComposition")
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
            beeInputLog("updateComposition!")
            let composed =
                (self.composedString(nil) as? String)
                ?? String(describing: self.composedString(nil) ?? "nil")
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
        MainActor.assumeIsolated { beeInputLog("recognizedEvents") }

        return Int(
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
        MainActor.assumeIsolated { beeInputLog("mouseDown") }

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
        let text =
            MainActor.assumeIsolated {
                beeInputLog("composedString!")
                return BeeIMEBridgeState.shared.currentSession?.currentMarkedText
            } ?? ""
        MainActor.assumeIsolated {
            beeInputLog(
                "composedString: returning text=\(text.prefix(80).debugDescription) clientID=\(self.currentClientIdentity() ?? "nil") markedRange=\(self.currentMarkedRangeDescription()) selectedRange=\(self.currentSelectedRangeDescription())"
            )
        }
        return text as NSString
    }

    nonisolated override func originalString(_ sender: Any!) -> NSAttributedString! {
        MainActor.assumeIsolated { beeInputLog("originalString!") }

        let attr = NSAttributedString(string: "")
        MainActor.assumeIsolated {
            beeInputLog(
                "originalString: returning empty attributed string clientID=\(self.currentClientIdentity() ?? "nil") markedRange=\(self.currentMarkedRangeDescription()) selectedRange=\(self.currentSelectedRangeDescription())"
            )
        }
        return attr
    }

    nonisolated override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        MainActor.assumeIsolated {
            beeInputLog("handleEvent! event=\(String(describing: event))")
        }

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

    @MainActor
    private func discardMarkedTextFromCapturedContext(reason: String) {
        guard let inputContext = activeInputContext ?? NSTextInputContext.current else {
            beeInputLog(
                "discardMarkedText[\(reason)]: no captured/current input context clientID=\(currentClientIdentity() ?? "nil") client=\(describeClient())"
            )
            return
        }

        let contextClientDescription = String(describing: type(of: inputContext.client))

        beeInputLog(
            "discardMarkedText[\(reason)]: via NSTextInputContext clientID=\(currentClientIdentity() ?? "nil") markedRange(before)=\(currentMarkedRangeDescription()) selectedRange(before)=\(currentSelectedRangeDescription()) contextClient=\(contextClientDescription)"
        )
        inputContext.discardMarkedText()
        beeInputLog(
            "discardMarkedText[\(reason)]: markedRange(after)=\(currentMarkedRangeDescription()) selectedRange(after)=\(currentSelectedRangeDescription())"
        )
    }

    // MARK: - IMKInputController overrides (logging)

    nonisolated override func compositionAttributes(at range: NSRange) -> NSMutableDictionary! {
        MainActor.assumeIsolated {
            beeInputLog("compositionAttributes(at: \(NSStringFromRange(range)))")
        }
        return super.compositionAttributes(at: range)
    }

    nonisolated override func selectionRange() -> NSRange {
        let range = super.selectionRange()
        MainActor.assumeIsolated {
            beeInputLog("selectionRange: \(NSStringFromRange(range))")
        }
        return range
    }

    nonisolated override func replacementRange() -> NSRange {
        let range = super.replacementRange()
        MainActor.assumeIsolated {
            beeInputLog("replacementRange: \(NSStringFromRange(range))")
        }
        return range
    }

    nonisolated override func mark(forStyle style: Int, at range: NSRange) -> [AnyHashable: Any]! {
        MainActor.assumeIsolated {
            beeInputLog("mark(forStyle: \(style), at: \(NSStringFromRange(range)))")
        }
        return super.mark(forStyle: style, at: range)
    }

    nonisolated override func doCommand(
        by aSelector: Selector!, command infoDictionary: [AnyHashable: Any]!
    ) {
        MainActor.assumeIsolated {
            beeInputLog(
                "doCommand(by: \(String(describing: aSelector)), command: \(String(describing: infoDictionary)))"
            )
        }
        super.doCommand(by: aSelector, command: infoDictionary)
    }

    nonisolated override func hidePalettes() {
        MainActor.assumeIsolated {
            beeInputLog("hidePalettes")
        }
        super.hidePalettes()
    }

    nonisolated override func menu() -> NSMenu! {
        MainActor.assumeIsolated {
            beeInputLog("menu")
        }
        return super.menu()
    }

    nonisolated override func delegate() -> Any! {
        MainActor.assumeIsolated {
            beeInputLog("delegate")
        }
        return super.delegate()
    }

    nonisolated override func setDelegate(_ newDelegate: Any!) {
        MainActor.assumeIsolated {
            beeInputLog("setDelegate: \(String(describing: newDelegate))")
        }
        super.setDelegate(newDelegate)
    }

    nonisolated override func server() -> IMKServer! {
        MainActor.assumeIsolated {
            beeInputLog("server")
        }
        return super.server()
    }

    nonisolated override func client() -> (any IMKTextInput & NSObjectProtocol)! {
        // Not logging to avoid infinite recursion (many other methods call client())
        return super.client()
    }

    @available(macOS 10.7, *)
    nonisolated override func inputControllerWillClose() {
        MainActor.assumeIsolated {
            beeInputLog("inputControllerWillClose")
        }
        super.inputControllerWillClose()
    }

    nonisolated override func annotationSelected(
        _ annotationString: NSAttributedString!, forCandidate candidateString: NSAttributedString!
    ) {
        MainActor.assumeIsolated {
            beeInputLog(
                "annotationSelected: \(String(describing: annotationString)) forCandidate: \(String(describing: candidateString))"
            )
        }
        super.annotationSelected(annotationString, forCandidate: candidateString)
    }

    nonisolated override func candidateSelectionChanged(_ candidateString: NSAttributedString!) {
        MainActor.assumeIsolated {
            beeInputLog("candidateSelectionChanged: \(String(describing: candidateString))")
        }
        super.candidateSelectionChanged(candidateString)
    }

    nonisolated override func candidateSelected(_ candidateString: NSAttributedString!) {
        MainActor.assumeIsolated {
            beeInputLog("candidateSelected: \(String(describing: candidateString))")
        }
        super.candidateSelected(candidateString)
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
