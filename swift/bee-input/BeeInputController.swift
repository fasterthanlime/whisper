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
        let senderId = describeClient(sender)

        MainActor.assumeIsolated {
            activeInputContext = NSTextInputContext.current
            let frontmostPID = NSWorkspace.shared.frontmostApplication?.processIdentifier

            let currClientIdentity = currentClientIdentity()
            let bridge = BeeIMEBridgeState.shared
            beeInputLog(
                "activateServer: entry senderID=\(senderId) frontmostPID=\(frontmostPID.map(String.init) ?? "nil") clientID=\(currClientIdentity) markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) inputContextCaptured=\(activeInputContext != nil) client=\(describeClient())"
            )
            let isNew = bridge.activate(self, pid: frontmostPID, clientID: currClientIdentity)
            if isNew {
                BeeVoxIMEClient.shared.imeAttach()
            }
        }
        super.activateServer(sender)
    }

    nonisolated override func deactivateServer(_ sender: Any!) {
        let senderId = describeClient(sender)

        var senderClearLog: String = ""
        if let senderAsClient = sender as? (any IMKTextInput & NSObjectProtocol) {
            let before = describeClient(sender)
            senderAsClient.setMarkedText(
                "", selectionRange: NSRange(location: 0, length: 0),
                replacementRange: NSRange(location: 0, length: 0))
            let after = describeClient(sender)
            senderClearLog = "senderClear: before=[\(before)] after=[\(after)]"
        }

        MainActor.assumeIsolated {
            if !senderClearLog.isEmpty {
                beeInputLog(senderClearLog)
            }
            let bridge = BeeIMEBridgeState.shared

            // guard bridge.activeController === self else {
            //     beeInputLog("deactivateServer: stale controller, ignoring")
            //     return
            // }

            let session = bridge.currentSession
            let isDictating = bridge.isDictating

            let client = self.client()
            let clientMarkedRange =
                client?.markedRange() ?? NSRange(location: NSNotFound, length: 0)
            let clientHasMarked =
                clientMarkedRange.location != NSNotFound && clientMarkedRange.length > 0

            let hadMarkedText = !(session?.currentMarkedText.isEmpty ?? true)
            beeInputLog(
                "deactivateServer: hadMarkedText=\(hadMarkedText) senderID=\(senderId) self.client()ID=\(currentClientIdentity()) markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription())"
            )

            if hadMarkedText || clientHasMarked {
                bridge.setMarkedText("")
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

    nonisolated override func commitComposition(_ sender: Any!) {
        MainActor.assumeIsolated {
            beeInputLog(
                "commitComposition: entry clientID=\(currentClientIdentity()) markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) -> cancel"
            )
        }
        cancelComposition()
    }

    nonisolated override func cancelComposition() {
        MainActor.assumeIsolated {
            beeInputLog(
                "cancelComposition: entry clientID=\(currentClientIdentity()) markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) reentrant=\(isHandlingCancelComposition)"
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

    nonisolated override func updateComposition() {
        MainActor.assumeIsolated {
            beeInputLog("updateComposition!")
            let composed =
                (self.composedString(nil) as? String)
                ?? String(describing: self.composedString(nil) ?? "nil")
            beeInputLog(
                "updateComposition: entry clientID=\(currentClientIdentity()) markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription()) composed=\(composed.prefix(80).debugDescription)"
            )
        }
        super.updateComposition()
        MainActor.assumeIsolated {
            beeInputLog(
                "updateComposition: exit clientID=\(currentClientIdentity()) markedRange=\(currentMarkedRangeDescription()) selectedRange=\(currentSelectedRangeDescription())"
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
                "composedString: returning text=\(text.prefix(80).debugDescription) clientID=\(self.currentClientIdentity()) markedRange=\(self.currentMarkedRangeDescription()) selectedRange=\(self.currentSelectedRangeDescription())"
            )
        }
        return text as NSString
    }

    nonisolated override func originalString(_ sender: Any!) -> NSAttributedString! {
        MainActor.assumeIsolated { beeInputLog("originalString!") }

        let attr = NSAttributedString(string: "")
        MainActor.assumeIsolated {
            beeInputLog(
                "originalString: returning empty attributed string clientID=\(self.currentClientIdentity()) markedRange=\(self.currentMarkedRangeDescription()) selectedRange=\(self.currentSelectedRangeDescription())"
            )
        }
        return attr
    }

    nonisolated override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        let eventDesc = String(describing: event)
        MainActor.assumeIsolated {
            beeInputLog("handleEvent! event=\(eventDesc)")
        }

        guard let event, event.type == .keyDown else {
            return false
        }
        let keyCode = event.keyCode
        let characters = event.characters
        return MainActor.assumeIsolated {
            guard BeeIMEBridgeState.shared.isDictating else {
                return false
            }

            switch Int(keyCode) {
            case kVK_Return, kVK_ANSI_KeypadEnter:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    eventType: "submit", keyCode: UInt32(keyCode), characters: "")
                beeInputLog("handle(keyDown): submit")
                return true

            case kVK_Escape:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    eventType: "cancel", keyCode: UInt32(keyCode), characters: "")
                beeInputLog("handle(keyDown): cancel")
                return true

            default:
                BeeVoxIMEClient.shared.imeKeyEvent(
                    eventType: "typed", keyCode: UInt32(keyCode), characters: characters ?? "")
                beeInputLog(
                    "handle(keyDown): typed keyCode=\(keyCode) chars=\((characters ?? "").debugDescription)"
                )
                return false
            }
        }
    }

    @MainActor
    private func discardMarkedTextFromCapturedContext(reason: String) {
        guard let inputContext = activeInputContext ?? NSTextInputContext.current else {
            beeInputLog(
                "discardMarkedText[\(reason)]: no captured/current input context clientID=\(currentClientIdentity()) client=\(describeClient())"
            )
            return
        }

        let contextClientDescription = String(describing: type(of: inputContext.client))

        beeInputLog(
            "discardMarkedText[\(reason)]: via NSTextInputContext clientID=\(currentClientIdentity()) markedRange(before)=\(currentMarkedRangeDescription()) selectedRange(before)=\(currentSelectedRangeDescription()) contextClient=\(contextClientDescription)"
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
        let selectorDesc = String(describing: aSelector)
        let dictDesc = String(describing: infoDictionary)
        MainActor.assumeIsolated {
            beeInputLog("doCommand(by: \(selectorDesc), command: \(dictDesc))")
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
        let desc = String(describing: newDelegate)
        MainActor.assumeIsolated {
            beeInputLog("setDelegate: \(desc)")
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
        let annDesc = String(describing: annotationString)
        let candDesc = String(describing: candidateString)
        MainActor.assumeIsolated {
            beeInputLog("annotationSelected: \(annDesc) forCandidate: \(candDesc)")
        }
        super.annotationSelected(annotationString, forCandidate: candidateString)
    }

    nonisolated override func candidateSelectionChanged(_ candidateString: NSAttributedString!) {
        let desc = String(describing: candidateString)
        MainActor.assumeIsolated {
            beeInputLog("candidateSelectionChanged: \(desc)")
        }
        super.candidateSelectionChanged(candidateString)
    }

    nonisolated override func candidateSelected(_ candidateString: NSAttributedString!) {
        let desc = String(describing: candidateString)
        MainActor.assumeIsolated {
            beeInputLog("candidateSelected: \(desc)")
        }
        super.candidateSelected(candidateString)
    }

    // MARK: - Utilities

    private func currentClientIdentity() -> String {
        describeClient(self.client())
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

    nonisolated private func describeClient(_ obj: Any!) -> String {
        guard let client = obj as? (any IMKTextInput & NSObjectProtocol) else {
            return "nil(\(obj.map { String(describing: type(of: $0)) } ?? "nil"))"
        }
        var parts: [String] = []
        parts.append("id=\(client.uniqueClientIdentifierString() ?? "?")")
        parts.append("bundle=\(client.bundleIdentifier() ?? "?")")
        parts.append("type=\(String(describing: type(of: client)))")
        let marked = client.markedRange()
        parts.append("markedRange=\(NSStringFromRange(marked))")
        parts.append("selectedRange=\(NSStringFromRange(client.selectedRange()))")
        if marked.location != NSNotFound && marked.length > 0 {
            let markedContent = client.attributedSubstring(from: marked)?.string ?? "?"
            parts.append("markedText=\(markedContent.prefix(80).debugDescription)")
        }
        parts.append("length=\(client.length())")
        parts.append("windowLevel=\(client.windowLevel())")
        parts.append("supportsUnicode=\(client.supportsUnicode())")
        parts.append("isProxy=\(client.isProxy())")
        if let validAttrs = client.validAttributesForMarkedText() {
            parts.append("validMarkedAttrs=\(validAttrs)")
        }
        return parts.joined(separator: " ")
    }
}
