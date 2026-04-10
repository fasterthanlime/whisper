import AppKit

class SpyTextView: NSView, NSTextInputClient {
    var markedTextStorage: NSAttributedString = NSAttributedString()
    var markedRangeStorage: NSRange = NSRange(location: NSNotFound, length: 0)
    var selectedRangeStorage: NSRange = NSRange(location: 0, length: 0)
    var textStorage: NSMutableAttributedString = NSMutableAttributedString()

    override var acceptsFirstResponder: Bool { true }

    private lazy var _inputContext: NSTextInputContext = {
        NSTextInputContext(client: self)
    }()

    override var inputContext: NSTextInputContext? {
        _inputContext
    }

    override func becomeFirstResponder() -> Bool {
        spy("becomeFirstResponder")
        return true
    }

    override func resignFirstResponder() -> Bool {
        spy("resignFirstResponder")
        return true
    }

    // MARK: - NSTextInputClient

    func insertText(_ string: Any, replacementRange: NSRange) {
        let text = (string as? String) ?? (string as? NSAttributedString)?.string ?? "?"
        spy("insertText: \(text.debugDescription) replacementRange=\(NSStringFromRange(replacementRange))")

        markedTextStorage = NSAttributedString()
        markedRangeStorage = NSRange(location: NSNotFound, length: 0)

        let insertStr = NSAttributedString(string: text)
        if replacementRange.location != NSNotFound {
            textStorage.replaceCharacters(in: replacementRange, with: insertStr)
            selectedRangeStorage = NSRange(location: replacementRange.location + text.utf16.count, length: 0)
        } else {
            textStorage.insert(insertStr, at: selectedRangeStorage.location)
            selectedRangeStorage = NSRange(location: selectedRangeStorage.location + text.utf16.count, length: 0)
        }

        needsDisplay = true
    }

    func setMarkedText(_ string: Any, selectedRange: NSRange, replacementRange: NSRange) {
        let text: String
        let attr: NSAttributedString
        if let s = string as? String {
            text = s
            attr = NSAttributedString(string: s)
        } else if let a = string as? NSAttributedString {
            text = a.string
            attr = a
        } else {
            text = "?"
            attr = NSAttributedString(string: "?")
        }

        spy("setMarkedText: \(text.debugDescription) selectedRange=\(NSStringFromRange(selectedRange)) replacementRange=\(NSStringFromRange(replacementRange)) markedRange(before)=\(NSStringFromRange(markedRangeStorage))")

        markedTextStorage = attr
        if text.isEmpty {
            markedRangeStorage = NSRange(location: NSNotFound, length: 0)
        } else {
            let loc = markedRangeStorage.location != NSNotFound ? markedRangeStorage.location : selectedRangeStorage.location
            markedRangeStorage = NSRange(location: loc, length: text.utf16.count)
        }

        needsDisplay = true
    }

    func unmarkText() {
        spy("unmarkText: markedRange(before)=\(NSStringFromRange(markedRangeStorage)) markedText=\(markedTextStorage.string.debugDescription)")

        if markedRangeStorage.location != NSNotFound {
            textStorage.insert(markedTextStorage, at: markedRangeStorage.location)
            selectedRangeStorage = NSRange(location: markedRangeStorage.location + markedTextStorage.length, length: 0)
        }
        markedTextStorage = NSAttributedString()
        markedRangeStorage = NSRange(location: NSNotFound, length: 0)

        needsDisplay = true
    }

    func selectedRange() -> NSRange {
        spy("selectedRange -> \(NSStringFromRange(selectedRangeStorage))")
        return selectedRangeStorage
    }

    func markedRange() -> NSRange {
        spy("markedRange -> \(NSStringFromRange(markedRangeStorage))")
        return markedRangeStorage
    }

    func hasMarkedText() -> Bool {
        let has = markedRangeStorage.location != NSNotFound
        spy("hasMarkedText -> \(has)")
        return has
    }

    func attributedSubstring(forProposedRange range: NSRange, actualRange: NSRangePointer?) -> NSAttributedString? {
        spy("attributedSubstring(forProposedRange: \(NSStringFromRange(range)))")
        guard range.location != NSNotFound,
              range.location + range.length <= textStorage.length else {
            return nil
        }
        actualRange?.pointee = range
        return textStorage.attributedSubstring(from: range)
    }

    func validAttributesForMarkedText() -> [NSAttributedString.Key] {
        spy("validAttributesForMarkedText")
        return [.markedClauseSegment, .underlineStyle, .underlineColor]
    }

    func firstRect(forCharacterRange range: NSRange, actualRange: NSRangePointer?) -> NSRect {
        spy("firstRect(forCharacterRange: \(NSStringFromRange(range)))")
        let windowRect = self.convert(NSRect(x: 10, y: 10, width: 100, height: 20), to: nil)
        return self.window?.convertToScreen(windowRect) ?? .zero
    }

    func characterIndex(for point: NSPoint) -> Int {
        spy("characterIndex(for: \(point))")
        return 0
    }

    // MARK: - Commands

    override func doCommand(by selector: Selector) {
        spy("doCommand(by: \(selector))")
        switch selector {
        case #selector(NSResponder.deleteBackward(_:)):
            if markedRangeStorage.location != NSNotFound {
                setMarkedText("", selectedRange: NSRange(location: 0, length: 0), replacementRange: markedRangeStorage)
            } else if selectedRangeStorage.location > 0 {
                let deleteRange = NSRange(location: selectedRangeStorage.location - 1, length: 1)
                textStorage.deleteCharacters(in: deleteRange)
                selectedRangeStorage = NSRange(location: deleteRange.location, length: 0)
                needsDisplay = true
            }
        case #selector(NSResponder.insertNewline(_:)):
            insertText("\n", replacementRange: NSRange(location: NSNotFound, length: 0))
        default:
            break
        }
    }

    // MARK: - Drawing

    override func draw(_ dirtyRect: NSRect) {
        NSColor.white.setFill()
        dirtyRect.fill()

        let committed = textStorage.string
        let marked = markedTextStorage.string

        let display = committed + (marked.isEmpty ? "" : "[\(marked)]")
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedSystemFont(ofSize: 16, weight: .regular),
            .foregroundColor: NSColor.black,
        ]
        (display as NSString).draw(at: NSPoint(x: 10, y: bounds.height - 30), withAttributes: attrs)
    }

    // MARK: - Key handling

    override func keyDown(with event: NSEvent) {
        spy("keyDown: keyCode=\(event.keyCode) chars=\(event.characters?.debugDescription ?? "nil")")
        inputContext?.handleEvent(event)
    }
}
