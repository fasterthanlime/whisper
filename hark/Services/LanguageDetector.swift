import AppKit
import NaturalLanguage
import os

/// Detects the language of text in the focused UI element using AX + NaturalLanguage.
@MainActor
struct LanguageDetector {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "LanguageDetector"
    )

    /// Map from NLLanguage to Qwen3 language names used by the ASR.
    private static let nlToQwen3: [NLLanguage: String] = [
        .english: "english",
        .french: "french",
        .polish: "polish",
    ]

    /// Detect the language from the focused text field's content.
    /// Returns a Qwen3 language name (e.g. "english", "french") or nil
    /// if detection failed or the language isn't supported.
    static func detectFromFocusedElement() -> String? {
        let systemWide = AXUIElementCreateSystemWide()

        // Get focused element
        var focusedRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            systemWide,
            kAXFocusedUIElementAttribute as CFString,
            &focusedRef
        ) == .success,
              let focusedRef,
              CFGetTypeID(focusedRef) == AXUIElementGetTypeID()
        else {
            return nil
        }
        let element = unsafeBitCast(focusedRef, to: AXUIElement.self)

        // Read the text value
        var valueRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element,
            kAXValueAttribute as CFString,
            &valueRef
        ) == .success, let text = valueRef as? String, !text.isEmpty else {
            return nil
        }

        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)

        guard let dominant = recognizer.dominantLanguage else {
            logger.info("No dominant language detected from \(text.count) chars")
            return nil
        }

        let confidence = recognizer.languageHypotheses(withMaximum: 1)[dominant] ?? 0

        guard let qwen3Name = nlToQwen3[dominant] else {
            logger.info("Detected \(dominant.rawValue, privacy: .public) (conf=\(confidence, format: .fixed(precision: 2))) — not supported")
            return nil
        }

        // For non-English, require high confidence to avoid false locks
        let threshold: Double = (dominant == .english) ? 0.5 : 0.8
        guard confidence >= threshold else {
            logger.info("Detected \(qwen3Name, privacy: .public) but confidence \(confidence, format: .fixed(precision: 2)) < \(threshold, format: .fixed(precision: 2))")
            return nil
        }

        logger.info("Detected \(qwen3Name, privacy: .public) (conf=\(confidence, format: .fixed(precision: 2))) from \(text.count) chars")
        return qwen3Name
    }
}
