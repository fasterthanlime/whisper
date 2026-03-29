import AppKit
import NaturalLanguage
import os

/// Detects the language of visible text in the frontmost window using AX + NaturalLanguage.
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

    /// Maximum number of AX elements to visit when collecting text.
    private static let maxElements = 200
    /// Maximum total characters to collect before stopping.
    private static let maxChars = 5000

    /// Detect the language from the frontmost window's visible text.
    /// Walks the AX element tree collecting text from all elements,
    /// then runs NLLanguageRecognizer on the result.
    /// Returns a Qwen3 language name or nil.
    static func detectFromFocusedWindow() -> String? {
        guard let app = NSWorkspace.shared.frontmostApplication else { return nil }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)

        // Get the focused window
        var windowRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            axApp,
            kAXFocusedWindowAttribute as CFString,
            &windowRef
        ) == .success,
              let windowRef,
              CFGetTypeID(windowRef) == AXUIElementGetTypeID()
        else {
            logger.info("No focused window for \(app.localizedName ?? "?", privacy: .public)")
            return nil
        }
        let window = unsafeBitCast(windowRef, to: AXUIElement.self)

        // Collect text from the element tree
        var texts: [String] = []
        var visited = 0
        var totalChars = 0
        collectText(from: window, into: &texts, visited: &visited, totalChars: &totalChars)

        let combined = texts.joined(separator: " ")
        guard combined.count >= 20 else {
            logger.info("Not enough text for detection: \(combined.count) chars from \(visited) elements")
            return nil
        }

        let recognizer = NLLanguageRecognizer()
        recognizer.processString(combined)

        guard let dominant = recognizer.dominantLanguage else {
            logger.info("No dominant language from \(combined.count) chars")
            return nil
        }

        let confidence = recognizer.languageHypotheses(withMaximum: 1)[dominant] ?? 0

        guard let qwen3Name = nlToQwen3[dominant] else {
            logger.info("Detected \(dominant.rawValue, privacy: .public) (conf=\(confidence, format: .fixed(precision: 2))) — not supported")
            return nil
        }

        // For non-English, require high confidence
        let threshold: Double = (dominant == .english) ? 0.5 : 0.8
        guard confidence >= threshold else {
            logger.info("\(qwen3Name, privacy: .public) conf=\(confidence, format: .fixed(precision: 2)) below threshold \(threshold, format: .fixed(precision: 2))")
            return nil
        }

        logger.info("Detected \(qwen3Name, privacy: .public) (conf=\(confidence, format: .fixed(precision: 2))) from \(combined.count) chars / \(visited) elements")
        return qwen3Name
    }

    /// Recursively collect text values from an AX element tree.
    private static func collectText(
        from element: AXUIElement,
        into texts: inout [String],
        visited: inout Int,
        totalChars: inout Int
    ) {
        guard visited < maxElements, totalChars < maxChars else { return }
        visited += 1

        // Try to read this element's text value
        var valueRef: AnyObject?
        if AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &valueRef) == .success,
           let text = valueRef as? String, !text.isEmpty {
            // Skip placeholder text
            var placeholderRef: AnyObject?
            let isPlaceholder = AXUIElementCopyAttributeValue(element, kAXPlaceholderValueAttribute as CFString, &placeholderRef) == .success
                && (placeholderRef as? String) == text
            if !isPlaceholder {
                texts.append(text)
                totalChars += text.count
            }
        }

        // Also try title
        var titleRef: AnyObject?
        if AXUIElementCopyAttributeValue(element, kAXTitleAttribute as CFString, &titleRef) == .success,
           let title = titleRef as? String, !title.isEmpty {
            texts.append(title)
            totalChars += title.count
        }

        // Also try description
        var descRef: AnyObject?
        if AXUIElementCopyAttributeValue(element, kAXDescriptionAttribute as CFString, &descRef) == .success,
           let desc = descRef as? String, !desc.isEmpty {
            texts.append(desc)
            totalChars += desc.count
        }

        guard totalChars < maxChars else { return }

        // Recurse into children
        var childrenRef: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXChildrenAttribute as CFString, &childrenRef) == .success,
              let children = childrenRef as? [AXUIElement] else {
            return
        }

        for child in children {
            guard visited < maxElements, totalChars < maxChars else { break }
            collectText(from: child, into: &texts, visited: &visited, totalChars: &totalChars)
        }
    }
}
