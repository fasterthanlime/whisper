import Foundation

/// XPC protocol for communication between the main Hark app and the input method.
/// The main app calls these methods to control text insertion.
@objc protocol HarkInputProtocol {
    /// Set provisional (marked) text during streaming transcription.
    /// The text appears underlined and is not yet committed.
    func setMarkedText(_ text: String)

    /// Commit the final text, replacing any marked text.
    func commitText(_ text: String)

    /// Clear any marked text without committing (cancel).
    func cancelInput()
}
