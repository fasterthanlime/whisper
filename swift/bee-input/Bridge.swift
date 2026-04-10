import AppKit
import Carbon
import Foundation
import InputMethodKit

// MARK: - Bridge State

/// Tracks which session is active and routes Vox callbacks to it.
@MainActor
final class Bridge: NSObject {
    static let shared = Bridge()

    /// The `bundleIdentifier` of the app where dictation started.
    private var dictationOriginBundleID: String?

    /// Bundle IDs whose dictation ended while we couldn't reach them directly
    /// (e.g. weak ref released). On next `activate` for a matching bundle,
    /// clear any lingering marked text.
    private var staleOriginBundleIDs: Set<String> = []

    private var controller: BeeInputController?

    // MARK: - State transitions

    /// Returns true if this is a fresh activation (not just a controller update).
    @discardableResult
    func activate(_ controller: BeeInputController, pid: pid_t?, clientID: String?) -> Bool {
        self.controller = controller
        let bundleID = controller.client().bundleIdentifier()
        if dictationOriginBundleID == nil {
            self.dictationOriginBundleID = clientID
            beeInputLog(
                "🟢 ACTIVATE pid=\(pid.map(String.init) ?? "-") client=\(clientID ?? "-") bundle=\(bundleID ?? "-") origin claimed"
            )
            return true
        }

        beeInputLog(
            "⏭️ ACTIVATE ignored pid=\(pid.map(String.init) ?? "-") client=\(clientID ?? "-") bundle=\(bundleID ?? "-") existingOrigin=\(dictationOriginBundleID ?? "-")"
        )
        return false
    }

    func deactivate(_ controller: BeeInputController, clientID: String?) {
        if clientID == dictationOriginBundleID {
            beeInputLog(
                "🔴 DEACTIVATE matched client=\(clientID ?? "-") origin=\(dictationOriginBundleID ?? "-") clearing origin"
            )
            dictationOriginBundleID = nil
            return
        }

        beeInputLog(
            "🚫 DEACTIVATE ignored client=\(clientID ?? "-") origin=\(dictationOriginBundleID ?? "-")"
        )
    }

    // MARK: - Text routing

    func setMarkedText(_ text: String) {
        let client = self.controller?.client()

        // Only deliver to sessions in the original bundle
        if client?.bundleIdentifier() != dictationOriginBundleID {
            beeInputLog(
                "🚫 BLOCKED setMarkedText — client=\(client?.bundleIdentifier() ?? "-") origin=\(self.dictationOriginBundleID ?? "-")"
            )
            return
        }

        beeInputLog("🟡 setMarkedText text=\(text)")
        client?.setMarkedText(
            text, selectionRange: NSRange(location: 0, length: (text as NSString).length),
            replacementRange: NSRange(location: NSNotFound, length: 0))
    }

    func commitText(_ text: String) {
        let client = self.controller?.client()

        // Only deliver to sessions in the original bundle
        if client?.bundleIdentifier() != dictationOriginBundleID {
            beeInputLog(
                "🚫 BLOCKED commitText — client=\(client?.bundleIdentifier() ?? "-") origin=\(self.dictationOriginBundleID ?? "-")"
            )
            return
        }

        beeInputLog("🟢 insertText text=\(text)")
        client?.insertText(
            text, replacementRange: NSRange(location: NSNotFound, length: 0))
    }
}
