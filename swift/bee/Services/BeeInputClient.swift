import AppKit
import Carbon
import Foundation

/// Communicates with the beeInput IME via Vox IPC (Unix socket).
final class BeeInputClient: Sendable {
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"

    init() {
        Task { await BeeIPCServer.shared.start() }
    }

    // MARK: - Input Source Switching

    @discardableResult
    func activate(sessionID: UUID, targetPID: pid_t?) async -> Bool {
        beeLog("IME ACTIVATE: prepareSession start id=\(sessionID.uuidString.prefix(8))")
        await BeeIPCServer.shared.prepareDictationSession(
            sessionId: sessionID.uuidString,
            targetPid: Int32(targetPID ?? 0)
        )

        beeLog("IME ACTIVATE: TIS SELECT start id=\(sessionID.uuidString.prefix(8))")
        let selected = await Self.selectBeeInputSource()
        guard selected else {
            beeLog("IME ACTIVATE: TIS SELECT failed id=\(sessionID.uuidString.prefix(8))")
            return false
        }

        beeLog("IME ACTIVATE: done id=\(sessionID.uuidString.prefix(8)), waiting for imeAttach")
        return true
    }

    @MainActor
    private static func selectBeeInputSource() async -> Bool {
        guard let beeSource = findBeeInputSource() else {
            beeLog("IME ACTIVATE: bee input source NOT FOUND")
            return false
        }
        let result = TISSelectInputSource(beeSource)
        beeLog("TIS SELECT: \(inputSourceID(beeSource)) result=\(result)")
        return result == noErr
    }

    func deactivate(caller: String = #function, file: String = #fileID, line: Int = #line) {
        beeLog("IME DEACTIVATE called from \(file):\(line) \(caller)")
        if let source = Self.findBeeInputSource() {
            let result = TISDeselectInputSource(source)
            beeLog("TIS DESELECT: \(Self.inputSourceID(source)) result=\(result)")
        }
    }

    static func waitForIMEReady() async -> Bool {
        await BeeIPCServer.shared.waitForIMEReady()
    }

    // MARK: - IME Commands

    func setMarkedText(_ text: String, sessionID: UUID) {
        beeLog("setMarkedText → vox session=\(sessionID.uuidString.prefix(8))")
        Task { await BeeIPCServer.shared.setMarkedText(sessionId: sessionID.uuidString, text: text) }
    }

    func logSetMarkedText(_ text: String, sessionID: UUID) {
        beeLog("IME setMarkedText: \(text.prefix(60).debugDescription)")
        setMarkedText(text, sessionID: sessionID)
    }

    func commitText(_ text: String, sessionID: UUID) {
        Task { await BeeIPCServer.shared.commitText(sessionId: sessionID.uuidString, text: text) }
    }

    func clearMarkedText(sessionID: UUID) {
        Task { await BeeIPCServer.shared.stopDictating(sessionId: sessionID.uuidString) }
    }

    func stopDictating(sessionID: UUID) {
        Task { await BeeIPCServer.shared.stopDictating(sessionId: sessionID.uuidString) }
    }

    func simulateReturn() {
        let src = CGEventSource(stateID: .hidSystemState)
        if let down = CGEvent(keyboardEventSource: src, virtualKey: 0x24, keyDown: true),
            let up = CGEvent(keyboardEventSource: src, virtualKey: 0x24, keyDown: false)
        {
            down.post(tap: .cghidEventTap)
            usleep(10_000)
            up.post(tap: .cghidEventTap)
        }
    }

    // MARK: - IME Registration

    @discardableResult
    static func ensureIMERegistered() -> Bool {
        let allProps: [CFString: Any] = [kTISPropertyBundleID: beeBundleID as CFString]
        let allSources =
            (TISCreateInputSourceList(allProps as CFDictionary, true)?
                .takeRetainedValue() as? [TISInputSource]) ?? []

        beeLog("IME REGISTER: found \(allSources.count) source(s) (includeAll=true)")

        if let source = allSources.first {
            let enabled =
                TISGetInputSourceProperty(source, kTISPropertyInputSourceIsEnabled)
                .map { Unmanaged<CFNumber>.fromOpaque($0).takeUnretainedValue() as! Bool } ?? false
            if !enabled {
                beeLog("IME REGISTER: source disabled, enabling")
                TISEnableInputSource(source)
            }
            return true
        }

        // Not registered — register from bundle (preferred) or fallback to ~/Library/Input Methods/
        let bundledIME = Bundle.main.bundleURL
            .appendingPathComponent("Contents/Library/Input Methods/beeInput.app")
        let fallbackIME = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Input Methods/beeInput.app")

        let inputMethodsDir: URL
        if FileManager.default.fileExists(atPath: bundledIME.path) {
            beeLog("IME REGISTER: using bundled beeInput.app at \(bundledIME.path)")
            inputMethodsDir = bundledIME
        } else if FileManager.default.fileExists(atPath: fallbackIME.path) {
            beeLog("IME REGISTER: using fallback beeInput.app at \(fallbackIME.path)")
            inputMethodsDir = fallbackIME
        } else {
            beeLog("IME REGISTER: beeInput.app not found in bundle or ~/Library/Input Methods/")
            return false
        }

        let status = TISRegisterInputSource(inputMethodsDir as CFURL)
        beeLog("IME REGISTER: TISRegisterInputSource result=\(status)")
        guard status == noErr else { return false }

        let newSources =
            (TISCreateInputSourceList(allProps as CFDictionary, true)?
                .takeRetainedValue() as? [TISInputSource]) ?? []
        beeLog("IME REGISTER: after registration, found \(newSources.count) source(s)")
        if let source = newSources.first {
            TISEnableInputSource(source)
            return true
        }
        return false
    }

    static func restoreInputSourceIfNeeded(
        caller: String = #function, file: String = #fileID, line: Int = #line
    ) {
        // Palette input sources stay selected permanently — nothing to restore.
    }

    private static func inputSourceID(_ source: TISInputSource) -> String {
        guard let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else {
            return "<unknown>"
        }
        return Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String
    }

    private static func findBeeInputSource() -> TISInputSource? {
        let properties: [CFString: Any] = [kTISPropertyBundleID: beeBundleID as CFString]
        guard
            let sources = TISCreateInputSourceList(properties as CFDictionary, false)?
                .takeRetainedValue() as? [TISInputSource],
            let source = sources.first
        else { return nil }
        return source
    }
}
