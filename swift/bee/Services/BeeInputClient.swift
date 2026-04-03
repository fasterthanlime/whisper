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
        if await !MainActor.run(body: { BeeIPCServer.shared.isIMEConnected }) {
            let installedIME = FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Input Methods/beeInput.app")
            await Self.launchBeeInputIfNeeded(at: installedIME)
        }

        // Select first so the OS sends activateServer to beeInput before prepareSession arrives.
        beeLog("IME ACTIVATE: TIS SELECT start id=\(sessionID.uuidString.prefix(8))")
        let selected = await Self.selectBeeInputSource()
        guard selected else {
            beeLog("IME ACTIVATE: TIS SELECT failed id=\(sessionID.uuidString.prefix(8))")
            return false
        }

        beeLog("IME ACTIVATE: prepareSession start id=\(sessionID.uuidString.prefix(8))")
        await BeeIPCServer.shared.prepareDictationSession(
            sessionId: sessionID.uuidString,
            targetPid: Int32(targetPID ?? 0)
        )

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
        // Don't deselect — keeping beeInput selected preserves its controller state
        // so the next session can start immediately without waiting for re-activation.
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
    static func ensureIMERegistered() async -> Bool {
        let allProps: [CFString: Any] = [kTISPropertyBundleID: beeBundleID as CFString]

        let installedIME = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Input Methods/beeInput.app")

        guard FileManager.default.fileExists(atPath: installedIME.path) else {
            beeLog("IME REGISTER: beeInput.app not found at \(installedIME.path)")
            return false
        }

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
            // Pre-select so activateServer fires on the next text-field focus,
            // before the user ever presses the dictation hotkey.
            let selectResult = TISSelectInputSource(source)
            beeLog("IME REGISTER: pre-selected beeInput result=\(selectResult)")
            await launchBeeInputIfNeeded(at: installedIME)
            return true
        }

        let status = TISRegisterInputSource(installedIME as CFURL)
        beeLog("IME REGISTER: TISRegisterInputSource result=\(status)")
        guard status == noErr else { return false }

        let newSources =
            (TISCreateInputSourceList(allProps as CFDictionary, true)?
                .takeRetainedValue() as? [TISInputSource]) ?? []
        beeLog("IME REGISTER: after registration, found \(newSources.count) source(s)")
        if let source = newSources.first {
            TISEnableInputSource(source)
            await launchBeeInputIfNeeded(at: installedIME)
            return true
        }
        return false
    }

    private static func launchBeeInputIfNeeded(at url: URL) async {
        if await MainActor.run(body: { BeeIPCServer.shared.isIMEConnected }) {
            beeLog("IME LAUNCH: already connected, skipping")
            return
        }
        beeLog("IME LAUNCH: opening \(url.path)")
        await MainActor.run {
            NSWorkspace.shared.open(url)
        }
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
