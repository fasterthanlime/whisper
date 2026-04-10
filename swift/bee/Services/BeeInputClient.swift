import AppKit
import Carbon
import Foundation

/// Communicates with the beeInput IME via Vox IPC (Unix socket).
final class BeeInputClient: Sendable {
    private static let beeBundleID = "fasterthanlime.inputmethod.bee"

    /// Returns the real ~/Library/Input Methods/beeInput.app URL.
    /// In a sandboxed app, homeDirectoryForCurrentUser points to the container,
    /// so we derive the real Library path from the group container URL instead.
    static var installedIMEURL: URL? {
        FileManager.default
            .containerURL(
                forSecurityApplicationGroupIdentifier: "B2N6FSRTPV.group.fasterthanlime.bee")?
            .deletingLastPathComponent()  // Group Containers/<id> → Group Containers
            .deletingLastPathComponent()  // Group Containers → Library
            .appendingPathComponent("Input Methods/beeInput.app")
    }

    init() {
        Task { await BeeIPCServer.shared.start() }
    }

    // MARK: - Input Source Switching

    @discardableResult
    func activate() async -> Bool {
        let isConnected = await MainActor.run(body: { BeeIPCServer.shared.isIMEConnected })
        beeLog("IME ACTIVATE: start connected=\(isConnected)")

        if !isConnected {
            guard let installedIME = Self.installedIMEURL else { return false }
            await Self.launchBeeInputIfNeeded(at: installedIME)
        }

        beeLog("IME ACTIVATE: TIS SELECT start")
        let selected = await Self.selectBeeInputSource()
        guard selected else {
            beeLog("IME ACTIVATE: TIS SELECT failed")
            return false
        }

        beeLog("IME ACTIVATE: done")
        return true
    }

    func deactivate(caller: String = #function, file: String = #fileID, line: Int = #line) {
        // beeLog("IME DEACTIVATE called from \(file):\(line) \(caller)")
        // Historical: used to call TISDeselectInputSource here when bee was
        // a keyboard IME. Now bee is palette-type — no need to deselect,
        // and doing so kills the proxy (can't clear marked text after).
    }

    static func waitForIMEReady() async -> Bool {
        await BeeIPCServer.shared.waitForIMEReady()
    }

    // MARK: - IME Commands

    func setMarkedText(_ text: String) {
        Task { await BeeIPCServer.shared.setMarkedText(text: text) }
    }

    func commitText(_ text: String) {
        Task { await BeeIPCServer.shared.commitText(text: text) }
    }

    func clearMarkedText() {
        Task { await BeeIPCServer.shared.stopDictating() }
    }

    func stopDictating() {
        Task { await BeeIPCServer.shared.stopDictating() }
    }

    func replaceText(oldText: String, newText: String) {
        Task { await BeeIPCServer.shared.replaceText(oldText: oldText, newText: newText) }
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

        guard let installedIME = Self.installedIMEURL else {
            beeLog("IME REGISTER: could not resolve group container URL")
            return false
        }
        beeLog("IME REGISTER: installedIME=\(installedIME.path)")

        let exists = FileManager.default.fileExists(atPath: installedIME.path)
        beeLog("IME REGISTER: file exists=\(exists)")
        guard exists else { return false }

        let allSources =
            (TISCreateInputSourceList(allProps as CFDictionary, true)?
                .takeRetainedValue() as? [TISInputSource]) ?? []
        beeLog("IME REGISTER: found \(allSources.count) source(s) (includeAll=true)")

        for (i, source) in allSources.enumerated() {
            let sid = inputSourceID(source)
            let enabled =
                TISGetInputSourceProperty(source, kTISPropertyInputSourceIsEnabled)
                .map { Unmanaged<CFNumber>.fromOpaque($0).takeUnretainedValue() as! Bool } ?? false
            let selected =
                TISGetInputSourceProperty(source, kTISPropertyInputSourceIsSelected)
                .map { Unmanaged<CFNumber>.fromOpaque($0).takeUnretainedValue() as! Bool } ?? false
            beeLog("IME REGISTER:   [\(i)] id=\(sid) enabled=\(enabled) selected=\(selected)")
        }

        if let source = allSources.first {
            let enabled =
                TISGetInputSourceProperty(source, kTISPropertyInputSourceIsEnabled)
                .map { Unmanaged<CFNumber>.fromOpaque($0).takeUnretainedValue() as! Bool } ?? false
            if !enabled {
                beeLog("IME REGISTER: enabling source")
                let r = TISEnableInputSource(source)
                beeLog("IME REGISTER: TISEnableInputSource result=\(r)")
            }
            await launchBeeInputIfNeeded(at: installedIME)
            return true
        }

        beeLog(
            "IME REGISTER: no sources found, calling TISRegisterInputSource url=\(installedIME.path)"
        )
        let status = TISRegisterInputSource(installedIME as CFURL)
        beeLog("IME REGISTER: TISRegisterInputSource result=\(status)")
        guard status == noErr else { return false }

        let newSources =
            (TISCreateInputSourceList(allProps as CFDictionary, true)?
                .takeRetainedValue() as? [TISInputSource]) ?? []
        beeLog("IME REGISTER: after registration, found \(newSources.count) source(s)")
        if let source = newSources.first {
            let r = TISEnableInputSource(source)
            beeLog("IME REGISTER: TISEnableInputSource result=\(r)")
            await launchBeeInputIfNeeded(at: installedIME)
            return true
        }
        beeLog("IME REGISTER: FAILED — no sources even after TISRegisterInputSource")
        return false
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

    private static func launchBeeInputIfNeeded(at url: URL) async {
        if await MainActor.run(body: { BeeIPCServer.shared.isIMEConnected }) {
            beeLog("IME LAUNCH: already connected, skipping")
            return
        }
        beeLog("IME LAUNCH: opening \(url.path)")
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            NSWorkspace.shared.openApplication(
                at: url,
                configuration: NSWorkspace.OpenConfiguration()
            ) { _, error in
                if let error {
                    beeLog("IME LAUNCH: failed: \(error)")
                } else {
                    beeLog("IME LAUNCH: launched ok")
                }
                continuation.resume()
            }
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
