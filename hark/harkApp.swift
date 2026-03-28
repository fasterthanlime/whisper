import AppKit
import AVFoundation
import Carbon.HIToolbox.Events
import os
import ServiceManagement
import SwiftUI

extension Notification.Name {
    static let cancelRecording = Notification.Name("cancelRecording")
    static let submitRecording = Notification.Name("submitRecording")
}

private enum StreamingSignal {
    case none              // normal exit (phase changed, key released)
    case over              // ". Over." — submit + keep recording
    case overAndOut        // ". Over and out." — submit + stop
}

private struct StreamingResult {
    let text: String
    let signal: StreamingSignal
    let processedSampleCount: Int
    let autoLockedLanguage: String?
}

private struct TailDebugMetadata: Codable, Sendable {
    let id: String
    let timestampISO8601: String
    let appBundle: String?
    let recordingDurationMs: Int
    let skipPaste: Bool
    let forceSubmit: Bool
    let totalSamples: Int
    let processedSamples: Int
    let remainingSamples: Int
    let finalizeSamples: Int
    let padSamples: Int
    let preFinalizeTextChars: Int
    let finalTextChars: Int
    let preFinalizeText: String
    let finalText: String
    let captureStopMs: Int
    let streamJoinMs: Int
    let finalizeFeedMs: Int
    let finishMs: Int
    let fallbackMs: Int?
}

private struct ForensicsSwiftEvent: Codable, Sendable {
    let tsUnixMs: Int64
    let name: String
    let payload: [String: String]
}

private struct ForensicsRustBatch: Codable, Sendable {
    let pulledTsUnixMs: Int64
    let eventsJSON: String
}

private struct ForensicsDump: Codable, Sendable {
    let id: String
    let startedTsUnixMs: Int64
    let finishedTsUnixMs: Int64
    let swiftEvents: [ForensicsSwiftEvent]
    let rustBatches: [ForensicsRustBatch]
    let finalText: String
}

private final class ForensicsSession: @unchecked Sendable {
    private let id: String
    private let startedTsUnixMs: Int64
    private let sessionDir: URL
    private let lock = NSLock()
    private let bufferWriteQueue = DispatchQueue(label: "hark.forensics.buffer-writer", qos: .utility)
    private var swiftEvents: [ForensicsSwiftEvent] = []
    private var rustBatches: [ForensicsRustBatch] = []
    private var allSamples: [Float] = []
    private var remainingSamples: [Float] = []
    private var finalizeSamples: [Float] = []
    private var finalText: String = ""
    private var nextStreamBufferIndex = 0
    private var nextFinalizeBufferIndex = 0

    init(id: String) {
        self.id = id
        self.startedTsUnixMs = Self.nowUnixMs()
        let root = URL(fileURLWithPath: "/tmp", isDirectory: true)
            .appendingPathComponent("hark-forensics", isDirectory: true)
        self.sessionDir = root.appendingPathComponent(id, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: sessionDir, withIntermediateDirectories: true)
            try FileManager.default.createDirectory(at: sessionDir.appendingPathComponent("buffers", isDirectory: true), withIntermediateDirectories: true)
        } catch {
            print("[hark] forensics init failed: \(error.localizedDescription)")
        }
    }

    func event(_ name: String, _ payload: [String: String] = [:]) {
        lock.lock()
        swiftEvents.append(
            ForensicsSwiftEvent(
                tsUnixMs: Self.nowUnixMs(),
                name: name,
                payload: payload
            )
        )
        lock.unlock()
    }

    func addRustBatch(_ json: String) {
        let trimmed = json.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != "[]" else { return }
        lock.lock()
        rustBatches.append(
            ForensicsRustBatch(
                pulledTsUnixMs: Self.nowUnixMs(),
                eventsJSON: trimmed
            )
        )
        lock.unlock()
    }

    func setAudio(all: [Float], remaining: [Float], finalize: [Float]) {
        lock.lock()
        allSamples = all
        remainingSamples = remaining
        finalizeSamples = finalize
        lock.unlock()
    }

    func setFinalText(_ text: String) {
        lock.lock()
        finalText = text
        lock.unlock()
    }

    @discardableResult
    func writeASRBuffer(mode: String, samples: [Float]) -> String {
        let index: Int
        lock.lock()
        if mode == "finalize" {
            nextFinalizeBufferIndex += 1
            index = nextFinalizeBufferIndex
        } else {
            nextStreamBufferIndex += 1
            index = nextStreamBufferIndex
        }
        lock.unlock()

        let relPath = "buffers/\(mode)-\(String(format: "%05d", index)).wav"
        let outputURL = sessionDir.appendingPathComponent(relPath)
        let samplesCopy = samples
        bufferWriteQueue.async {
            do {
                try Self.writePCM16WAV(samples: samplesCopy, sampleRate: 16_000, to: outputURL)
            } catch {
                print("[hark] forensics buffer write failed: \(error.localizedDescription)")
            }
        }
        return relPath
    }

    func writeHTMLDump() {
        let snapshot: (
            swiftEvents: [ForensicsSwiftEvent],
            rustBatches: [ForensicsRustBatch],
            allSamples: [Float],
            remainingSamples: [Float],
            finalizeSamples: [Float],
            finalText: String
        )
        lock.lock()
        snapshot = (swiftEvents, rustBatches, allSamples, remainingSamples, finalizeSamples, finalText)
        lock.unlock()

        do {
            bufferWriteQueue.sync {}
            try Self.writePCM16WAV(
                samples: snapshot.allSamples,
                sampleRate: 16_000,
                to: sessionDir.appendingPathComponent("all.wav")
            )
            if !snapshot.remainingSamples.isEmpty {
                try Self.writePCM16WAV(
                    samples: snapshot.remainingSamples,
                    sampleRate: 16_000,
                    to: sessionDir.appendingPathComponent("remaining.wav")
                )
            }
            if !snapshot.finalizeSamples.isEmpty {
                try Self.writePCM16WAV(
                    samples: snapshot.finalizeSamples,
                    sampleRate: 16_000,
                    to: sessionDir.appendingPathComponent("finalize.wav")
                )
            }

            let dump = ForensicsDump(
                id: id,
                startedTsUnixMs: startedTsUnixMs,
                finishedTsUnixMs: Self.nowUnixMs(),
                swiftEvents: snapshot.swiftEvents,
                rustBatches: snapshot.rustBatches,
                finalText: snapshot.finalText
            )
            let dumpData = try JSONEncoder().encode(dump)
            let dumpJSON = String(data: dumpData, encoding: .utf8) ?? "{}"
            let escapedJSON = dumpJSON
                .replacingOccurrences(of: "</script>", with: "<\\/script>")
            let html = Self.htmlTemplate(json: escapedJSON, finalText: snapshot.finalText)
            let htmlURL = sessionDir.appendingPathComponent("index.html")
            try html.write(to: htmlURL, atomically: true, encoding: .utf8)
            print("[hark] forensics dump: \(htmlURL.path)")
        } catch {
            print("[hark] forensics write failed: \(error.localizedDescription)")
        }
    }

    private static func nowUnixMs() -> Int64 {
        Int64((Date().timeIntervalSince1970 * 1000).rounded())
    }

    private static func htmlTemplate(json: String, finalText: String) -> String {
        let escapedFinal = finalText
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Hark Forensics</title>
  <style>
    :root { --bg: #0b0d12; --card: #121722; --card2: #171d2b; --text: #eaf0ff; --muted: #93a2c5; --line: #2a3550; --accent: #6ecbff; }
    body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background: radial-gradient(1200px 500px at 10% -10%, #1b2f56 0%, transparent 50%), var(--bg); color: var(--text); margin: 0; padding: 20px; }
    h1, h2 { margin: 0 0 12px 0; letter-spacing: 0.2px; }
    .row { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px; }
    .card { background: linear-gradient(180deg, var(--card2), var(--card)); border: 1px solid var(--line); border-radius: 12px; padding: 12px; min-width: 280px; box-shadow: 0 8px 20px rgba(0,0,0,0.35); }
    .small { font-size: 11px; color: var(--muted); }
    audio { width: 340px; max-width: 100%; height: 32px; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { border-bottom: 1px solid var(--line); padding: 7px 6px; text-align: left; vertical-align: top; }
    th { color: var(--muted); position: sticky; top: 0; background: #0f1420; z-index: 2; }
    pre { white-space: pre-wrap; background: #0f1420; border: 1px solid var(--line); padding: 10px; border-radius: 8px; max-height: 260px; overflow: auto; }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .payload-k { color: #8fb4ff; }
    .payload-v { color: #e5eeff; }
    .payload-item { margin-bottom: 6px; }
    .payload-audio { margin-top: 4px; }
  </style>
</head>
<body>
  <h1>Hark Forensics Timeline</h1>
  <div class="row">
    <div class="card"><strong>all.wav</strong><br/><audio controls src="./all.wav"></audio></div>
    <div class="card"><strong>remaining.wav</strong><br/><audio controls src="./remaining.wav"></audio></div>
    <div class="card"><strong>finalize.wav</strong><br/><audio controls src="./finalize.wav"></audio></div>
  </div>
  <div class="card">
    <h2>Final Text</h2>
    <pre>\(escapedFinal)</pre>
  </div>
  <div class="card" style="margin-top: 16px;">
    <h2>Timeline</h2>
    <table id="timeline"><thead><tr><th>t+ms</th><th>source</th><th>name</th><th>payload</th></tr></thead><tbody></tbody></table>
  </div>

  <script id="forensics-data" type="application/json">\(json)</script>
  <script>
    const raw = document.getElementById('forensics-data').textContent;
    const dump = JSON.parse(raw);
    const rows = [];
    for (const e of dump.swiftEvents || []) {
      rows.push({ ts: e.tsUnixMs, source: "swift", name: e.name, payload: e.payload || {} });
    }
    for (const b of dump.rustBatches || []) {
      try {
        const events = JSON.parse(b.eventsJSON || "[]");
        for (const e of events) {
          rows.push({ ts: Number(e.ts_unix_ms || b.pulledTsUnixMs), source: "rust", name: e.stage || "unknown", payload: e.payload || {} });
        }
      } catch (_) {
        rows.push({ ts: b.pulledTsUnixMs, source: "rust", name: "batch_parse_error", payload: { eventsJSON: b.eventsJSON }});
      }
    }
    rows.sort((a, b) => a.ts - b.ts);
    const t0 = rows.length ? rows[0].ts : Date.now();
    const tbody = document.querySelector("#timeline tbody");
    for (const r of rows) {
      const tr = document.createElement("tr");
      const rel = document.createElement("td");
      rel.textContent = String(r.ts - t0);
      const src = document.createElement("td");
      src.textContent = r.source;
      const name = document.createElement("td");
      name.textContent = r.name;
      const payload = document.createElement("td");
      payload.innerHTML = renderPayload(r.payload);
      tr.append(rel, src, name, payload);
      tbody.appendChild(tr);
    }

    function escapeHtml(s) {
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function renderPayload(payload) {
      const entries = Object.entries(payload || {});
      if (!entries.length) return "{}";
      return entries.map(([k, v]) => {
        const value = String(v);
        const escaped = escapeHtml(value);
        const key = `<span class="payload-k">${escapeHtml(k)}</span>`;
        if (value.endsWith(".wav")) {
          return `<div class="payload-item">${key}: <a href="./${escaped}" target="_blank" rel="noopener">${escaped}</a><div class="payload-audio"><audio controls preload="none" src="./${escaped}"></audio></div></div>`;
        }
        if (k === "text" || k.endsWith("_text")) {
          return `<div class="payload-item">${key}: <pre>${escaped}</pre></div>`;
        }
        return `<div class="payload-item">${key}: <span class="payload-v">${escaped}</span></div>`;
      }).join("");
    }
  </script>
</body>
</html>
"""
    }

    private static func writePCM16WAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        let channelCount: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample / 8)
        let pcmDataByteCount = samples.count * bytesPerSample
        let byteRate = sampleRate * Int(channelCount) * bytesPerSample
        let blockAlign = Int(channelCount) * bytesPerSample
        let riffChunkSize = 36 + pcmDataByteCount

        var data = Data(capacity: 44 + pcmDataByteCount)
        data.append("RIFF".data(using: .ascii)!)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(riffChunkSize).littleEndian, Array.init))
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: channelCount.littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian, Array.init))
        data.append("data".data(using: .ascii)!)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(pcmDataByteCount).littleEndian, Array.init))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let pcm = Int16((clamped * Float(Int16.max)).rounded())
            data.append(contentsOf: withUnsafeBytes(of: pcm.littleEndian, Array.init))
        }

        try data.write(to: url, options: .atomic)
    }
}

/// Intercepts Esc/Return while recording so those keys do not leak to the app behind Hark.
private final class RecordingControlInterceptor: @unchecked Sendable {
    nonisolated(unsafe) var onIntercept: ((UInt16) -> Void)?
    nonisolated(unsafe) var shouldIntercept: ((UInt16) -> Bool)?

    nonisolated(unsafe) private var eventTap: CFMachPort?
    nonisolated(unsafe) private var runLoopSource: CFRunLoopSource?
    nonisolated(unsafe) private var swallowedKeyUps: Set<UInt16> = []

    nonisolated func start() {
        guard eventTap == nil else { return }

        let mask: CGEventMask =
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.keyUp.rawValue)

        let refcon = Unmanaged.passUnretained(self).toOpaque()
        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: mask,
            callback: recordingControlCallback,
            userInfo: refcon
        ) else {
            return
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        if let source = runLoopSource {
            CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        }
        CGEvent.tapEnable(tap: tap, enable: true)
    }

    nonisolated func stop() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        swallowedKeyUps.removeAll()
    }

    deinit {
        stop()
    }

    fileprivate func handle(type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            swallowedKeyUps.removeAll()
            if let tap = eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
            return Unmanaged.passUnretained(event)
        }

        guard type == .keyDown || type == .keyUp else {
            return Unmanaged.passUnretained(event)
        }

        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let isEscapeOrReturn = keyCode == UInt16(kVK_Escape) || keyCode == UInt16(kVK_Return)
        guard isEscapeOrReturn else {
            return Unmanaged.passUnretained(event)
        }

        switch type {
        case .keyDown:
            let isRepeat = event.getIntegerValueField(.keyboardEventAutorepeat) != 0
            if isRepeat {
                return swallowedKeyUps.contains(keyCode) ? nil : Unmanaged.passUnretained(event)
            }

            guard shouldIntercept?(keyCode) == true else {
                return Unmanaged.passUnretained(event)
            }

            swallowedKeyUps.insert(keyCode)
            onIntercept?(keyCode)
            return nil

        case .keyUp:
            guard swallowedKeyUps.contains(keyCode) else {
                return Unmanaged.passUnretained(event)
            }
            swallowedKeyUps.remove(keyCode)
            return nil

        default:
            return Unmanaged.passUnretained(event)
        }
    }
}

nonisolated(unsafe) private let recordingControlCallback: CGEventTapCallBack = { _, type, event, refcon in
    guard let refcon else {
        return Unmanaged.passUnretained(event)
    }
    let interceptor = Unmanaged<RecordingControlInterceptor>.fromOpaque(refcon).takeUnretainedValue()
    return interceptor.handle(type: type, event: event)
}

@main
struct HarkApp: App {
    private static let sharedTranscriptionService = TranscriptionService()
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "startup"
    )

    @State private var appState = AppState()
    @State private var overlayManager = OverlayManager()
    @State private var audioRecorder = AudioRecorder()
    private let transcriptionService = sharedTranscriptionService
    @State private var hotkeyMonitor = HotkeyMonitor()
    @State private var modelLoadTask: Task<Void, Never>?
    @State private var modelLoadGeneration: UInt64 = 0
    @State private var hasLaunched = false
    @State private var recordingTimeoutTask: Task<Void, Never>?
    @State private var streamingTask: Task<StreamingResult, Never>?
    @State private var streamingSession: StreamingSession?
    @State private var inputDeviceMonitor = InputDeviceMonitor()
    @State private var keyDownTime: Date?
    @State private var recordingStartedAt: Date?
    @State private var ignoreHotkeyUntil: Date?
    @State private var pasteTargetBundleID: String?
    @State private var forensicsSession: ForensicsSession?
    /// Skip the next keyUp after locking (so releasing the hotkey after ⌘-lock doesn't submit).
    @State private var skipNextKeyUp = false
    @State private var recordingControlInterceptor = RecordingControlInterceptor()
    @State private var recordingControlObservers: [NSObjectProtocol] = []

    private static let maxRecordingDurationSeconds = AudioRecorder.defaultMaximumDuration
    private static let toggleModeThresholdSeconds: TimeInterval = 0.3
    private static let minimumSpeechDurationSeconds = 0.2
    private static let accidentalDoublePressRecordingThresholdSeconds: TimeInterval = 0.5
    private static let accidentalDoublePressIgnoreWindowSeconds: TimeInterval = 0.5
    private static let streamingChunkSizeSec: Float = 0.75
    private static let transcriptionSampleRate = 16_000.0
    private static let finalizationSilencePaddingSeconds = 0.20
    private static let finalizationMinimumSilencePaddingSeconds = 0.05
    private static let tailDebugDumpEnabled = true
    private static let forensicsDumpEnabled = true

    var body: some Scene {
        MenuBarExtra {
            MenuBarView(
                appState: appState,
                onModelSelect: { model in
                    selectModel(model)
                },
                onDeleteLocalModel: { model in
                    Task { @MainActor in
                        deleteLocalModel(model)
                    }
                },
                onHotkeyBindingSave: { binding in
                    updateHotkeyBinding(binding)
                },
                onHotkeyEditorPresentedChange: { isPresented in
                    appState.isEditingHotkey = isPresented
                },
                runOnStartupEnabled: appState.runOnStartupEnabled,
                onRunOnStartupToggle: {
                    toggleRunOnStartup()
                },
                onSelectInputDevice: { uid in
                    selectInputDevice(uid: uid)
                },
                onSetActiveInputDeviceKeepWarm: { keepWarm in
                    setActiveInputDeviceKeepWarm(keepWarm)
                },
                onRequestMicrophonePermission: {
                    Task { @MainActor in
                        await requestMicrophonePermissionFromMenu()
                    }
                },
                onRequestAccessibilityPermission: {
                    requestAccessibilityPermissionFromMenu()
                },
                onRecheckPermissions: {
                    refreshPermissionState()
                },
                onQuit: {
                    NSApplication.shared.terminate(nil)
                }
            )
        } label: {
            let icon = menuBarNSImage(symbolName: appState.menuBarIcon, size: 18)
            Image(nsImage: icon)
                .task {
                    guard !hasLaunched else { return }
                    hasLaunched = true
                    await onLaunch()
                }
        }
        .menuBarExtraStyle(.window)
    }

    private func menuBarNSImage(symbolName: String, size: CGFloat) -> NSImage {
        let config = NSImage.SymbolConfiguration(pointSize: size, weight: .regular)
        let image = NSImage(systemSymbolName: symbolName, accessibilityDescription: nil)?
            .withSymbolConfiguration(config) ?? NSImage()
        image.isTemplate = true
        return image
    }

    // MARK: - Launch

    @MainActor
    private func onLaunch() async {
        registerBundledFonts()

        let savedID = UserDefaults.standard.string(forKey: "selectedModelID")
        let validIDs = Set(STTModelDefinition.allModels.map(\.id))
        let defaultID = savedID.flatMap { validIDs.contains($0) ? $0 : nil }
            ?? STTModelDefinition.default.id

        appState.selectedModelID = defaultID
        if let saved = UserDefaults.standard.dictionary(forKey: "appLanguages") as? [String: String] {
            appState.appLanguages = saved
        }
        if let saved = UserDefaults.standard.dictionary(forKey: "appVocabPrompts") as? [String: String] {
            appState.appVocabPrompts = saved
        }
        if let saved = UserDefaults.standard.dictionary(forKey: AppState.appAutoSubmitDefaultsKey) as? [String: Bool] {
            appState.appAutoSubmit = saved
        }
        if let saved = UserDefaults.standard.dictionary(
            forKey: AppState.inputDeviceWarmPreferencesDefaultsKey
        ) as? [String: Bool] {
            appState.inputDeviceKeepWarmByUID = saved
        }
        syncRunOnStartupState()
        configureHotkeyFromDefaults()

        await requestPermissions()

        let model = STTModelDefinition.allModels.first { $0.id == defaultID }
            ?? STTModelDefinition.default
        await loadModel(model)

        setupHotkey()
        startInputDeviceMonitoring()
    }

    @MainActor
    private func startInputDeviceMonitoring() {
        inputDeviceMonitor.start { [weak appState] snapshot in
            Task { @MainActor in
                guard let appState else { return }

                let previousDeviceUID = appState.activeInputDeviceUID
                let wasRecording = appState.phase == .recording
                appState.applyInputDeviceSnapshot(snapshot)

                if let active = snapshot.activeDevice {
                    let keepWarm = appState.keepWarmPreference(for: active.uid)
                    Self.logger.info(
                        "Active input device: \(active.name, privacy: .public) (keepWarm=\(keepWarm, privacy: .public))"
                    )
                }

                if previousDeviceUID != nil, previousDeviceUID != snapshot.activeDevice?.uid {
                    Self.logger.info(
                        "Input device changed (wasRecording=\(wasRecording, privacy: .public)); refreshing audio engine"
                    )
                    await reconfigureAudioForCurrentDevice(wasRecording: wasRecording)
                } else {
                    applyActiveInputWarmPolicy()
                }
            }
        }
    }

    @MainActor
    private func warmUpAudio(force: Bool = false) {
        guard appState.hasMicrophonePermission else { return }
        guard force || appState.activeInputDeviceKeepWarm else { return }
        guard !audioRecorder.isWarmedUp else { return }

        do {
            try audioRecorder.warmUp(
                onLevel: { [appState] level in
                    Task { @MainActor in
                        appState.audioLevel = level
                    }
                },
                onSpectrum: { [appState] bands in
                    Task { @MainActor in
                        appState.spectrumBands = bands
                    }
                }
            )
        } catch {
            Self.logger.error("Failed to warm up audio: \(error.localizedDescription, privacy: .public)")
        }
    }

    @MainActor
    private func selectInputDevice(uid: String) {
        guard !isAudioBusy else { return }
        let selected = inputDeviceMonitor.setDefaultInputDevice(uid: uid)
        if !selected {
            _ = appState.transition(to: .error("Could not switch to the selected input device."))
            resetAfterDelay(seconds: 2)
        }
    }

    @MainActor
    private func setActiveInputDeviceKeepWarm(_ keepWarm: Bool) {
        appState.setKeepWarmForActiveInputDevice(keepWarm)
        Task { @MainActor in
            applyActiveInputWarmPolicy()
        }
    }

    @MainActor
    private func reconfigureAudioForCurrentDevice(wasRecording: Bool) async {
        let shouldKeepWarm = appState.activeInputDeviceKeepWarm
        let shouldRunEngine = shouldKeepWarm || wasRecording

        audioRecorder.coolDown()
        appState.audioLevel = 0
        appState.spectrumBands = Array(repeating: 0, count: AudioRecorder.spectrumBandCount)

        guard shouldRunEngine, appState.hasMicrophonePermission else { return }
        try? await Task.sleep(for: .milliseconds(100))
        warmUpAudio(force: true)
        if wasRecording, audioRecorder.isWarmedUp {
            audioRecorder.startCapture()
            Self.logger.info("Restarted capture on newly selected input device")
        }
    }

    @MainActor
    private func applyActiveInputWarmPolicy() {
        guard appState.hasMicrophonePermission else { return }
        let shouldKeepWarm = appState.activeInputDeviceKeepWarm

        if shouldKeepWarm {
            warmUpAudio()
            return
        }

        if !isAudioBusy, audioRecorder.isWarmedUp {
            audioRecorder.coolDown()
            appState.audioLevel = 0
            appState.spectrumBands = Array(repeating: 0, count: AudioRecorder.spectrumBandCount)
        }
    }

    @MainActor
    private var isAudioBusy: Bool {
        switch appState.phase {
        case .recording, .transcribing, .pasting:
            return true
        default:
            return false
        }
    }

    // MARK: - Startup Login Item

    @MainActor
    private func toggleRunOnStartup() {
        let service = SMAppService.mainApp
        let statusBefore = service.status
        let shouldDisable = isRunOnStartupEnabled(statusBefore)
        let action = shouldDisable ? "disable" : "enable"

        do {
            if shouldDisable {
                try service.unregister()
            } else {
                try service.register()
            }

            appState.runOnStartupError = nil
        } catch {
            appState.runOnStartupError =
                "Could not \(action) Run on Startup: \(error.localizedDescription)"
            Self.logger.error(
                "Run on startup toggle failed. action=\(action, privacy: .public) statusBefore=\(String(describing: statusBefore), privacy: .public) error=\(error.localizedDescription, privacy: .public)"
            )
        }

        appState.runOnStartupEnabled = isRunOnStartupEnabled(service.status)
    }

    @MainActor
    private func syncRunOnStartupState() {
        appState.runOnStartupEnabled = isRunOnStartupEnabled(SMAppService.mainApp.status)
    }

    private func isRunOnStartupEnabled(_ status: SMAppService.Status) -> Bool {
        status == .enabled || status == .requiresApproval
    }

    // MARK: - Hotkey Handling

    @MainActor
    private func setupHotkey() {
        hotkeyMonitor.binding = appState.hotkeyBinding

        hotkeyMonitor.onKeyDown = {
            Task { @MainActor in
                await handleKeyDown()
            }
        }
        hotkeyMonitor.onKeyUp = {
            Task { @MainActor in
                await handleKeyUp()
            }
        }
        hotkeyMonitor.onModifierWhileHeld = { keyCode in
            Task { @MainActor in
                // Command pressed while hotkey is held → lock into toggle mode
                if keyCode == 55 || keyCode == 54 { // left/right Command
                    guard appState.phase == .recording, !appState.isLockedMode else { return }
                    appState.isLockedMode = true
                    skipNextKeyUp = true
                }
            }
        }
        hotkeyMonitor.start()
    }

    @MainActor
    private func configureHotkeyFromDefaults() {
        let (binding, fallbackMessage) = HotkeyBinding.load()
        appState.hotkeyBinding = binding
        appState.hotkeySettingsMessage = fallbackMessage
    }

    @MainActor
    private func updateHotkeyBinding(_ binding: HotkeyBinding) {
        appState.hotkeyBinding = binding
        appState.hotkeySettingsMessage = nil
        hotkeyMonitor.binding = binding
        binding.save()
    }

    private func traceEvent(_ name: String, _ payload: [String: String] = [:]) {
        forensicsSession?.event(name, payload)
    }

    private func drainAsrDebugEvents(into trace: ForensicsSession?, session: StreamingSession) {
        guard let trace else { return }
        let json = transcriptionService.takeDebugEventsJSON(session: session)
        trace.addRustBatch(json)
    }

    @MainActor
    private func handleKeyDown() async {
        refreshPermissionState()

        guard !appState.isEditingHotkey else { return }

        // In toggle mode, stop on keyUp (not keyDown) so the user can
        // hold hotkey + press ESC to cancel before releasing.
        if appState.phase == .recording && appState.isLockedMode {
            return
        }

        // Allow starting a new recording even if the previous one is still wrapping up
        // (e.g. pasting or showing the success animation).
        if appState.phase == .transcribing || appState.phase == .pasting {
            overlayManager.hide()
            _ = appState.transition(to: .idle)
        }

        if let ignoreUntil = ignoreHotkeyUntil, Date() < ignoreUntil {
            Self.logger.warning("[hark] hotkey: ignored accidental double-press until=\(ignoreUntil.timeIntervalSince1970)")
            return
        }

        guard appState.phase == .idle else { return }
        guard appState.modelStatus == .loaded else { return }
        traceEvent("hotkey_down")

        guard appState.hasRequiredPermissions else {
            _ = appState.transition(
                to: .error("Missing \(appState.missingPermissionSummary). Open the menu to grant access.")
            )
            resetAfterDelay(seconds: 4)
            return
        }

        _ = appState.transition(to: .recording)
        if Self.forensicsDumpEnabled {
            let trace = ForensicsSession(id: Self.newTailDebugSessionID())
            forensicsSession = trace
            trace.event("recording_start", [
                "front_app_bundle": NSWorkspace.shared.frontmostApplication?.bundleIdentifier ?? "none",
                "front_app_name": NSWorkspace.shared.frontmostApplication?.localizedName ?? "none",
                "language_pref": appState.currentLanguage ?? "auto",
            ])
        } else {
            forensicsSession = nil
        }
        appState.partialTranscript = ""
        appState.partialTranscriptCommittedUTF16 = 0
        keyDownTime = Date()
        recordingStartedAt = Date()
        let frontApp = NSWorkspace.shared.frontmostApplication
        pasteTargetBundleID = frontApp?.bundleIdentifier
        appState.overlayLockedBundleID = pasteTargetBundleID
        appState.overlayLockedAppName = frontApp?.localizedName
        appState.overlayTetherOutOfApp = false
        appState.isLockedMode = false
        hotkeyMonitor.allowExtraModifiers = true

        // Pause media if setting is enabled.
        if MediaController.isEnabled {
            MediaController.pauseIfPlaying()
        }
        overlayManager.show(appState: appState)
        startRecordingTimeout()
        installEscapeMonitor()
        playStartSound()

        // Create streaming session with per-app language preference
        streamingSession = transcriptionService.createSession(
            chunkSizeSec: Self.streamingChunkSizeSec,
            language: appState.currentLanguage,
            prompt: appState.vocabPrompt
        )

        // If audio is warm, just start capturing (instant).
        if audioRecorder.isWarmedUp {
            audioRecorder.startCapture()
        } else {
            do {
                try audioRecorder.start(
                    onLevel: { [appState] (level: Float) in
                        Task { @MainActor in
                            appState.audioLevel = level
                        }
                    },
                    onSpectrum: { [appState] bands in
                        Task { @MainActor in
                            appState.spectrumBands = bands
                        }
                    }
                )
            } catch {
                cancelRecordingTimeout()
                _ = appState.transition(to: .error(error.localizedDescription))
                overlayManager.hide()
                resetAfterDelay()
                return
            }
        }

        // Start streaming transcription
        traceEvent("streaming_session_create", [
            "session_exists": (streamingSession != nil).description,
            "chunk_size_sec": String(format: "%.2f", Self.streamingChunkSizeSec),
        ])
        startStreamingTranscription()
    }

    @MainActor
    private func startStreamingTranscription() {
        startStreamingLoop()
    }

    /// Start (or restart) the streaming loop. On "over", this pastes + submits
    /// then calls itself to keep recording. On "over and out", it stops entirely.
    @MainActor
    private func startStreamingLoop() {
        let transcriptionService = self.transcriptionService
        let audioRecorder = self.audioRecorder
        let appState = self.appState
        let session = self.streamingSession!
        let trace = self.forensicsSession
        let shouldAutoLockLanguage = appState.currentLanguage == nil

        streamingTask = Task.detached { () -> StreamingResult in
            var processedCount = 0
            var lastText = ""
            var signal: StreamingSignal = .none
            var autoLockedLanguage: String?

            while !Task.isCancelled {
                // Check phase on main actor
                let isRecording = await MainActor.run { appState.phase == .recording }
                guard isRecording else { break }

                let allSamples = await MainActor.run { audioRecorder.peekCapture() }

                guard allSamples.count > processedCount + 800 else {
                    try? await Task.sleep(for: .milliseconds(30))
                    continue
                }

                let processedBefore = processedCount
                let newChunk = Array(allSamples[processedCount...])
                let bufferRelPath = trace?.writeASRBuffer(mode: "stream", samples: newChunk)
                trace?.event("stream_feed_call", [
                    "new_chunk_samples": String(newChunk.count),
                    "all_samples": String(allSamples.count),
                    "processed_before": String(processedBefore),
                    "buffer_audio_relpath": bufferRelPath ?? "",
                ])

                let update: StreamingTranscriptUpdate? = transcriptionService.feed(session: session, samples: newChunk)
                let rustEvents = transcriptionService.takeDebugEventsJSON(session: session)
                trace?.addRustBatch(rustEvents)
                processedCount = allSamples.count

                if let update {
                    let trimmed = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
                    trace?.event("stream_feed_result", [
                        "text_len": String(trimmed.count),
                        "committed_utf16": String(update.committedUTF16Count),
                        "detected_language": update.detectedLanguage ?? "none",
                    ])
                    if !trimmed.isEmpty {
                        lastText = trimmed
                        let committedUTF16 = min(
                            max(0, update.committedUTF16Count),
                            (trimmed as NSString).length
                        )
                        await MainActor.run {
                            appState.partialTranscript = trimmed
                            appState.partialTranscriptCommittedUTF16 = committedUTF16
                        }

                        if shouldAutoLockLanguage, autoLockedLanguage == nil {
                            if let detectedLanguage = HarkApp.normalizedSupportedLanguage(update.detectedLanguage) {
                                if transcriptionService.setLanguage(session: session, language: detectedLanguage) {
                                    autoLockedLanguage = detectedLanguage
                                    print("[hark] language-lock lang=\(detectedLanguage)")
                                    trace?.event("language_lock", [
                                        "language": detectedLanguage,
                                    ])
                                    let langEvents = transcriptionService.takeDebugEventsJSON(session: session)
                                    trace?.addRustBatch(langEvents)
                                }
                            }
                        }

                        // Check "over and out" first (higher priority).
                        if trimmed.range(of: #"(?i)[.!?,]\s+over\s+and\s+out\.?\s*$"#, options: .regularExpression) != nil {
                            signal = .overAndOut
                            break
                        }
                        // Then check "over".
                        if trimmed.range(of: #"(?i)[.!?,]\s+over\.?\s*$"#, options: .regularExpression) != nil {
                            signal = .over
                            break
                        }
                    }
                }
            }

            // Strip the trigger phrase from the text.
            lastText = HarkApp.stripTrigger(lastText, signal: signal)
            trace?.event("streaming_loop_exit", [
                "signal": String(describing: signal),
                "processed_samples": String(processedCount),
                "text_len": String(lastText.count),
            ])

            return StreamingResult(
                text: lastText,
                signal: signal,
                processedSampleCount: processedCount,
                autoLockedLanguage: autoLockedLanguage
            )
        }

        // Watcher: handles "over" and "over and out" when they fire mid-recording.
        Task { @MainActor in
            guard let result = await streamingTask?.value else { return }
            guard result.signal != .none && appState.phase == .recording else { return }
            traceEvent("signal_detected", [
                "signal": String(describing: result.signal),
                "text_len": String(result.text.count),
            ])

            print("[hark] signal: \(result.signal) text='\(result.text)'")
            appState.partialTranscript = ""
            appState.partialTranscriptCommittedUTF16 = 0

            if !result.text.isEmpty {
                // Paste + Enter for the current sentence.
                // Temporarily leave recording to paste, then come back if "over" (not "over and out").
                _ = appState.transition(to: .transcribing)
                if self.canPasteIntoLockedTarget() {
                    _ = appState.transition(to: .pasting)
                    do {
                        traceEvent("over_paste_begin", ["text_len": String(result.text.count)])
                        try await PasteController.paste(result.text, submit: true)
                        traceEvent("over_paste_done", ["submit": "true"])
                    } catch {
                        traceEvent("over_paste_error", ["error": error.localizedDescription])
                        print("[hark] paste error: \(error)")
                    }
                    appState.addToHistory(result.text)
                } else {
                    overlayManager.hideWithResult(.cancelled)
                    playCancelSound()
                }
            }

            if result.signal == .over {
                // Keep recording — reset audio buffer and start a fresh streaming session.
                if audioRecorder.isWarmedUp {
                    _ = audioRecorder.stopCapture()
                    audioRecorder.startCapture()
                }
                _ = appState.transition(to: .idle)
                _ = appState.transition(to: .recording)
                appState.partialTranscript = ""
                appState.partialTranscriptCommittedUTF16 = 0
                let languageForRestart = appState.currentLanguage ?? result.autoLockedLanguage
                streamingSession = transcriptionService.createSession(
                    language: languageForRestart,
                    prompt: appState.vocabPrompt
                )
                startStreamingLoop()
            } else {
                // "Over and out" — stop recording entirely.
                traceEvent("over_and_out_stop")
                cancelRecordingTimeout()
                removeEscapeMonitor()
                appState.isLockedMode = false
                keyDownTime = nil
                streamingTask = nil
                streamingSession = nil
                pasteTargetBundleID = nil
                appState.overlayLockedBundleID = nil
                appState.overlayLockedAppName = nil
                appState.overlayTetherOutOfApp = false

                if audioRecorder.isWarmedUp {
                    _ = audioRecorder.stopCapture()
                } else {
                    _ = audioRecorder.stop()
                }
                appState.audioLevel = 0
                if MediaController.isEnabled { MediaController.resumeIfPaused() }
                _ = appState.transition(to: .idle)
                overlayManager.hideWithResult(.success)
                if let trace = forensicsSession {
                    trace.event("session_end")
                    let traceCopy = trace
                    Task.detached {
                        traceCopy.writeHTMLDump()
                    }
                    forensicsSession = nil
                }
            }
        }
    }

    /// Strip "over" or "over and out" from the end of the text.
    private nonisolated static func stripTrigger(_ text: String, signal: StreamingSignal) -> String {
        guard signal != .none else { return text }
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)

        let suffix: String
        switch signal {
        case .overAndOut:
            suffix = "over and out"
        case .over:
            suffix = "over"
        case .none:
            return s
        }

        // Strip trailing period/comma, then the trigger word(s).
        if s.hasSuffix(".") || s.hasSuffix(",") { s = String(s.dropLast()) }
        s = s.trimmingCharacters(in: .whitespaces)
        if s.lowercased().hasSuffix(suffix) {
            s = String(s.dropLast(suffix.count))
            s = s.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return s
    }

    private nonisolated static func normalizedSupportedLanguage(_ raw: String?) -> String? {
        guard let raw else { return nil }
        var normalized = raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return nil }

        if normalized.hasPrefix("language ") {
            normalized = String(normalized.dropFirst("language ".count)).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        let aliases: [String: String] = [
            "en": "english",
            "english": "english",
            "fr": "french",
            "french": "french",
            "es": "spanish",
            "spanish": "spanish",
            "de": "german",
            "german": "german",
            "pl": "polish",
            "polish": "polish",
        ]

        let mapped = aliases[normalized] ?? normalized

        // User policy: French stays French, Polish stays Polish, everything else locks to English.
        if mapped == "french" {
            return "french"
        }
        if mapped == "polish" {
            return "polish"
        }
        return "english"
    }

    @MainActor
    private func handleKeyUp() async {
        guard appState.phase == .recording else { return }
        traceEvent("hotkey_up")

        // After ⌘-lock, skip the keyUp from releasing the original hotkey hold.
        if skipNextKeyUp {
            skipNextKeyUp = false
            return
        }

        // In locked mode, this keyUp is the "stop and submit" action.
        if appState.isLockedMode {
            await stopRecordingAndTranscribe()
            return
        }

        // Check if this was a quick press (toggle mode)
        if let downTime = keyDownTime {
            let pressDuration = Date().timeIntervalSince(downTime)
            if pressDuration < Self.toggleModeThresholdSeconds {
                appState.isLockedMode = true
                return
            }
        }

        await stopRecordingAndTranscribe()
    }

    @MainActor
    private func startRecordingTimeout() {
        cancelRecordingTimeout()
        recordingTimeoutTask = Task { @MainActor in
            do {
                try await Task.sleep(for: .seconds(Self.maxRecordingDurationSeconds))
            } catch {
                return
            }

            guard appState.phase == .recording else { return }
            recordingTimeoutTask = nil
            await stopRecordingAndTranscribe(cancelTimeoutTask: false)
        }
    }

    @MainActor
    private func cancelRecordingTimeout() {
        recordingTimeoutTask?.cancel()
        recordingTimeoutTask = nil
    }

    @MainActor
    private func stopRecordingAndTranscribe(cancelTimeoutTask: Bool = true, skipPaste: Bool = false, forceSubmit: Bool = false) async {
        guard appState.phase == .recording else { return }
        let stopStartedAt = ProcessInfo.processInfo.systemUptime
        let recordingDurationMs = Int(
            ((recordingStartedAt.map { Date().timeIntervalSince($0) } ?? 0) * 1000).rounded()
        )
        traceEvent("stop_begin", [
            "recording_duration_ms": String(recordingDurationMs),
            "skip_paste": String(skipPaste),
            "force_submit": String(forceSubmit),
        ])

        _ = appState.transition(to: .transcribing)

        if cancelTimeoutTask {
            cancelRecordingTimeout()
        }
        removeEscapeMonitor()
        appState.isLockedMode = false
        keyDownTime = nil
        recordingStartedAt = nil
        hotkeyMonitor.allowExtraModifiers = false

        // Stop the streaming loop.
        let stask = streamingTask
        streamingTask = nil
        stask?.cancel()

        // Stop capture and use the definitive captured buffer at stop time.
        // Using `peekCapture()` here can miss tail audio arriving between peek and stop.
        let captureStopStartedAt = ProcessInfo.processInfo.systemUptime
        let allSamples: [Float]
        if audioRecorder.isWarmedUp {
            allSamples = audioRecorder.stopCapture()
        } else {
            allSamples = audioRecorder.stop()
        }
        let captureStopMs = Int(((ProcessInfo.processInfo.systemUptime - captureStopStartedAt) * 1000).rounded())
        traceEvent("capture_stopped", [
            "capture_stop_ms": String(captureStopMs),
            "all_samples": String(allSamples.count),
        ])
        appState.audioLevel = 0
        if !appState.activeInputDeviceKeepWarm, audioRecorder.isWarmedUp {
            audioRecorder.coolDown()
            appState.spectrumBands = Array(repeating: 0, count: AudioRecorder.spectrumBandCount)
        }

        // Feed any remaining samples and finalize the session to get the complete transcript.
        var text = appState.partialTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        let preFinalizeText = text
        var debugProcessedCount = 0
        var debugRemainingSamples: [Float] = []
        var debugFinalizeChunk: [Float] = []
        var debugPadSampleCount = 0
        var debugStreamJoinMs = 0
        var debugFinalizeFeedMs = 0
        var debugFinishMs = 0
        var debugFallbackMs: Int?
        Self.logger.warning("[hark] stop: partial='\(text, privacy: .public)' sessionExists=\(streamingSession != nil) samples=\(allSamples.count)")
        if let session = streamingSession {
            appState.isFinishing = true

            // Wait for the streaming loop to exit so we don't race on the session.
            let streamJoinStartedAt = ProcessInfo.processInfo.systemUptime
            let result = await stask?.value
            let streamJoinMs = Int(((ProcessInfo.processInfo.systemUptime - streamJoinStartedAt) * 1000).rounded())
            let processedCount = result?.processedSampleCount ?? 0
            let remainingCount = max(0, allSamples.count - processedCount)
            drainAsrDebugEvents(into: forensicsSession, session: session)
            debugProcessedCount = processedCount
            debugStreamJoinMs = streamJoinMs
            traceEvent("stream_join_done", [
                "stream_join_ms": String(streamJoinMs),
                "processed_samples": String(processedCount),
                "remaining_samples": String(remainingCount),
            ])

            Self.logger.warning("[hark] finalize: processed=\(processedCount) total=\(allSamples.count) remaining=\(remainingCount)")

            // Feed remaining unprocessed audio and finalize — off main thread.
            let transcriptionService = self.transcriptionService
            let remaining = processedCount < allSamples.count ? Array(allSamples[processedCount...]) : nil
            let configuredPadSampleCount = max(
                0,
                Int((Self.finalizationSilencePaddingSeconds * Self.transcriptionSampleRate).rounded())
            )
            let minimumPadSampleCount = max(
                0,
                Int((Self.finalizationMinimumSilencePaddingSeconds * Self.transcriptionSampleRate).rounded())
            )
            let hasTrailingSilence = Self.hasTrailingSilence(allSamples)
            let padSampleCount = hasTrailingSilence
                ? minimumPadSampleCount
                : configuredPadSampleCount
            let finalizeChunk: [Float] = {
                var chunk = remaining ?? []
                if padSampleCount > 0 {
                    chunk.append(contentsOf: repeatElement(Float(0), count: padSampleCount))
                }
                return chunk
            }()
            let finalizeBufferRel = forensicsSession?.writeASRBuffer(mode: "finalize", samples: finalizeChunk)
            debugRemainingSamples = remaining ?? []
            debugFinalizeChunk = finalizeChunk
            debugPadSampleCount = padSampleCount
            let finalizeMetrics: (text: String?, feedMs: Int, finishMs: Int) = await Task.detached {
                [transcriptionService, session, finalizeChunk] in
                var feedMs = 0
                if !finalizeChunk.isEmpty {
                    let feedStartedAt = ProcessInfo.processInfo.systemUptime
                    _ = transcriptionService.feedFinalizing(session: session, samples: finalizeChunk)
                    feedMs = Int(((ProcessInfo.processInfo.systemUptime - feedStartedAt) * 1000).rounded())
                }
                let finishStartedAt = ProcessInfo.processInfo.systemUptime
                let finishText = transcriptionService.finish(session: session)
                let finishMs = Int(((ProcessInfo.processInfo.systemUptime - finishStartedAt) * 1000).rounded())
                return (text: finishText, feedMs: feedMs, finishMs: finishMs)
            }.value
            debugFinalizeFeedMs = finalizeMetrics.feedMs
            debugFinishMs = finalizeMetrics.finishMs
            drainAsrDebugEvents(into: forensicsSession, session: session)
            traceEvent("finalize_done", [
                "finalize_feed_ms": String(finalizeMetrics.feedMs),
                "finish_ms": String(finalizeMetrics.finishMs),
                "finalize_chunk_samples": String(finalizeChunk.count),
                "pad_samples": String(padSampleCount),
                "finalize_buffer_audio_relpath": finalizeBufferRel ?? "",
            ])
            let finalText = finalizeMetrics.text
            if let finalText {
                let trimmed = finalText.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    text = trimmed
                }
            }

            // If finalization produced nothing, or failed to grow despite meaningful
            // unprocessed tail audio, run a one-shot fallback on full captured audio.
            let fallbackTailThresholdSamples = max(1, Int((0.18 * Self.transcriptionSampleRate).rounded()))
            let shouldRunFallback =
                !allSamples.isEmpty
                && !Self.isEffectivelySilent(allSamples)
                && (
                    text.isEmpty
                    || (
                        remainingCount >= fallbackTailThresholdSamples
                        && text.count <= preFinalizeText.count
                    )
                )
            if shouldRunFallback {
                let transcriptionService = self.transcriptionService
                var samples = allSamples
                if padSampleCount > 0 {
                    samples.append(contentsOf: repeatElement(Float(0), count: padSampleCount))
                }
                let fallbackStartedAt = ProcessInfo.processInfo.systemUptime
                let fallbackText: String? = await Task.detached {
                    transcriptionService.transcribeSamples(samples)
                }.value
                let fallbackMs = Int(((ProcessInfo.processInfo.systemUptime - fallbackStartedAt) * 1000).rounded())
                debugFallbackMs = fallbackMs
                traceEvent("fallback_decode_done", [
                    "fallback_ms": String(fallbackMs),
                    "fallback_samples": String(samples.count),
                ])
                if let fallbackText {
                    let trimmed = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty && (text.isEmpty || trimmed.count > text.count) {
                        text = trimmed
                    }
                }
                Self.logger.warning(
                    "[hark] stop-timing fallback_ms=\(fallbackMs) samples=\(samples.count) pre_len=\(preFinalizeText.count) post_len=\(text.count) remaining=\(remainingCount)"
                )
            }

            Self.logger.warning(
                "[hark] stop-timing capture_stop_ms=\(captureStopMs) stream_join_ms=\(streamJoinMs) finalize_feed_ms=\(finalizeMetrics.feedMs) finish_ms=\(finalizeMetrics.finishMs) finalize_chunk=\(finalizeChunk.count) pad_samples=\(padSampleCount) processed=\(processedCount) remaining=\(remainingCount)"
            )

            // Show the final transcript in the overlay (no animation).
            appState.partialTranscript = text
            appState.partialTranscriptCommittedUTF16 = (text as NSString).length
            appState.isFinishing = false
        }
        streamingSession = nil

        let totalStopMs = Int(((ProcessInfo.processInfo.systemUptime - stopStartedAt) * 1000).rounded())
        Self.logger.warning("[hark] stop-timing total_stop_to_finish_ms=\(totalStopMs)")
        traceEvent("stop_ready_to_paste", [
            "total_stop_to_finish_ms": String(totalStopMs),
            "final_text_len": String(text.count),
        ])

        if recordingDurationMs > 0,
           recordingDurationMs < Int((Self.accidentalDoublePressRecordingThresholdSeconds * 1000).rounded()) {
            let ignoreUntil = Date().addingTimeInterval(Self.accidentalDoublePressIgnoreWindowSeconds)
            ignoreHotkeyUntil = ignoreUntil
            Self.logger.warning(
                "[hark] hotkey: short_recording duration_ms=\(recordingDurationMs) ignore_until=\(ignoreUntil.timeIntervalSince1970)"
            )
        }

        if Self.tailDebugDumpEnabled {
            let dumpID = Self.newTailDebugSessionID()
            let finalTextChars = text.count
            let metadata = TailDebugMetadata(
                id: dumpID,
                timestampISO8601: ISO8601DateFormatter().string(from: Date()),
                appBundle: NSWorkspace.shared.frontmostApplication?.bundleIdentifier,
                recordingDurationMs: recordingDurationMs,
                skipPaste: skipPaste,
                forceSubmit: forceSubmit,
                totalSamples: allSamples.count,
                processedSamples: debugProcessedCount,
                remainingSamples: debugRemainingSamples.count,
                finalizeSamples: debugFinalizeChunk.count,
                padSamples: debugPadSampleCount,
                preFinalizeTextChars: preFinalizeText.count,
                finalTextChars: finalTextChars,
                preFinalizeText: preFinalizeText,
                finalText: text,
                captureStopMs: captureStopMs,
                streamJoinMs: debugStreamJoinMs,
                finalizeFeedMs: debugFinalizeFeedMs,
                finishMs: debugFinishMs,
                fallbackMs: debugFallbackMs
            )
            Task.detached {
                Self.dumpTailDebugArtifacts(
                    metadata: metadata,
                    allSamples: allSamples,
                    remainingSamples: debugRemainingSamples,
                    finalizeSamples: debugFinalizeChunk
                )
            }
            Self.logger.warning("[hark] tail-debug dump_id=\(dumpID, privacy: .public)")
        }

        forensicsSession?.setAudio(all: allSamples, remaining: debugRemainingSamples, finalize: debugFinalizeChunk)
        forensicsSession?.setFinalText(text)

        await finishAndPaste(text: text, skipPaste: skipPaste, forceSubmit: forceSubmit)
    }

    private nonisolated static func isEffectivelySilent(_ samples: [Float], rmsThreshold: Float = 0.006) -> Bool {
        guard !samples.isEmpty else { return true }
        let sumSquares = samples.reduce(Float(0)) { $0 + ($1 * $1) }
        let rms = sqrtf(sumSquares / Float(samples.count))
        return rms < rmsThreshold
    }

    private nonisolated static func hasTrailingSilence(
        _ samples: [Float],
        durationSec: Double = 0.12,
        rmsThreshold: Float = 0.008
    ) -> Bool {
        guard !samples.isEmpty else { return true }
        let trailingCount = max(1, Int((durationSec * Self.transcriptionSampleRate).rounded()))
        let start = max(0, samples.count - trailingCount)
        let tail = Array(samples[start...])
        return isEffectivelySilent(tail, rmsThreshold: rmsThreshold)
    }

    private nonisolated static func newTailDebugSessionID() -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyyMMdd-HHmmss-SSS"
        return formatter.string(from: Date())
    }

    private nonisolated static func tailDebugRootDirectory() -> URL? {
        guard let libraryDir = FileManager.default.urls(for: .libraryDirectory, in: .userDomainMask).first else {
            return nil
        }
        return libraryDir
            .appendingPathComponent("Application Support", isDirectory: true)
            .appendingPathComponent("hark", isDirectory: true)
            .appendingPathComponent("tail-debug", isDirectory: true)
    }

    private nonisolated static func dumpTailDebugArtifacts(
        metadata: TailDebugMetadata,
        allSamples: [Float],
        remainingSamples: [Float],
        finalizeSamples: [Float]
    ) {
        guard let root = tailDebugRootDirectory() else { return }
        let fm = FileManager.default

        do {
            try fm.createDirectory(at: root, withIntermediateDirectories: true)
            let sessionDir = root.appendingPathComponent(metadata.id, isDirectory: true)
            try fm.createDirectory(at: sessionDir, withIntermediateDirectories: true)

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let metadataURL = sessionDir.appendingPathComponent("metadata.json")
            let metadataData = try encoder.encode(metadata)
            try metadataData.write(to: metadataURL, options: .atomic)

            try writePCM16WAV(
                samples: allSamples,
                sampleRate: Int(transcriptionSampleRate),
                to: sessionDir.appendingPathComponent("all.wav")
            )
            if !remainingSamples.isEmpty {
                try writePCM16WAV(
                    samples: remainingSamples,
                    sampleRate: Int(transcriptionSampleRate),
                    to: sessionDir.appendingPathComponent("remaining.wav")
                )
            }
            if !finalizeSamples.isEmpty {
                try writePCM16WAV(
                    samples: finalizeSamples,
                    sampleRate: Int(transcriptionSampleRate),
                    to: sessionDir.appendingPathComponent("finalize.wav")
                )
            }

            pruneTailDebugSessions(in: root, keepLatest: 60)
        } catch {
            Self.logger.error("[hark] tail-debug write failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    private nonisolated static func pruneTailDebugSessions(in root: URL, keepLatest: Int) {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey, .creationDateKey],
            options: [.skipsHiddenFiles]
        ) else {
            return
        }

        let directories = entries.filter { url in
            (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
        }
        guard directories.count > keepLatest else { return }

        let sorted = directories.sorted { lhs, rhs in
            let l = (try? lhs.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            let r = (try? rhs.resourceValues(forKeys: [.creationDateKey]).creationDate) ?? .distantPast
            return l < r
        }
        let toDelete = sorted.prefix(max(0, sorted.count - keepLatest))
        for dir in toDelete {
            try? fm.removeItem(at: dir)
        }
    }

    private nonisolated static func writePCM16WAV(samples: [Float], sampleRate: Int, to url: URL) throws {
        let channelCount: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let bytesPerSample = Int(bitsPerSample / 8)
        let pcmDataByteCount = samples.count * bytesPerSample
        let byteRate = sampleRate * Int(channelCount) * bytesPerSample
        let blockAlign = Int(channelCount) * bytesPerSample
        let riffChunkSize = 36 + pcmDataByteCount

        var data = Data(capacity: 44 + pcmDataByteCount)

        data.append("RIFF".data(using: .ascii)!)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(riffChunkSize).littleEndian, Array.init))
        data.append("WAVE".data(using: .ascii)!)
        data.append("fmt ".data(using: .ascii)!)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian, Array.init)) // PCM
        data.append(contentsOf: withUnsafeBytes(of: channelCount.littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian, Array.init))
        data.append(contentsOf: withUnsafeBytes(of: bitsPerSample.littleEndian, Array.init))
        data.append("data".data(using: .ascii)!)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(pcmDataByteCount).littleEndian, Array.init))

        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let pcm = Int16((clamped * Float(Int16.max)).rounded())
            data.append(contentsOf: withUnsafeBytes(of: pcm.littleEndian, Array.init))
        }

        try data.write(to: url, options: .atomic)
    }

    @MainActor
    private func canPasteIntoLockedTarget() -> Bool {
        guard let lockedBundle = pasteTargetBundleID else { return true }
        let currentBundle = NSWorkspace.shared.frontmostApplication?.bundleIdentifier
        let matches = currentBundle == lockedBundle
        if !matches {
            Self.logger.warning(
                "[hark] paste-lock: blocked locked_bundle=\(lockedBundle, privacy: .public) current_bundle=\(currentBundle ?? "none", privacy: .public)"
            )
        }
        return matches
    }

    @MainActor
    private func finishAndPaste(text: String, skipPaste: Bool = false, forceSubmit: Bool = false) async {
        // Don't clear partialTranscript here — let the dismiss animation show the final text.
        if MediaController.isEnabled { MediaController.resumeIfPaused() }
        traceEvent("finish_and_paste_enter", [
            "text_len": String(text.count),
            "skip_paste": String(skipPaste),
            "force_submit": String(forceSubmit),
            "text": text,
        ])
        forensicsSession?.setFinalText(text)
        defer {
            if let trace = forensicsSession {
                trace.event("session_end")
                let traceCopy = trace
                Task.detached {
                    traceCopy.writeHTMLDump()
                }
            }
            forensicsSession = nil
            pasteTargetBundleID = nil
            appState.overlayLockedBundleID = nil
            appState.overlayLockedAppName = nil
            appState.overlayTetherOutOfApp = false
        }

        if text.isEmpty || skipPaste {
            traceEvent("finish_no_paste", [
                "reason": text.isEmpty ? "empty_text" : "skip_paste",
            ])
            _ = appState.transition(to: .idle)
            overlayManager.hideWithResult(.cancelled)
            playCancelSound()
            if !text.isEmpty { appState.addToHistory(text) }
            return
        }

        guard canPasteIntoLockedTarget() else {
            traceEvent("finish_paste_blocked", [
                "reason": "locked_target_mismatch",
            ])
            _ = appState.transition(to: .idle)
            overlayManager.hideWithResult(.cancelled)
            playCancelSound()
            appState.addToHistory(text)
            return
        }

        appState.addToHistory(text)

        // Log for training data collection
        Task {
            await TranscriptionLogger.shared.log(
                text: text,
                app: NSWorkspace.shared.frontmostApplication?.bundleIdentifier
            )
        }

        let shouldSubmit = forceSubmit || PasteController.isReturnKeyPressed() || appState.currentAutoSubmit
        traceEvent("paste_begin", [
            "submit": String(shouldSubmit),
            "text_len": String(text.count),
            "text": text,
        ])

        // Paste immediately (don't wait for overlay dismiss).
        _ = appState.transition(to: .pasting)
        let pasteTask = Task {
            try await PasteController.paste(text, submit: shouldSubmit)
        }

        // Keep overlay showing final text briefly, then dismiss.
        try? await Task.sleep(for: .milliseconds(350))
        overlayManager.hideWithResult(.success)

        do {
            try await pasteTask.value
            traceEvent("paste_success", [
                "submit": String(shouldSubmit),
                "text_len": String(text.count),
                "text": text,
            ])
            playPastedSound()
            _ = appState.transition(to: .idle)
        } catch {
            traceEvent("paste_error", [
                "error": error.localizedDescription,
            ])
            _ = appState.transition(to: .error(error.localizedDescription))
            overlayManager.hide()
            resetAfterDelay(seconds: 1)
        }
    }

    @MainActor
    private func installEscapeMonitor() {
        removeEscapeMonitor()

        let currentAppState = appState
        let currentHotkeyMonitor = hotkeyMonitor

        recordingControlInterceptor.shouldIntercept = { keyCode in
            guard currentAppState.phase == .recording else { return false }
            guard keyCode == UInt16(kVK_Escape) || keyCode == UInt16(kVK_Return) else { return false }

            if currentAppState.isLockedMode {
                // In locked mode, require that the configured hotkey keys are still held.
                guard currentHotkeyMonitor.binding.keyCodeSet.isSubset(of: currentHotkeyMonitor.pressedKeyCodes) else {
                    return false
                }
            }

            return true
        }

        recordingControlInterceptor.onIntercept = { keyCode in
            if keyCode == UInt16(kVK_Escape) {
                NotificationCenter.default.post(name: .cancelRecording, object: nil)
            } else if keyCode == UInt16(kVK_Return) {
                NotificationCenter.default.post(name: .submitRecording, object: nil)
            }
        }
        recordingControlInterceptor.start()

        let cancelObserver = NotificationCenter.default.addObserver(
            forName: .cancelRecording,
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                await self.stopRecordingAndTranscribe(skipPaste: true, forceSubmit: false)
            }
        }
        recordingControlObservers.append(cancelObserver)

        let submitObserver = NotificationCenter.default.addObserver(
            forName: .submitRecording,
            object: nil,
            queue: .main
        ) { [self] _ in
            Task { @MainActor in
                await self.stopRecordingAndTranscribe(skipPaste: false, forceSubmit: true)
            }
        }
        recordingControlObservers.append(submitObserver)
    }

    @MainActor
    private func removeEscapeMonitor() {
        recordingControlInterceptor.stop()
        recordingControlInterceptor.shouldIntercept = nil
        recordingControlInterceptor.onIntercept = nil

        for observer in recordingControlObservers {
            NotificationCenter.default.removeObserver(observer)
        }
        recordingControlObservers.removeAll()
    }

    // MARK: - Model Management

    @MainActor
    private func selectModel(_ model: STTModelDefinition) {
        switch appState.phase {
        case .recording, .transcribing, .pasting:
            return
        default:
            break
        }

        appState.selectedModelID = model.id
        UserDefaults.standard.set(model.id, forKey: "selectedModelID")

        _ = startModelLoad(model)
    }

    @MainActor
    private func loadModel(_ model: STTModelDefinition) async {
        let task = startModelLoad(model)
        await task.value
    }

    @MainActor
    private func deleteLocalModel(_ model: STTModelDefinition) {
        let cacheDir = STTModelDefinition.cacheDirectory
        let modelDir = URL(fileURLWithPath: cacheDir).appendingPathComponent(model.cacheDirName)

        if appState.selectedModelID == model.id {
            modelLoadTask?.cancel()
            transcriptionService.unloadModel()
        }

        do {
            if FileManager.default.fileExists(atPath: modelDir.path) {
                try FileManager.default.removeItem(at: modelDir)
            }
            appState.downloadedModelIDs.remove(model.id)

            if appState.selectedModelID == model.id {
                appState.modelStatus = .notLoaded
                if case .loading = appState.phase {
                    _ = appState.transition(to: .idle)
                }
            }
        } catch {
            _ = appState.transition(to: .error("Failed to delete model: \(error.localizedDescription)"))
            resetAfterDelay()
        }
    }

    @discardableResult
    @MainActor
    private func startModelLoad(_ model: STTModelDefinition) -> Task<Void, Never> {
        // Ensure any existing streaming session is fully released before
        // loading a different model. A live session keeps an Arc to the old
        // engine and can delay VRAM reclamation.
        streamingTask?.cancel()
        streamingTask = nil
        streamingSession = nil

        modelLoadTask?.cancel()
        modelLoadGeneration &+= 1
        let generation = modelLoadGeneration
        let modelID = model.id

        appState.modelStatus = .loading
        _ = appState.transition(to: .loading("Checking model files..."))

        let task = Task(priority: .userInitiated) {
            do {
                try await transcriptionService.loadModel(
                    model: model,
                    cacheDir: STTModelDefinition.cacheDirectory
                ) { update in
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == modelID else { return }

                    switch update {
                    case .downloading(let progress):
                        appState.modelStatus = .downloading(progress: progress)
                        _ = appState.transition(to: .loading("Downloading model..."))
                    case .initializing:
                        appState.modelStatus = .loading
                        _ = appState.transition(to: .loading("Initializing model..."))
                    }
                }

                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == modelID else { return }

                    appState.modelStatus = .loaded
                    appState.downloadedModelIDs.insert(modelID)
                    _ = appState.transition(to: .idle)
                    modelLoadTask = nil
                }
            } catch is CancellationError {
                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    modelLoadTask = nil

                    switch appState.modelStatus {
                    case .loading, .downloading:
                        appState.modelStatus = .notLoaded
                    default:
                        break
                    }
                    if case .loading = appState.phase {
                        _ = appState.transition(to: .idle)
                    }
                }
            } catch {
                await MainActor.run {
                    guard generation == modelLoadGeneration else { return }
                    guard appState.selectedModelID == modelID else { return }

                    appState.modelStatus = .error(error.localizedDescription)
                    _ = appState.transition(to: .error("Model load failed: \(error.localizedDescription)"))
                    modelLoadTask = nil
                    resetAfterDelay()
                }
            }
        }

        modelLoadTask = task
        return task
    }

    // MARK: - Permissions

    @MainActor
    private func requestPermissions() async {
        let microphoneGranted = await AudioRecorder.requestPermission()
        appState.microphonePermission = microphoneGranted ? .granted : .denied

        if !PasteController.hasAccessibilityPermission {
            PasteController.requestAccessibilityPermission()
        }

        appState.accessibilityPermission =
            PasteController.hasAccessibilityPermission ? .granted : .denied
    }

    @MainActor
    private func requestMicrophonePermissionFromMenu() async {
        let granted = await AudioRecorder.requestPermission()
        refreshPermissionState()

        if granted {
            applyActiveInputWarmPolicy()
            return
        }
        openPrivacySettings(anchor: "Privacy_Microphone")
    }

    @MainActor
    private func requestAccessibilityPermissionFromMenu() {
        PasteController.requestAccessibilityPermission()
        refreshPermissionState()

        if !appState.hasAccessibilityPermission {
            openPrivacySettings(anchor: "Privacy_Accessibility")
        }
    }

    @MainActor
    private func refreshPermissionState() {
        appState.microphonePermission = microphonePermissionStatus()
        appState.accessibilityPermission =
            PasteController.hasAccessibilityPermission ? .granted : .denied
    }

    private func microphonePermissionStatus() -> PermissionStatus {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            return .granted
        case .denied, .restricted:
            return .denied
        case .notDetermined:
            return .unknown
        @unknown default:
            return .denied
        }
    }

    private func openPrivacySettings(anchor: String) {
        guard let url = URL(
            string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)"
        ) else {
            return
        }

        NSWorkspace.shared.open(url)
    }

    // MARK: - Helpers

    @MainActor
    private func resetAfterDelay(seconds: Int = 3) {
        Task {
            try? await Task.sleep(for: .seconds(seconds))
            if case .error = appState.phase {
                _ = appState.transition(to: .idle)
            }
        }
    }

    // MARK: - Fonts

    private func registerBundledFonts() {
        guard let resourceURL = Bundle.main.resourceURL else { return }
        let fontExtensions: Set<String> = ["ttf", "otf"]
        guard let enumerator = FileManager.default.enumerator(
            at: resourceURL, includingPropertiesForKeys: nil
        ) else { return }
        for case let url as URL in enumerator where fontExtensions.contains(url.pathExtension.lowercased()) {
            CTFontManagerRegisterFontsForURL(url as CFURL, .process, nil)
        }
    }

    // MARK: - Sound Effects

    private func systemNotificationVolume() -> Float {
        let globalDefaults = UserDefaults(suiteName: UserDefaults.globalDomain)
        if let raw = globalDefaults?.object(forKey: "com.apple.sound.beep.volume") {
            if let number = raw as? NSNumber {
                return min(max(number.floatValue, 0.0), 1.0)
            }
            if let string = raw as? String, let value = Float(string) {
                return min(max(value, 0.0), 1.0)
            }
        }
        return 1.0
    }

    private func playStartSound() {
        guard let sound = NSSound(named: "Tink") else { return }
        sound.volume = systemNotificationVolume()
        sound.play()
    }

    private func playPastedSound() {
        guard let sound = NSSound(named: "Tink") else { return }
        sound.volume = systemNotificationVolume() * 0.6
        sound.play()
    }

    private func playCancelSound() {
        guard let sound = NSSound(named: "Pop") else { return }
        sound.volume = systemNotificationVolume()
        sound.play()
    }
}
