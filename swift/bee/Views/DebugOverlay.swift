import AVFoundation
import os
import SwiftUI

struct DebugOverlay: View {
    let appState: AppState

    @State private var refreshTick = false
    @State private var refreshTimer: Timer?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            let _ = refreshTick // force redraw on timer
            Text("bee debug")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)

            Divider()

            row("ui", uiStateLabel)
            row("ime", imeStateLabel)
            row("model", modelLabel)

            Divider()

            // Audio engine
            let eng = appState.audioEngine
            row("engine", eng.state == .warm ? "warm" : "cold")
            row("device", appState.activeInputDeviceName ?? "default")
            row("dev uid", eng.selectedDeviceUID ?? "(none)")
            if eng.state == .warm {
                row("rate", "\(Int(eng.nativeSampleRate)) Hz → \(Int(AudioEngine.targetSampleRate)) Hz")
                row("ch", "\(eng.channelCount)")
                row("bufs", "\(eng.totalBuffersReceived)")
                row("samples", "\(eng.totalSamplesReceived)")
                row("rms", String(format: "%.4f", eng.currentRMS))
                row("peak", String(format: "%.4f", eng.peakLevel))
                row("level", String(format: "%.1f%%", eng.currentLevel * 100))
                row("pipes", "\(eng.activePipelineCount)")
            }

            Divider()

            // Devices
            row("devices", "\(appState.availableInputDevices.count) available")
            row("warm", warmDevicesLabel)

            Divider()

            // Stats
            row("sessions", "\(appState.totalSessions)")
            row("words", "\(appState.totalWords)")
            row("chars", "\(appState.totalCharacters)")

            if let session = appState.hotkeyState.session {
                Divider()
                Text("active session")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.orange)
                SessionDebugView(sessionDiag: session.diag)
            }

            if let diag = appState.lastSessionDiag {
                Divider()
                Text("last session")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundStyle(.green)
                DiagnosticsView(diag: diag, transcriptionService: appState.transcriptionService)
            }

            Divider()
            CorrectionTestView(appState: appState)
        }
        .padding(8)
        .frame(width: 300, alignment: .leading)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(.white.opacity(0.1), lineWidth: 0.5)
        }
        .onAppear {
            refreshTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                refreshTick.toggle()
            }
        }
        .onDisappear {
            refreshTimer?.invalidate()
        }
        .onChange(of: refreshTick) { _, _ in }
    }

    private var uiStateLabel: String {
        switch appState.hotkeyState {
        case .idle: "Idle"
        case .held: "Held"
        case .released: "Released"
        case .pushToTalk: "PushToTalk"
        case .locked: "Locked"
        case .lockedOptionHeld: "LockedOptionHeld"
        }
    }

    private var imeStateLabel: String {
        switch appState.imeSessionState {
        case .inactive: "Inactive"
        case .activating: "Activating"
        case .active: "Active"
        case .parked: "Parked"
        }
    }

    private var warmDevicesLabel: String {
        let warm = appState.audioEngine.deviceWarmPolicy.filter { $0.value }.count
        return "\(warm) device(s)"
    }

    private var modelLabel: String {
        switch appState.modelStatus {
        case .notLoaded: "not loaded"
        case .downloading(let p, let model): "downloading \(model) \(Int(p * 100))%"
        case .loading: "loading..."
        case .loaded: "ready"
        case .error(let e): "error: \(e.prefix(30))"
        }
    }

    private func row(_ label: some StringProtocol, _ value: some StringProtocol) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .trailing)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

struct SessionDebugView: View {
    let sessionDiag: SessionDiag

    @State private var snapshot: SessionDiag.Snapshot?

    var body: some View {
        Group {
            if let snapshot {
                DiagnosticsView(diag: snapshot)
            } else {
                Text("loading...")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
        .task(id: UUID()) {
            while !Task.isCancelled {
                // No actor hop — just reads a lock-protected struct
                snapshot = sessionDiag.snapshot
                try? await Task.sleep(for: .milliseconds(100))
            }
        }
    }
}

struct DiagnosticsView: View {
    let diag: SessionDiag.Snapshot
    var transcriptionService: TranscriptionService? = nil

    @State private var batchResult: String?
    @State private var isBatchRunning = false
    @State private var playingSound: NSSound?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            row("ending", diag.ending.isEmpty ? "—" : diag.ending)
            row("rec dur", "\(diag.recordingDurationMs) ms")

            Divider().padding(.vertical, 1)

            row("feeds", "\(diag.feeds) (last \(diag.lastFeedUs / 1000)ms, total \(diag.totalFeedUs / 1000)ms)")
            row("captured", "\(diag.capturedSamples) @16k")
            row("fed", "\(diag.fedSamples) @16k")
            row("total", "\(diag.totalSamples) @16k (\(diag.totalAudioDurationMs) ms)")

            if diag.drainBuffers > 0 {
                row("drain", "\(diag.drainBuffers) bufs (\(diag.drainSamples) @16k)")
            }

            Divider().padding(.vertical, 1)

            row("finalize", "\(diag.finalizeUs / 1000) ms")

            if !diag.finalText.isEmpty {
                Text(diag.finalText.prefix(100))
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.cyan)
                    .lineLimit(3)
                    .padding(.leading, 66)
            }

            if !diag.audioWavPath.isEmpty {
                Divider().padding(.vertical, 1)

                HStack(spacing: 8) {
                    if playingSound?.isPlaying == true {
                        Button("Stop") {
                            playingSound?.stop()
                            playingSound = nil
                        }
                        .font(.system(size: 10, weight: .semibold))
                    } else {
                        Button("Play") {
                            let sound = NSSound(contentsOfFile: diag.audioWavPath, byReference: true)
                            sound?.play()
                            playingSound = sound
                        }
                        .font(.system(size: 10, weight: .semibold))
                    }

                    if let transcriptionService {
                        Button("Re-transcribe") {
                            guard !isBatchRunning else { return }
                            isBatchRunning = true
                            Task {
                                let samples = loadAudioSamples(url: URL(fileURLWithPath: diag.audioWavPath))
                                if !samples.isEmpty {
                                    let result = await transcriptionService.transcribeSamples(samples)
                                    batchResult = result ?? "(no result)"
                                } else {
                                    batchResult = "(failed to load audio)"
                                }
                                isBatchRunning = false
                            }
                        }
                        .font(.system(size: 10, weight: .semibold))
                        .disabled(isBatchRunning)
                    }

                    Button("Reveal") {
                        NSWorkspace.shared.selectFile(diag.audioWavPath, inFileViewerRootedAtPath: "")
                    }
                    .font(.system(size: 10, weight: .semibold))
                }
                .padding(.leading, 66)

                if let batchResult {
                    Text("batch: \(batchResult.prefix(120))")
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundStyle(.yellow)
                        .lineLimit(3)
                        .padding(.leading, 66)
                }
            }
        }
    }

    private func row(_ label: some StringProtocol, _ value: some StringProtocol) -> some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(.secondary)
                .frame(width: 60, alignment: .trailing)
            Text(value)
                .font(.system(size: 10, design: .monospaced))
                .lineLimit(1)
                .truncationMode(.tail)
        }
    }
}

// MARK: - Correction Test

private let corrTestLogger = Logger(subsystem: "fasterthanlime.bee", category: "CorrectionTest")

struct CorrectionTestView: View {
    let appState: AppState

    @State private var corpusDir: URL?
    @State private var isRunning = false
    @State private var statusText: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("correction test")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.purple)

            HStack(spacing: 8) {
                if corpusDir == nil {
                    Button("Pick corpus dir") {
                        let panel = NSOpenPanel()
                        panel.canChooseDirectories = true
                        panel.canChooseFiles = false
                        panel.message = "Select the corpus WAV directory (data/phonetic-seed/audio-wav)"
                        if panel.runModal() == .OK {
                            corpusDir = panel.url
                        }
                    }
                    .font(.system(size: 10, weight: .semibold))
                } else {
                    Button("Test Correction") {
                        guard !isRunning else { return }
                        isRunning = true
                        statusText = "picking file..."
                        Task {
                            await runCorrectionTest()
                            isRunning = false
                        }
                    }
                    .font(.system(size: 10, weight: .semibold))
                    .disabled(isRunning)

                    Button("Change dir") {
                        corpusDir = nil
                    }
                    .font(.system(size: 10, weight: .semibold))
                }
            }

            if let statusText {
                Text(statusText)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.yellow)
                    .lineLimit(5)
                    .padding(.leading, 66)
            }
        }
    }

    private func runCorrectionTest() async {
        guard let dir = corpusDir else { return }

        // List OGG files
        let files: [String]
        do {
            let contents = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            files = contents.filter { $0.hasSuffix(".wav") }
        } catch {
            statusText = "failed to list dir: \(error)"
            return
        }

        guard let randomFile = files.randomElement() else {
            statusText = "no .ogg files found"
            return
        }

        let wavURL = dir.appendingPathComponent(randomFile)
        statusText = "loading \(randomFile)..."
        corrTestLogger.info("Selected: \(randomFile)")

        let samples = loadAudioSamples(url: wavURL)
        guard !samples.isEmpty else {
            statusText = "empty audio: \(randomFile)"
            return
        }

        statusText = "transcribing \(samples.count) samples..."
        corrTestLogger.info("Loaded \(samples.count) samples from \(randomFile)")

        // Create streaming session, feed in 1-second chunks, finish
        let ts = appState.transcriptionService
        guard let session = await ts.createSession() else {
            statusText = "failed to create session"
            return
        }

        // Feed all samples at once — the session handles chunking internally
        corrTestLogger.info("Feeding \(samples.count) samples to session \(session.id)")
        _ = await ts.feed(session: session, samples: samples)

        statusText = "finalizing..."
        let transcript = await ts.finish(session: session) ?? ""
        corrTestLogger.info("Transcript for \(session.id): \"\(transcript)\"")

        if transcript.isEmpty {
            statusText = "empty transcript from \(randomFile)"
            return
        }

        corrTestLogger.info("Transcript: \(transcript)")
        statusText = "correcting: \"\(transcript.prefix(60))\"..."

        // Run correction
        let cs = appState.correctionService
        let corrOutput = await cs.process(text: transcript, appId: "")

        if let corrOutput, !corrOutput.edits.isEmpty {
            statusText = "\(randomFile)\n\"\(transcript.prefix(60))\"\n\(corrOutput.edits.count) correction(s)"
            corrTestLogger.info("Found \(corrOutput.edits.count) correction(s)")

            // Show the correction panel
            CorrectionPanel.shared.show(
                output: corrOutput,
                correctionService: cs,
                inputClient: appState.inputClient
            )
        } else {
            statusText = "\(randomFile)\n\"\(transcript.prefix(80))\"\nno corrections"
        }
    }
}

/// Load audio file as 16kHz mono float32 samples using AVAudioFile.
private func loadAudioSamples(url: URL) -> [Float] {
    guard let file = try? AVAudioFile(forReading: url) else { return [] }
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!
    guard let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else { return [] }
    do {
        try file.read(into: buf)
    } catch {
        return []
    }
    guard let data = buf.floatChannelData?[0] else { return [] }
    return Array(UnsafeBufferPointer(start: data, count: Int(buf.frameLength)))
}
