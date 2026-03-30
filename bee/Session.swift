import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "Session")

/// A Session is a self-contained unit of work for a single dictation attempt.
///
/// It has three internal layers — Capture, ASR, and IME — that run
/// concurrently. The streaming loop feeds audio from Capture to ASR
/// continuously, including during the drain phase after commit/cancel.
///
/// Three endings:
/// - abort: immediate teardown, no trace
/// - cancel: drain → finalize → history entry, IME clears
/// - commit: drain → finalize → IME commits text
actor Session {
    let id: UUID
    let targetBundleID: String?
    let createdAt: Date

    // Layer states — observable by the debug overlay
    private(set) var capture: CaptureState = .idle
    private(set) var asr: ASRState = .idle
    private(set) var ime: IMEState = .inactive

    // Dependencies
    private let audioEngine: AudioEngine
    private let transcriptionService: TranscriptionService
    private let inputClient: BeeInputClient

    // Internal
    private var processedNativeCount: Int = 0
    private var asrSession: StreamingSession?
    private var streamingTask: Task<Void, Never>?
    private var partialTranscript: String = ""

    // Callbacks
    private var onComplete: (@Sendable (SessionResult) -> Void)?

    // Diagnostics
    private(set) var diag = SessionDiagnostics()

    func setOnComplete(_ handler: @Sendable @escaping (SessionResult) -> Void) {
        onComplete = handler
    }

    init(
        audioEngine: AudioEngine,
        transcriptionService: TranscriptionService,
        inputClient: BeeInputClient,
        targetBundleID: String?
    ) {
        self.id = UUID()
        self.createdAt = Date()
        self.audioEngine = audioEngine
        self.transcriptionService = transcriptionService
        self.inputClient = inputClient
        self.targetBundleID = targetBundleID
    }

    // MARK: - Start

    func start(language: String?) async {
        logger.info("[\(self.id)] Starting session")

        // Warm up engine if cold
        if !audioEngine.isWarm {
            do { try audioEngine.warmUp() }
            catch {
                logger.error("[\(self.id)] Failed to warm up: \(error)")
                onComplete?(.aborted(id: id))
                return
            }
        }

        // Capture: start (copies pre-buffer)
        audioEngine.startCapture(for: self.id)
        capture = .buffering
        diag.startedAt = Date()
        diag.nativeRate = audioEngine.nativeSampleRate

        // IME: activate
        await MainActor.run { inputClient.activate() }
        ime = .active

        // ASR: create session
        asrSession = transcriptionService.createSession(language: language)
        asr = asrSession != nil ? .streaming : .idle

        // Start the streaming loop — runs until capture is delivered
        streamingTask = Task { [weak self] in
            await self?.streamingLoop()
        }
    }

    // MARK: - Endings

    /// Immediate teardown. No drain, no finalize, no history.
    func abort() async {
        logger.info("[\(self.id)] Aborting")

        // Kill streaming loop
        streamingTask?.cancel()

        // Capture: discard
        audioEngine.cancelCapture(for: self.id)
        capture = .discarded

        // ASR: drop
        asrSession = nil
        asr = .done

        // IME: deactivate without committing
        await MainActor.run { inputClient.deactivate() }
        ime = .tornDown

        onComplete?(.aborted(id: id))
    }

    enum EndMode: Sendable {
        case commit(submit: Bool)
        case cancel
    }

    func commit(submit: Bool) async { await end(.commit(submit: submit)) }
    func cancel() async { await end(.cancel) }

    /// Shared ending flow for commit and cancel.
    ///
    /// 1. Signal capture to drain (VAD tail monitoring)
    /// 2. Streaming loop continues during drain, feeding audio to ASR
    /// 3. Drain completes → capture = delivered → streaming loop exits
    /// 4. Finalize ASR with remaining samples
    /// 5. Branch: commit → IME commits | cancel → IME clears
    private func end(_ mode: EndMode) async {
        let modeLabel: String
        switch mode {
        case .commit(let submit): modeLabel = submit ? "commit+submit" : "commit"
        case .cancel: modeLabel = "cancel"
        }
        logger.info("[\(self.id)] Ending: \(modeLabel)")

        diag.endedAt = Date()
        diag.ending = modeLabel

        // 1. Signal capture to drain — streaming loop keeps running
        audioEngine.beginDrain(for: self.id)
        capture = .draining

        // 2. Wait for streaming loop to finish
        //    (it exits when capture becomes .delivered)
        await streamingTask?.value
        streamingTask = nil

        // 3. Collect all captured samples
        let allNativeSamples = audioEngine.collectSamples(for: self.id)
        capture = .delivered
        diag.totalNativeSamples = allNativeSamples.count

        // 4. Finalize ASR
        let finalText = await finalizeASR(allNativeSamples: allNativeSamples)

        // 5. Save audio for debugging
        saveDebugAudio(nativeSamples: allNativeSamples, finalText: finalText)

        // 6. Branch on mode
        switch mode {
        case .commit(let submit):
            if !finalText.isEmpty {
                inputClient.commitText(finalText)
                try? await Task.sleep(for: .milliseconds(50))
                await MainActor.run { inputClient.deactivate() }
                ime = .committed

                if submit {
                    try? await Task.sleep(for: .milliseconds(50))
                    inputClient.simulateReturn()
                }
            } else {
                await MainActor.run { inputClient.deactivate() }
                ime = .committed
            }
            onComplete?(.committed(id: id, text: finalText, submitted: mode.isSubmit))

        case .cancel:
            inputClient.clearMarkedText()
            await MainActor.run { inputClient.deactivate() }
            ime = .cleared
            onComplete?(.cancelled(id: id, text: finalText))
        }
    }

    // MARK: - Streaming Loop

    /// Runs continuously, feeding audio from the capture layer to the ASR.
    /// Keeps running during drain — only exits when capture is delivered
    /// (drain complete) or the task is cancelled (abort).
    private func streamingLoop() async {
        guard let session = asrSession else { return }
        let nativeRate = audioEngine.nativeSampleRate
        let minNativeChunk = Int(nativeRate * 0.05)

        while !Task.isCancelled {
            // Check if capture is done (drain completed, samples collected)
            if audioEngine.isDrained(for: self.id) {
                // One final peek to get everything including drain tail
                let allNative = audioEngine.peekCapture(for: self.id)
                let newCount = allNative.count
                if newCount > processedNativeCount {
                    let chunk = Array(allNative[processedNativeCount...])
                    processedNativeCount = newCount
                    let resampled = AudioEngine.resample(chunk, from: nativeRate)
                    diag.streamingFeeds += 1
                    diag.streamedNativeSamples = processedNativeCount
                    diag.streamedResampledSamples += resampled.count
                    if let update = transcriptionService.feed(session: session, samples: resampled) {
                        partialTranscript = update.text
                        inputClient.setMarkedText(update.text)
                    }
                }
                break // drain complete — exit loop
            }

            let allNative = audioEngine.peekCapture(for: self.id)
            let newCount = allNative.count

            guard newCount > processedNativeCount + minNativeChunk else {
                try? await Task.sleep(for: .milliseconds(30))
                continue
            }

            let chunk = Array(allNative[processedNativeCount...])
            processedNativeCount = newCount
            let resampled = AudioEngine.resample(chunk, from: nativeRate)

            diag.streamingFeeds += 1
            diag.streamedNativeSamples = processedNativeCount
            diag.streamedResampledSamples += resampled.count

            if let update = transcriptionService.feed(session: session, samples: resampled) {
                partialTranscript = update.text
                inputClient.setMarkedText(update.text)
            }
        }
    }

    // MARK: - ASR Finalization

    private func finalizeASR(allNativeSamples: [Float]) async -> String {
        guard let session = asrSession else {
            asr = .done
            return partialTranscript
        }

        asr = .finalizing
        diag.finalizeStartedAt = Date()

        let nativeRate = audioEngine.nativeSampleRate
        let remainingNative = allNativeSamples.count > processedNativeCount
            ? Array(allNativeSamples[processedNativeCount...])
            : []

        diag.remainingNativeSamples = remainingNative.count

        // Feed any remaining samples via feedFinalizing
        if !remainingNative.isEmpty {
            var resampled = AudioEngine.resample(remainingNative, from: nativeRate)
            diag.remainingResampledSamples = resampled.count

            // Silence padding (100ms) to signal end of speech
            let pad = Int(AudioEngine.targetSampleRate * 0.1)
            resampled.append(contentsOf: repeatElement(Float(0), count: pad))

            if let update = transcriptionService.feedFinalizing(session: session, samples: resampled) {
                partialTranscript = update.text
            }
        }

        // Final inference
        let result = transcriptionService.finish(session: session)
        let finalText: String
        if let text = result?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            finalText = text
        } else {
            finalText = partialTranscript
        }

        asr = .done
        asrSession = nil
        diag.finalizeEndedAt = Date()
        diag.finalText = finalText

        logger.info("[\(self.id)] Finalized: \"\(finalText.prefix(80))\"")
        return finalText
    }

    // MARK: - Debug

    private func saveDebugAudio(nativeSamples: [Float], finalText: String) {
        let nativeRate = audioEngine.nativeSampleRate
        let allResampled = AudioEngine.resample(nativeSamples, from: nativeRate)
        let debugDir = FileManager.default.temporaryDirectory.appendingPathComponent("bee-debug")
        try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
        let wavURL = debugDir.appendingPathComponent("\(id.uuidString.prefix(8)).wav")
        try? WavWriter.write(samples: allResampled, to: wavURL)
        diag.audioWavPath = wavURL.path
    }
}

// MARK: - Layer States

extension Session {
    enum CaptureState: Sendable, CustomStringConvertible {
        case idle, buffering, draining, delivered, discarded
        var description: String {
            switch self {
            case .idle: "idle"
            case .buffering: "buffering"
            case .draining: "draining"
            case .delivered: "delivered"
            case .discarded: "discarded"
            }
        }
    }

    enum ASRState: Sendable, CustomStringConvertible {
        case idle, streaming, finalizing, done
        var description: String {
            switch self {
            case .idle: "idle"
            case .streaming: "streaming"
            case .finalizing: "finalizing"
            case .done: "done"
            }
        }
    }

    enum IMEState: Sendable, CustomStringConvertible {
        case inactive, active, parked, committed, cleared, tornDown
        var description: String {
            switch self {
            case .inactive: "inactive"
            case .active: "active"
            case .parked: "parked"
            case .committed: "committed"
            case .cleared: "cleared"
            case .tornDown: "torn down"
            }
        }
    }
}

// MARK: - EndMode helpers

extension Session.EndMode {
    var isSubmit: Bool {
        if case .commit(let submit) = self { return submit }
        return false
    }
}

// MARK: - Supporting types

struct StreamingUpdate: Sendable {
    let text: String
    let committedUTF16Count: Int
    let detectedLanguage: String?
}

struct SessionDiagnostics: Sendable {
    var startedAt: Date?
    var endedAt: Date?
    var ending: String = ""
    var nativeRate: Double = 0

    var streamingFeeds: Int = 0
    var streamedNativeSamples: Int = 0
    var streamedResampledSamples: Int = 0

    var totalNativeSamples: Int = 0

    var remainingNativeSamples: Int = 0
    var remainingResampledSamples: Int = 0
    var finalizeStartedAt: Date?
    var finalizeEndedAt: Date?
    var finalText: String = ""
    var audioWavPath: String = ""

    var recordingDurationMs: Int {
        guard let s = startedAt, let e = endedAt else { return 0 }
        return Int((e.timeIntervalSince(s) * 1000).rounded())
    }

    var finalizeDurationMs: Int {
        guard let s = finalizeStartedAt, let e = finalizeEndedAt else { return 0 }
        return Int((e.timeIntervalSince(s) * 1000).rounded())
    }

    var totalNativeDurationMs: Int {
        guard nativeRate > 0 else { return 0 }
        return Int((Double(totalNativeSamples) / nativeRate * 1000).rounded())
    }

    var remainingNativeDurationMs: Int {
        guard nativeRate > 0 else { return 0 }
        return Int((Double(remainingNativeSamples) / nativeRate * 1000).rounded())
    }
}

enum SessionResult: Sendable {
    case aborted(id: UUID)
    case cancelled(id: UUID, text: String)
    case committed(id: UUID, text: String, submitted: Bool)
}
