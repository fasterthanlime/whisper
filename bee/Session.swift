import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "Session")

/// A Session is a self-contained unit of work for a single dictation attempt.
///
/// Three tasks connected by three channels:
///
/// ```
/// AudioEngine ──Channel 0──→ Capture Task ──Channel 1──→ ASR Task ──Channel 2──→ Session actor
///               [Float]       (VAD, drain)    AudioChunk   (Rust FFI)  TranscriptEvent  (IME)
/// ```
///
/// End mode flows forward through all three channels.
///
/// Three endings:
/// - abort: all channels cancelled, no finalize
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

    // Channels
    private var ch0: RawAudioPipeline?
    private var ch1: AudioPipeline?
    private var ch2: TranscriptPipeline?

    // Drain signal — set by Session, read by capture task
    private let drainSignal = DrainSignal()

    // Tasks
    private var captureTask: Task<Void, Never>?
    private var asrTask: Task<Void, Never>?
    private var consumerTask: Task<Void, Never>?

    // Callbacks
    private var onComplete: (@Sendable (SessionResult) -> Void)?

    // Diagnostics — lock-protected, written by tasks, read by debug overlay
    let diag = SessionDiag()

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

        diag.update { $0.startedAt = Date() }

        // Create all three channels
        let ch0 = RawAudioPipeline()
        let ch1 = AudioPipeline()
        let ch2 = TranscriptPipeline()
        self.ch0 = ch0
        self.ch1 = ch1
        self.ch2 = ch2

        // Register with AudioEngine (Channel 0 starts flowing)
        audioEngine.startCapture(for: self.id, pipeline: ch0)
        capture = .buffering

        // IME: activate
        await MainActor.run { inputClient.activate() }
        ime = .active

        // ASR: create session
        let asrSession = transcriptionService.createSession(language: language)
        asr = asrSession != nil ? .streaming : .idle

        // --- Capture Task ---
        // Reads Channel 0, forwards to Channel 1.
        // When drain is signaled: does VAD, then sends .end(mode) on Channel 1
        // and stops Channel 0. The capture task owns Channel 0's lifecycle.
        let sessionID = self.id
        let engine = self.audioEngine
        let diagRef = self.diag
        let drain = self.drainSignal

        // VAD parameters (at 16kHz)
        let vadSilenceThreshold: Float = 0.008
        let vadRequiredSilenceSamples = Int(AudioEngine.targetSampleRate * 0.15)
        let vadDrainTimeoutSamples = Int(AudioEngine.targetSampleRate * 0.5)

        captureTask = Task.detached {
            var allSamples: [Float] = []
            var silenceSamples = 0
            var drainSamplesRemaining = 0
            var drainMode: EndMode?
            var drainBufferCount = 0
            var drainSampleCount = 0

            for await _ in ch0.stream {
                let buffers = ch0.drain()
                for buf in buffers {
                    allSamples.append(contentsOf: buf)
                    ch1.send(.samples(buf))

                    // Check if drain was requested
                    if drainMode == nil, let mode = drain.get() {
                        drainMode = mode
                        drainSamplesRemaining = vadDrainTimeoutSamples
                    }

                    // If draining, do VAD
                    if drainMode != nil {
                        drainBufferCount += 1
                        drainSampleCount += buf.count

                        let rms = Self.computeRMS(buf)
                        if rms < vadSilenceThreshold {
                            silenceSamples += buf.count
                        } else {
                            silenceSamples = 0
                        }
                        drainSamplesRemaining -= buf.count

                        let reachedSilence = silenceSamples >= vadRequiredSilenceSamples
                        let reachedTimeout = drainSamplesRemaining <= 0

                        if reachedSilence || reachedTimeout {
                            diagRef.update {
                                $0.drainBuffers = drainBufferCount
                                $0.drainSamples = drainSampleCount
                            }
                            break
                        }
                    }
                }

                // If we broke out of the inner loop, we're done draining
                if let mode = drainMode, (silenceSamples >= vadRequiredSilenceSamples || drainSamplesRemaining <= 0) {
                    diagRef.update {
                        $0.capturedSamples = allSamples.count
                        $0.totalSamples = allSamples.count
                    }

                    // Save debug WAV
                    let debugDir = FileManager.default.temporaryDirectory.appendingPathComponent("bee-debug")
                    try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
                    let wavURL = debugDir.appendingPathComponent("\(sessionID.uuidString.prefix(8)).wav")
                    try? WavWriter.write(samples: allSamples, to: wavURL)
                    diagRef.update { $0.audioWavPath = wavURL.path }

                    // Send end on Channel 1, stop Channel 0
                    ch1.send(.end(mode))
                    ch1.finish()
                    engine.stopCapture(for: sessionID)
                    return
                }
            }

            // Channel 0 ended without drain completing — abort path
            // Drain any remaining buffers
            let remaining = ch0.drain()
            for buf in remaining {
                allSamples.append(contentsOf: buf)
                ch1.send(.samples(buf))
            }
            diagRef.update {
                $0.capturedSamples = allSamples.count
                $0.totalSamples = allSamples.count
            }

            // Save debug WAV even on abort
            let debugDir = FileManager.default.temporaryDirectory.appendingPathComponent("bee-debug")
            try? FileManager.default.createDirectory(at: debugDir, withIntermediateDirectories: true)
            let wavURL = debugDir.appendingPathComponent("\(sessionID.uuidString.prefix(8)).wav")
            try? WavWriter.write(samples: allSamples, to: wavURL)
            diagRef.update { $0.audioWavPath = wavURL.path }
        }

        // --- ASR Task ---
        // Reads Channel 1, feeds Rust FFI, writes to Channel 2.
        let ts = self.transcriptionService

        asrTask = Task.detached {
            guard let asrSession else {
                // No ASR — just forward end mode
                for await _ in ch1.stream {
                    for chunk in ch1.drain() {
                        if case .end(let mode) = chunk {
                            ch2.send(.done(text: "", mode: mode))
                            ch2.finish()
                            return
                        }
                    }
                }
                return
            }

            for await _ in ch1.stream {
                let chunks = ch1.drain()

                // Batch all samples, check for end
                var batch: [Float] = []
                var endMode: EndMode?
                for chunk in chunks {
                    switch chunk {
                    case .samples(let s):
                        batch.append(contentsOf: s)
                    case .end(let mode):
                        endMode = mode
                    }
                }

                // Feed accumulated audio
                if !batch.isEmpty {
                    let t0 = ProcessInfo.processInfo.systemUptime
                    let update: StreamingUpdate?
                    if endMode != nil {
                        // Last feed — use feedFinalizing
                        update = ts.feedFinalizing(session: asrSession, samples: batch)
                    } else {
                        update = ts.feed(session: asrSession, samples: batch)
                    }
                    let feedUs = Int((ProcessInfo.processInfo.systemUptime - t0) * 1_000_000)

                    diagRef.update {
                        $0.feeds += 1
                        $0.lastFeedUs = feedUs
                        $0.totalFeedUs += feedUs
                        $0.fedSamples += batch.count
                    }

                    if let update {
                        ch2.send(.partial(update.text))
                    }
                }

                // End of stream — finalize
                if let endMode {
                    let t0 = ProcessInfo.processInfo.systemUptime
                    let result = ts.finish(session: asrSession)
                    let finalizeUs = Int((ProcessInfo.processInfo.systemUptime - t0) * 1_000_000)

                    let finalText = result?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    diagRef.update {
                        $0.finalizeUs = finalizeUs
                        $0.finalText = finalText
                    }

                    logger.info("[\(sessionID)] Finalized: \"\(finalText.prefix(80))\"")
                    ch2.send(.done(text: finalText, mode: endMode))
                    ch2.finish()
                    return
                }
            }

            // Channel 1 ended without .end — abort path
        }

        // --- Consumer Task ---
        // Reads Channel 2, updates IME. Runs on this actor.
        let ic = self.inputClient

        consumerTask = Task {
            var lastPartial = ""

            for await _ in ch2.stream {
                let events = ch2.drain()
                for event in events {
                    switch event {
                    case .partial(let text):
                        lastPartial = text
                        ic.setMarkedText(text)

                    case .done(let text, let mode):
                        let finalText = text.isEmpty ? lastPartial : text
                        diag.update { $0.finalText = finalText }

                        await self.finishIME(text: finalText, mode: mode)
                        return
                    }
                }
            }

            // Channel 2 ended without .done — abort path
        }
    }

    // MARK: - Endings

    /// Immediate teardown. No drain, no finalize, no history.
    func abort() async {
        logger.info("[\(self.id)] Aborting")

        // Stop Channel 0 — capture task will exit and won't send .end
        audioEngine.stopCapture(for: self.id)
        // Close downstream channels — ASR and consumer tasks exit
        ch1?.finish()
        ch2?.finish()
        capture = .discarded

        // Wait for all tasks
        await captureTask?.value
        await asrTask?.value
        await consumerTask?.value
        captureTask = nil
        asrTask = nil
        consumerTask = nil
        asr = .done

        // IME: deactivate
        await MainActor.run { inputClient.deactivate() }
        ime = .tornDown

        onComplete?(.aborted(id: id))
    }

    func commit(submit: Bool) async { await end(.commit(submit: submit)) }
    func cancel() async { await end(.cancel) }

    /// Shared ending flow for commit and cancel.
    ///
    /// Tells the capture task to drain, then waits for the result to
    /// flow all the way through Channel 1 → Channel 2.
    private func end(_ mode: EndMode) async {
        let modeLabel: String
        switch mode {
        case .commit(let submit): modeLabel = submit ? "commit+submit" : "commit"
        case .cancel: modeLabel = "cancel"
        }
        logger.info("[\(self.id)] Ending: \(modeLabel)")

        diag.update {
            $0.endedAt = Date()
            $0.ending = modeLabel
        }
        capture = .draining

        // Begin drain: the capture task will monitor VAD, then send
        // .end(mode) on Channel 1 when silence is detected.
        beginDrain(mode: mode)

        // Wait for the consumer task — it exits when it sees .done on Channel 2
        await consumerTask?.value
        consumerTask = nil

        // Also wait for the other tasks to clean up
        await captureTask?.value
        await asrTask?.value
        captureTask = nil
        asrTask = nil
        capture = .delivered
        asr = .done
    }

    // MARK: - Drain

    /// Signals the capture task to start VAD monitoring.
    /// The capture task will send .end(mode) on Channel 1 and stop
    /// Channel 0 when silence is detected or timeout is reached.
    private func beginDrain(mode: EndMode) {
        drainSignal.set(mode)
    }

    // MARK: - Helpers

    static func computeRMS(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return sqrtf(sum / Float(samples.count))
    }

    // MARK: - IME Finalization

    private func finishIME(text: String, mode: EndMode) async {
        switch mode {
        case .commit(let submit):
            if !text.isEmpty {
                inputClient.commitText(text)
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
            onComplete?(.committed(id: id, text: text, submitted: submit))

        case .cancel:
            inputClient.clearMarkedText()
            await MainActor.run { inputClient.deactivate() }
            ime = .cleared
            onComplete?(.cancelled(id: id, text: text))
        }
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

// MARK: - Supporting types

struct StreamingUpdate: Sendable {
    let text: String
    let committedUTF16Count: Int
    let detectedLanguage: String?
}

/// Thread-safe diagnostics container. Written by pipeline tasks,
/// read by the debug overlay. No actor hops needed.
final class SessionDiag: @unchecked Sendable {
    struct Snapshot: Sendable {
        var startedAt: Date?
        var endedAt: Date?
        var ending: String = ""

        var capturedSamples: Int = 0
        var totalSamples: Int = 0
        var feeds: Int = 0
        var fedSamples: Int = 0
        var lastFeedUs: Int = 0
        var totalFeedUs: Int = 0

        var drainBuffers: Int = 0
        var drainSamples: Int = 0

        var finalizeUs: Int = 0
        var finalText: String = ""
        var audioWavPath: String = ""

        var recordingDurationMs: Int {
            guard let s = startedAt, let e = endedAt else { return 0 }
            return Int((e.timeIntervalSince(s) * 1000).rounded())
        }

        var totalAudioDurationMs: Int {
            guard totalSamples > 0 else { return 0 }
            return Int((Double(totalSamples) / AudioEngine.targetSampleRate * 1000).rounded())
        }
    }

    private let lock = NSLock()
    private var _snapshot = Snapshot()

    var snapshot: Snapshot {
        lock.withLock { _snapshot }
    }

    func update(_ body: (inout Snapshot) -> Void) {
        lock.withLock { body(&_snapshot) }
    }
}

/// Thread-safe signal from Session to the capture task.
/// Session sets the end mode; capture task polls it on each buffer.
final class DrainSignal: @unchecked Sendable {
    private let lock = NSLock()
    private var mode: EndMode?

    func set(_ mode: EndMode) {
        lock.withLock { self.mode = mode }
    }

    func get() -> EndMode? {
        lock.withLock { mode }
    }
}

enum SessionResult: Sendable {
    case aborted(id: UUID)
    case cancelled(id: UUID, text: String)
    case committed(id: UUID, text: String, submitted: Bool)
}
