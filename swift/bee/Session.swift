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
    let targetProcessID: pid_t?
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
    private var didComplete = false

    // Diagnostics — lock-protected, written by tasks, read by debug overlay
    let diag = SessionDiag()
    private let textSnapshot = SessionTextSnapshot()

    func setOnComplete(_ handler: @Sendable @escaping (SessionResult) -> Void) {
        onComplete = handler
    }

    init(
        audioEngine: AudioEngine,
        transcriptionService: TranscriptionService,
        inputClient: BeeInputClient,
        targetProcessID: pid_t?
    ) {
        self.id = UUID()
        self.createdAt = Date()
        self.audioEngine = audioEngine
        self.transcriptionService = transcriptionService
        self.inputClient = inputClient
        self.targetProcessID = targetProcessID
    }

    // MARK: - Start

    func start(language: String?, asrConfig: TranscriptionService.SessionConfig) async {
        logger.info("[\(self.id)] Starting session")

        // Warm up engine if cold
        if !audioEngine.isWarm {
            do { try audioEngine.warmUp() } catch {
                logger.error("[\(self.id)] Failed to warm up: \(error)")
                emitCompletion(.aborted(id: id))
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

        // IME: activate and show bee cursor
        beeLog("SESSION START")
        let imeActivated = await inputClient.activate(sessionID: id)
        guard imeActivated else {
            logger.error("[\(self.id)] Failed to activate IME for target pid")
            inputClient.stopDictating(sessionID: id)
            emitCompletion(.aborted(id: id))
            return
        }
        beeLog("SESSION: IME selected, awaiting IME session confirmation")
        ime = .parked

        // Register with AudioEngine (Channel 0 starts flowing)
        audioEngine.startCapture(for: self.id, pipeline: ch0)
        capture = .buffering

        // ASR: create session
        var config = asrConfig
        config.language = language
        let asrSession = transcriptionService.createSession(config)
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
                if let mode = drainMode,
                    silenceSamples >= vadRequiredSilenceSamples || drainSamplesRemaining <= 0
                {
                    diagRef.update {
                        $0.capturedSamples = allSamples.count
                        $0.totalSamples = allSamples.count
                    }

                    // Save debug WAV
                    let debugDir = FileManager.default.temporaryDirectory.appendingPathComponent(
                        "bee-debug")
                    try? FileManager.default.createDirectory(
                        at: debugDir, withIntermediateDirectories: true)
                    let wavURL = debugDir.appendingPathComponent(
                        "\(sessionID.uuidString.prefix(8)).wav")
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
            let debugDir = FileManager.default.temporaryDirectory.appendingPathComponent(
                "bee-debug")
            try? FileManager.default.createDirectory(
                at: debugDir, withIntermediateDirectories: true)
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
        // Reads Channel 2, updates IME with typewriter animation.
        let ic = self.inputClient

        consumerTask = Task {
            var displayedText = ""
            var targetText = ""

            for await _ in ch2.stream {
                let events = ch2.drain()

                // If a .done is in this batch, skip animation and go straight to commit
                let hasDone = events.contains {
                    if case .done = $0 { return true }
                    return false
                }

                for event in events {
                    switch event {
                    case .partial(let text):
                        targetText = text
                        textSnapshot.set(targetText)

                        if hasDone || displayedText == targetText { break }

                        // Matrix-style morph: randomly alternate between
                        // appending the next char and fixing a wrong char in-place
                        var chars = Array(displayedText)
                        let target = Array(targetText)

                        var steps = 0
                        while chars != target {
                            if !ch2.isEmpty { break }
                            if Task.isCancelled { return }

                            // Collect available actions
                            let canAppend = chars.count < target.count
                            let canTrim = chars.count > target.count
                            let wrongIndices: [Int] = (0..<min(chars.count, target.count))
                                .filter { chars[$0] != target[$0] }

                            if wrongIndices.isEmpty && !canAppend && !canTrim { break }

                            // Randomly pick action: morph a wrong char, append, or trim
                            let morphWeight = wrongIndices.count
                            let appendWeight = canAppend ? max(1, wrongIndices.count / 2) : 0
                            let trimWeight = canTrim ? max(1, wrongIndices.count / 2) : 0
                            let total = morphWeight + appendWeight + trimWeight
                            let roll = Int.random(in: 0..<max(total, 1))

                            if roll < morphWeight && !wrongIndices.isEmpty {
                                // Fix a random wrong character in-place
                                let idx = wrongIndices.randomElement()!
                                chars[idx] = target[idx]
                            } else if roll < morphWeight + appendWeight && canAppend {
                                // Append next correct character
                                chars.append(target[chars.count])
                            } else if canTrim {
                                // Remove last character
                                chars.removeLast()
                            } else if canAppend {
                                chars.append(target[chars.count])
                            } else if !wrongIndices.isEmpty {
                                let idx = wrongIndices.randomElement()!
                                chars[idx] = target[idx]
                            }

                            displayedText = String(chars)
                            textSnapshot.set(displayedText)
                            await self.renderMarkedTextIfActive(
                                Self.addCursor(displayedText),
                                inputClient: ic,
                                sessionID: sessionID
                            )

                            steps += 1
                            let delayMs = steps > 20 ? 8 : (steps > 10 ? 15 : 25)
                            try? await Task.sleep(for: .milliseconds(delayMs))
                        }

                        // Snap to target in case animation was interrupted
                        displayedText = targetText
                        textSnapshot.set(displayedText)
                        await self.renderMarkedTextIfActive(
                            Self.addCursor(displayedText),
                            inputClient: ic,
                            sessionID: sessionID
                        )

                    case .done(let text, let mode):
                        let finalText = text.isEmpty ? targetText : text
                        diag.update { $0.finalText = finalText }
                        textSnapshot.set(finalText)

                        // Snap to final text (no cursor)
                        await self.renderMarkedTextIfActive(
                            finalText,
                            inputClient: ic,
                            sessionID: sessionID
                        )
                        await self.finishIME(text: finalText, mode: mode)
                        return
                    }
                }
            }

            // Channel 2 ended without .done — abort path
        }
    }

    // MARK: - Endings

    func park() async {
        guard ime == .active else { return }
        ime = .parked
        beeLog("SESSION: park id=\(id.uuidString.prefix(8))")
        await MainActor.run { inputClient.deactivate() }
    }

    @discardableResult
    func requestResumeActivation() async -> Bool {
        guard ime == .parked else { return true }
        let activated = await inputClient.activate(sessionID: id)
        if !activated {
            beeLog("SESSION: resume activation failed id=\(id.uuidString.prefix(8))")
        } else {
            beeLog("SESSION: resume activation requested id=\(id.uuidString.prefix(8))")
        }
        return activated
    }

    func routeDidBecomeActive() {
        guard ime == .parked else { return }
        ime = .active
        let snapshot = textSnapshot.get()
        inputClient.setMarkedText(Self.addCursor(snapshot), sessionID: id)
        beeLog("SESSION: resumed id=\(id.uuidString.prefix(8))")
    }

    @discardableResult
    func resume() async -> Bool {
        let activated = await requestResumeActivation()
        guard activated else { return false }
        routeDidBecomeActive()
        return true
    }

    func liveText() -> String {
        textSnapshot.get()
    }

    /// Immediate commit path for manual typing takeover.
    /// Skips ASR finalization and commits the latest rendered snapshot.
    func immediateCommitFromTyping() async {
        guard !didComplete else { return }
        beeLog("SESSION: immediate commit id=\(id.uuidString.prefix(8))")

        let text = textSnapshot.get()
        ch2?.finish()
        consumerTask?.cancel()
        await consumerTask?.value
        consumerTask = nil

        let result: SessionResult
        if !text.isEmpty {
            inputClient.commitText(text, sessionID: id)
            ime = .committed
            result = .committed(id: id, text: text, submitted: false)
        } else {
            inputClient.clearMarkedText(sessionID: id)
            inputClient.stopDictating(sessionID: id)
            ime = .cleared
            result = .cancelled(id: id, text: "")
        }

        await MainActor.run { inputClient.deactivate() }
        emitCompletion(result)

        audioEngine.stopCapture(for: self.id)
        ch1?.finish()
        captureTask?.cancel()
        asrTask?.cancel()
        await captureTask?.value
        await asrTask?.value
        captureTask = nil
        asrTask = nil
        capture = .discarded
        asr = .done
    }

    /// Immediate teardown. No drain, no finalize, no history.
    func abort() async {
        guard !didComplete else { return }
        logger.info("[\(self.id)] Aborting")

        await terminatePipelines(cancelTasks: true)
        capture = .discarded
        asr = .done

        // IME: deactivate
        inputClient.stopDictating(sessionID: id)
        await MainActor.run { inputClient.deactivate() }
        ime = .tornDown

        emitCompletion(.aborted(id: id))
    }

    func commit(submit: Bool) async {
        guard !didComplete else { return }
        await end(.commit(submit: submit))
    }

    func cancel() async {
        guard !didComplete else { return }
        await end(.cancel)
    }

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

    private func terminatePipelines(cancelTasks: Bool) async {
        audioEngine.stopCapture(for: self.id)
        ch1?.finish()
        ch2?.finish()

        if cancelTasks {
            captureTask?.cancel()
            asrTask?.cancel()
            consumerTask?.cancel()
        }

        await captureTask?.value
        await asrTask?.value
        await consumerTask?.value
        captureTask = nil
        asrTask = nil
        consumerTask = nil
    }

    // MARK: - Drain

    /// Signals the capture task to start VAD monitoring.
    /// The capture task will send .end(mode) on Channel 1 and stop
    /// Channel 0 when silence is detected or timeout is reached.
    private func beginDrain(mode: EndMode) {
        drainSignal.set(mode)
    }

    // MARK: - Helpers

    static func addCursor(_ text: String) -> String {
        var t = text
        if t.hasSuffix(".") || t.hasSuffix("。") {
            t = String(t.dropLast())
        }
        return t.isEmpty ? "🐝" : "\(t) 🐝"
    }

    private func renderMarkedTextIfActive(
        _ text: String, inputClient: BeeInputClient, sessionID: UUID
    ) async {
        guard ime == .active else { return }
        inputClient.setMarkedText(text, sessionID: sessionID)
    }

    // MARK: - Shortest Edit Script (LCS-based)

    enum EditOp {
        case keep
        case delete
        case insert(Character)
    }

    /// Compute the shortest edit script to transform `from` into `to` using
    /// character-level LCS (longest common subsequence). Returns a sequence of
    /// keep/delete/insert operations that, applied left-to-right, morph `from` into `to`.
    static func shortestEdit(from old: String, to new: String) -> [EditOp] {
        let a = Array(old)
        let b = Array(new)
        let m = a.count
        let n = b.count

        // DP table for LCS lengths — full table needed for backtrace
        // For strings up to ~500 chars this is fine
        var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        for i in 1...max(m, 1) {
            for j in 1...max(n, 1) {
                if i <= m && j <= n && a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else if i <= m && j <= n {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        // Backtrace to produce edit script
        var ops: [EditOp] = []
        var i = m
        var j = n
        while i > 0 || j > 0 {
            if i > 0 && j > 0 && a[i - 1] == b[j - 1] {
                ops.append(.keep)
                i -= 1
                j -= 1
            } else if j > 0 && (i == 0 || dp[i][j - 1] >= dp[i - 1][j]) {
                ops.append(.insert(b[j - 1]))
                j -= 1
            } else {
                ops.append(.delete)
                i -= 1
            }
        }
        ops.reverse()
        return ops
    }

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
                inputClient.commitText(text, sessionID: id)
                try? await Task.sleep(for: .milliseconds(50))
                await MainActor.run { inputClient.deactivate() }
                ime = .committed

                if submit {
                    try? await Task.sleep(for: .milliseconds(50))
                    inputClient.simulateReturn()
                }
            } else {
                inputClient.stopDictating(sessionID: id)
                await MainActor.run { inputClient.deactivate() }
                ime = .committed
            }
            emitCompletion(.committed(id: id, text: text, submitted: submit))

        case .cancel:
            inputClient.clearMarkedText(sessionID: id)
            await MainActor.run { inputClient.deactivate() }
            ime = .cleared
            emitCompletion(.cancelled(id: id, text: text))
        }
    }

    private func emitCompletion(_ result: SessionResult) {
        guard !didComplete else { return }
        didComplete = true
        onComplete?(result)
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
    let alignmentsJSON: String?
    let debugJSON: String?
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

        var lastRms: Float = 0
        var peakRms: Float = 0

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

final class SessionTextSnapshot: @unchecked Sendable {
    private let lock = NSLock()
    private var text = ""

    func set(_ text: String) {
        lock.withLock { self.text = text }
    }

    func get() -> String {
        lock.withLock { text }
    }
}

extension String {
    func commonPrefixLength(with other: String) -> Int {
        var count = 0
        var i = self.startIndex
        var j = other.startIndex
        while i < self.endIndex && j < other.endIndex && self[i] == other[j] {
            count += 1
            i = self.index(after: i)
            j = other.index(after: j)
        }
        return count
    }
}

enum SessionResult: Sendable {
    case aborted(id: UUID)
    case cancelled(id: UUID, text: String)
    case committed(id: UUID, text: String, submitted: Bool)
}

/// Log to the shared bee log file.
func beeLog(_ msg: String) {
    let ts = ProcessInfo.processInfo.systemUptime
    let line = String(format: "[%.3f] APP: %@\n", ts, msg)
    if let data = line.data(using: .utf8),
        let fh = FileHandle(forWritingAtPath: "/tmp/bee.log")
    {
        fh.seekToEndOfFile()
        fh.write(data)
        fh.closeFile()
    } else if let data = line.data(using: .utf8) {
        try? data.write(to: URL(fileURLWithPath: "/tmp/bee.log"))
    }
}

/// Simple file logger for debugging IME text flow.
private final class IMELog: Sendable {
    private let path = "/tmp/bee.log"

    init() {
        // Don't truncate — shared log file
    }

    func write(_ msg: String) {
        let ts = ProcessInfo.processInfo.systemUptime
        let line = String(format: "[%.3f] APP/IME: %@\n", ts, msg)
        if let data = line.data(using: .utf8),
            let fh = FileHandle(forWritingAtPath: path)
        {
            fh.seekToEndOfFile()
            fh.write(data)
            fh.closeFile()
        }
    }
}
