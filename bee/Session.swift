import Foundation

// MARK: - Session

// h[impl session.abort]
// h[impl session.cancel]
// h[impl session.commit]
/// A Session is a self-contained unit of work for a single dictation attempt.
/// It owns three layers — Capture, ASR, and IME — each with its own state.
/// Multiple sessions can coexist (e.g., the previous one finalizing while a
/// new one is streaming).
actor Session {
    let id: UUID
    let targetBundleID: String?
    let createdAt: Date

    private(set) var capture: CaptureState = .buffering
    private(set) var asr: ASRState = .idle
    private(set) var ime: IMEState = .inactive

    private let audioEngine: AudioEngine
    private let transcriptionService: TranscriptionService
    private let inputClient: BeeInputClient

    private var capturedSamples: [Float] = []
    private var processedSampleCount: Int = 0
    private var asrSession: OpaquePointer?
    private var streamingTask: Task<Void, Never>?
    private var finalText: String = ""
    private var partialTranscript: String = ""

    /// Called on every streaming update (on the actor).
    /// The UI layer observes this to update the status indicator.
    var onStreamingUpdate: ((StreamingUpdate) -> Void)?

    /// Called when the session reaches a terminal state.
    var onComplete: ((SessionResult) -> Void)?

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

    // MARK: - Starting

    // h[impl capture.start]
    // h[impl ime.activate]
    func start(language: String?) async {
        // Copy pre-buffer from audio engine and begin accumulating
        capture = .buffering
        audioEngine.startCapture(for: self)

        // Activate IME
        inputClient.activate()
        ime = .active

        // Create ASR session
        asr = .streaming
        // TODO: actual FFI call to create session with language

        // Start the streaming loop
        streamingTask = Task { [weak self] in
            await self?.streamingLoop()
        }
    }

    // MARK: - Ending

    /// Immediate teardown. No finalization, no history, no trace.
    func abort() async {
        guard !isTerminal else { return }

        // Discard audio — no drain
        // h[impl capture.abort-discard]
        audioEngine.cancelCapture(for: self)
        capture = .discarded

        // Drop ASR session
        asr = .done

        // h[impl ime.abort-teardown]
        inputClient.deactivate()
        ime = .tornDown

        streamingTask?.cancel()
        onComplete?(.aborted(id: id))
    }

    /// Finalize in background, create history entry, but don't insert text.
    func cancel() async {
        guard !isTerminal else { return }

        // h[impl capture.drain]
        capture = .draining
        let samples = await drainAudio()
        capture = .delivered

        // h[impl ime.clear-on-cancel]
        inputClient.clearMarkedText()
        inputClient.deactivate()
        ime = .cleared

        // Finalize ASR in background
        // h[impl asr.finalize-background]
        Task.detached { [self] in
            await self.finalize(samples: samples, insert: false, submit: false)
        }
    }

    /// Finalize and insert text. If submit, simulate Return after insertion.
    func commit(submit: Bool) async {
        guard !isTerminal else { return }

        // h[impl capture.drain]
        capture = .draining
        let samples = await drainAudio()
        capture = .delivered

        // Finalize ASR in background, then commit IME
        // h[impl asr.finalize-background]
        Task.detached { [self] in
            await self.finalize(samples: samples, insert: true, submit: submit)
        }
    }

    // MARK: - Internal

    private var isTerminal: Bool {
        switch (capture, asr, ime) {
        case (.discarded, _, _): return true
        case (_, .done, .committed): return true
        case (_, .done, .cleared): return true
        case (_, .done, .tornDown): return true
        default: return false
        }
    }

    // h[impl asr.streaming]
    private func streamingLoop() async {
        while !Task.isCancelled && capture == .buffering {
            let allSamples = audioEngine.peekCapture(for: self)
            let newCount = allSamples.count
            guard newCount > processedSampleCount + 800 else {
                try? await Task.sleep(for: .milliseconds(30))
                continue
            }

            let chunk = Array(allSamples[processedSampleCount...])
            processedSampleCount = newCount

            // TODO: feed chunk to ASR session, get update
            // let update = transcriptionService.feed(session: asrSession, samples: chunk)
            // if let update {
            //     partialTranscript = update.text
            //     // h[impl ime.marked-text]
            //     inputClient.setMarkedText(update.text)
            //     onStreamingUpdate?(StreamingUpdate(text: update.text, ...))
            // }
        }
    }

    // h[impl capture.drain-delivers]
    // h[impl asr.tail-audio]
    private func drainAudio() async -> [Float] {
        streamingTask?.cancel()
        await streamingTask?.value
        let samples = audioEngine.stopCapture(for: self)
        return samples
    }

    // h[impl asr.finalize]
    // h[impl coord.drain-before-finalize]
    private func finalize(samples: [Float], insert: Bool, submit: Bool) async {
        asr = .finalizing

        // Feed remaining samples to ASR
        let remaining = samples.count > processedSampleCount
            ? Array(samples[processedSampleCount...])
            : []
        // TODO: transcriptionService.feedFinalizing(session: asrSession, samples: remaining)
        // TODO: let text = transcriptionService.finish(session: asrSession)
        let text = partialTranscript // placeholder

        asr = .done
        finalText = text

        if insert && !text.isEmpty {
            // h[impl ime.commit]
            inputClient.commitText(text)
            // h[impl ime.deactivate]
            inputClient.deactivate()
            ime = .committed

            if submit {
                // h[impl ime.submit]
                try? await Task.sleep(for: .milliseconds(50))
                inputClient.simulateReturn()
            }
        } else if !insert {
            // Cancel path — IME already cleared above
        }

        if insert {
            onComplete?(.committed(id: id, text: finalText, submitted: submit))
        } else {
            onComplete?(.cancelled(id: id, text: finalText))
        }
    }
}

// MARK: - Layer States

extension Session {
    enum CaptureState: Sendable {
        case buffering
        case draining
        case delivered
        case discarded
    }

    enum ASRState: Sendable {
        case idle
        case streaming
        case finalizing
        case done
    }

    enum IMEState: Sendable {
        case inactive
        case active
        case parked
        case committed
        case cleared
        case tornDown
    }
}

// MARK: - Supporting types

struct StreamingUpdate: Sendable {
    let text: String
    let committedUTF16Count: Int
    let detectedLanguage: String?
}

enum SessionResult: Sendable {
    case aborted(id: UUID)
    case cancelled(id: UUID, text: String)
    case committed(id: UUID, text: String, submitted: Bool)
}
