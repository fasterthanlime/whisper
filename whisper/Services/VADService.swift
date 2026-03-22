import AVFoundation
import Foundation
import MLX
import MLXAudioVAD
import os

/// Voice Activity Detection service using Sortformer.
/// Detects speech boundaries in real-time audio and emits segments for transcription.
actor VADService {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "whisper",
        category: "VADService"
    )

    // Model and streaming state are owned by this actor.
    // We use @unchecked Sendable wrapper to pass to feed() which is async but not actor-isolated.
    private var model: SortformerModel?
    private var streamingState: StreamingState?

    private let sampleRate: Int = 16000

    // Accumulate audio samples for the current speech segment
    private var currentSegmentAudio: [Float] = []

    // Track speech state
    private var isSpeaking = false
    private var silenceFrameCount = 0

    // Configuration
    private let silenceThresholdFrames = 2  // ~200ms silence to trigger segment
    private let minSegmentSamples = 4800     // 300ms minimum segment

    // Wrapper for non-Sendable types
    private struct UncheckedBox<T>: @unchecked Sendable {
        let value: T
    }

    enum VADEvent: Sendable {
        case speechStarted
        case speechSegment(audio: [Float])
        case speechEnded
    }

    /// Load the Sortformer VAD model.
    func loadModel() async throws {
        guard model == nil else { return }

        Self.logger.info("Loading Sortformer VAD model...")
        let loadedModel = try await SortformerModel.fromPretrained(
            "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
        )

        self.model = loadedModel
        self.streamingState = loadedModel.initStreamingState()

        Self.logger.info("Sortformer VAD model loaded")
    }

    /// Check if model is loaded.
    var isLoaded: Bool { model != nil }

    /// Reset VAD state for a new recording session.
    func reset() {
        if let model {
            streamingState = model.initStreamingState()
        }
        currentSegmentAudio.removeAll()
        isSpeaking = false
        silenceFrameCount = 0
    }

    /// Process an audio chunk and detect speech boundaries.
    /// - Parameters:
    ///   - samples: Float32 audio samples at 16kHz
    /// - Returns: VAD events (speech started, segment ready, speech ended)
    func processAudio(_ samples: [Float]) async throws -> [VADEvent] {
        guard let model, let currentState = streamingState else {
            throw VADError.modelNotLoaded
        }

        // Accumulate audio
        currentSegmentAudio.append(contentsOf: samples)

        // Wrap for Sendable crossing
        let modelBox = UncheckedBox(value: model)
        let stateBox = UncheckedBox(value: currentState)
        let chunk = MLXArray(samples)

        // Feed to VAD model
        let (result, newState) = try await modelBox.value.feed(
            chunk: chunk,
            state: stateBox.value,
            sampleRate: sampleRate,
            threshold: 0.5
        )

        // Update state
        streamingState = newState

        var events: [VADEvent] = []
        let hasActiveSpeaker = !result.segments.isEmpty

        if hasActiveSpeaker {
            silenceFrameCount = 0

            if !isSpeaking {
                isSpeaking = true
                events.append(.speechStarted)
            }
        } else {
            if isSpeaking {
                silenceFrameCount += 1

                if silenceFrameCount >= silenceThresholdFrames {
                    if currentSegmentAudio.count >= minSegmentSamples {
                        events.append(.speechSegment(audio: currentSegmentAudio))
                    }
                    currentSegmentAudio.removeAll()
                    isSpeaking = false
                    silenceFrameCount = 0
                    events.append(.speechEnded)
                }
            }
        }

        return events
    }

    /// Flush any remaining audio as a final segment.
    func flush() -> [Float]? {
        guard currentSegmentAudio.count >= minSegmentSamples else {
            let remaining = currentSegmentAudio
            currentSegmentAudio.removeAll()
            return remaining.isEmpty ? nil : remaining
        }
        let segment = currentSegmentAudio
        currentSegmentAudio.removeAll()
        return segment
    }

    /// Get all accumulated audio (for final transcription on key release).
    func getAllAudio() -> [Float] {
        return currentSegmentAudio
    }
}

enum VADError: LocalizedError {
    case modelNotLoaded

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "VAD model not loaded"
        }
    }
}
