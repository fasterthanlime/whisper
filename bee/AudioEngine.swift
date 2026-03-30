import AVFoundation
import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "AudioEngine")

/// Shared audio infrastructure. Runs continuously when warm, capturing audio
/// into a circular pre-buffer that sessions tap into.
///
/// Audio is resampled to 16kHz mono at capture time (in the audio callback).
/// All buffers (pre-buffer, session captures) store 16kHz samples. This avoids
/// resampling mismatches between peekCapture and stopCapture.
final class AudioEngine: @unchecked Sendable {
    enum State {
        case cold
        case warm
    }

    private(set) var state: State = .cold
    private let lock = NSLock()

    private var engine: AVAudioEngine?
    private var nativeSampleRate: Double = 0
    private let targetSampleRate: Double = 16_000

    // Circular pre-buffer for warm mode (~200ms at 16kHz = 3200 samples)
    private let preBufferDuration: TimeInterval = 0.2
    private var preBuffer: [Float] = []
    private var preBufferWriteIndex = 0
    private var preBufferCapacity = 0

    // Per-session capture state
    private var activeCaptures: [UUID: CaptureHandle] = [:]

    // Device management
    var selectedDeviceUID: String?
    var deviceWarmPolicy: [String: Bool] = [:]
    var onDeviceListChanged: (() -> Void)?

    // MARK: - Engine Lifecycle

    func warmUp() throws {
        lock.lock()
        guard state == .cold else { lock.unlock(); return }
        lock.unlock()

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let nativeFormat = inputNode.outputFormat(forBus: 0)

        guard nativeFormat.sampleRate > 0 else {
            throw AudioEngineError.noMicrophone
        }

        let nativeRate = nativeFormat.sampleRate

        // Install tap with 16kHz mono format — AVAudioEngine resamples for us
        let tapFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        )!

        lock.lock()
        self.nativeSampleRate = nativeRate
        // Pre-buffer at 16kHz
        preBufferCapacity = Int(targetSampleRate * preBufferDuration)
        preBuffer = [Float](repeating: 0, count: preBufferCapacity)
        preBufferWriteIndex = 0
        state = .warm
        lock.unlock()

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: tapFormat) {
            [weak self] buffer, _ in
            self?.handleAudioBuffer(buffer)
        }

        try engine.start()
        self.engine = engine
        logger.info("Audio engine warm: native rate = \(nativeRate) Hz, resampling to 16kHz")
    }

    func coolDown() {
        lock.lock()
        state = .cold
        preBuffer.removeAll()
        preBufferCapacity = 0
        preBufferWriteIndex = 0
        lock.unlock()

        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        logger.info("Audio engine cooled down")
    }

    var isWarm: Bool {
        lock.withLock { state == .warm }
    }

    // MARK: - Capture API

    /// Begin capturing audio for a session. Copies the pre-buffer.
    func startCapture(for sessionID: UUID) {
        lock.lock()
        defer { lock.unlock() }

        var handle = CaptureHandle()

        // Copy pre-buffer (circular read) — already at 16kHz
        if preBufferCapacity > 0 {
            if preBufferWriteIndex >= preBufferCapacity {
                let startIndex = preBufferWriteIndex % preBufferCapacity
                handle.samples.append(contentsOf: preBuffer[startIndex...])
                handle.samples.append(contentsOf: preBuffer[..<startIndex])
            } else {
                handle.samples.append(contentsOf: preBuffer[..<preBufferWriteIndex])
            }
        }

        handle.isCapturing = true
        activeCaptures[sessionID] = handle
    }

    /// Peek at current captured audio. Already at 16kHz.
    func peekCapture(for sessionID: UUID) -> [Float] {
        lock.lock()
        let samples = activeCaptures[sessionID]?.samples ?? []
        lock.unlock()
        return samples
    }

    /// Stop capturing. Returns all samples at 16kHz. No VAD drain for now.
    func stopCapture(for sessionID: UUID) -> [Float] {
        lock.lock()
        let samples = activeCaptures.removeValue(forKey: sessionID)?.samples ?? []
        lock.unlock()

        logger.info("stopCapture: \(samples.count) samples at 16kHz")
        return samples
    }

    /// Cancel capture, discard audio. Non-blocking.
    func cancelCapture(for sessionID: UUID) {
        lock.lock()
        activeCaptures.removeValue(forKey: sessionID)
        lock.unlock()
    }

    // MARK: - Audio Callback

    private func handleAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

        // Buffer is already at 16kHz mono (AVAudioEngine resamples via tap format)
        let samples = UnsafeBufferPointer(start: channelData[0], count: count)

        lock.lock()

        // Append to all active captures
        for (id, var handle) in activeCaptures {
            if handle.isCapturing {
                handle.samples.append(contentsOf: samples)
                activeCaptures[id] = handle
            }
        }

        // Fill circular pre-buffer (always, when warm)
        if state == .warm && preBufferCapacity > 0 {
            for sample in samples {
                preBuffer[preBufferWriteIndex % preBufferCapacity] = sample
                preBufferWriteIndex += 1
            }
        }

        lock.unlock()
    }

    // MARK: - Mic Permission

    static func requestPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized: return true
        case .notDetermined: return await AVCaptureDevice.requestAccess(for: .audio)
        default: return false
        }
    }

    func selectDevice(uid: String) {
        selectedDeviceUID = uid
    }

    // MARK: - Per-capture state

    private struct CaptureHandle {
        var samples: [Float] = []
        var isCapturing = false
    }
}

enum AudioEngineError: LocalizedError {
    case noMicrophone
    var errorDescription: String? { "No microphone available" }
}
