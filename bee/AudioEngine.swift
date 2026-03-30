import AVFoundation
import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "AudioEngine")

/// Shared audio infrastructure. Runs continuously when warm, capturing audio
/// into a circular pre-buffer that sessions tap into.
///
/// Stores samples at the device's native sample rate. Sessions resample
/// per-chunk before feeding to the ASR.
final class AudioEngine: @unchecked Sendable {
    enum State {
        case cold
        case warm
    }

    private(set) var state: State = .cold
    private let lock = NSLock()

    private var engine: AVAudioEngine?
    private(set) var nativeSampleRate: Double = 0
    static let targetSampleRate: Double = 16_000

    // Circular pre-buffer (~200ms at native rate)
    private let preBufferDuration: TimeInterval = 0.2
    private var preBuffer: [Float] = []
    private var preBufferWriteIndex = 0
    private var preBufferCapacity = 0

    // Per-session capture handles
    private var activeCaptures: [UUID: CaptureHandle] = [:]

    // VAD parameters
    private let vadRequiredSilenceSeconds: TimeInterval = 0.15
    private let vadSilenceRmsThreshold: Float = 0.008
    private let vadDrainTimeoutSeconds: TimeInterval = 0.5

    // Device management
    var selectedDeviceUID: String?
    var deviceWarmPolicy: [String: Bool] = [:]

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

        lock.lock()
        self.nativeSampleRate = nativeRate
        preBufferCapacity = Int(nativeRate * preBufferDuration)
        preBuffer = [Float](repeating: 0, count: preBufferCapacity)
        preBufferWriteIndex = 0
        state = .warm
        lock.unlock()

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nativeFormat) {
            [weak self] buffer, _ in
            self?.handleAudioBuffer(buffer)
        }

        try engine.start()
        self.engine = engine
        logger.info("Audio engine warm: native rate = \(nativeRate) Hz")
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
    }

    var isWarm: Bool {
        lock.withLock { state == .warm }
    }

    // MARK: - Capture API

    /// Begin capturing audio. Copies the pre-buffer.
    func startCapture(for sessionID: UUID) {
        lock.lock()
        defer { lock.unlock() }

        var handle = CaptureHandle()

        if preBufferCapacity > 0 {
            if preBufferWriteIndex >= preBufferCapacity {
                let startIndex = preBufferWriteIndex % preBufferCapacity
                handle.samples.append(contentsOf: preBuffer[startIndex...])
                handle.samples.append(contentsOf: preBuffer[..<startIndex])
            } else {
                handle.samples.append(contentsOf: preBuffer[..<preBufferWriteIndex])
            }
        }

        handle.phase = .capturing
        activeCaptures[sessionID] = handle
    }

    /// Peek at current captured audio at native sample rate.
    func peekCapture(for sessionID: UUID) -> [Float] {
        lock.lock()
        let samples = activeCaptures[sessionID]?.samples ?? []
        lock.unlock()
        return samples
    }

    /// Signal the capture to begin draining (VAD tail monitoring).
    /// The capture continues accumulating audio during the drain.
    /// Call `isDrained(for:)` to check when drain is complete.
    func beginDrain(for sessionID: UUID) {
        lock.lock()
        defer { lock.unlock() }

        guard var handle = activeCaptures[sessionID] else { return }
        guard handle.phase == .capturing else { return }

        handle.phase = .draining
        handle.drainSilenceSamples = 0
        handle.drainSamplesUntilTimeout = max(1, Int((nativeSampleRate * vadDrainTimeoutSeconds).rounded()))
        activeCaptures[sessionID] = handle
        logger.info("beginDrain for \(sessionID.uuidString.prefix(8))")
    }

    /// Check if the drain is complete.
    func isDrained(for sessionID: UUID) -> Bool {
        lock.lock()
        let phase = activeCaptures[sessionID]?.phase
        lock.unlock()
        return phase == .drained
    }

    /// Collect all captured samples and remove the handle.
    /// Call after drain is complete (or for abort).
    func collectSamples(for sessionID: UUID) -> [Float] {
        lock.lock()
        let samples = activeCaptures.removeValue(forKey: sessionID)?.samples ?? []
        lock.unlock()
        return samples
    }

    /// Cancel capture, discard audio immediately.
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

        let bufferPointer = UnsafeBufferPointer(start: channelData[0], count: count)
        let rms = computeRMS(bufferPointer)

        lock.lock()

        for (id, var handle) in activeCaptures {
            switch handle.phase {
            case .capturing:
                handle.samples.append(contentsOf: bufferPointer)
                handle.lastRms = rms
                activeCaptures[id] = handle

            case .draining:
                // Still accumulating audio during drain
                handle.samples.append(contentsOf: bufferPointer)
                handle.lastRms = rms

                // VAD: check for silence
                if rms < vadSilenceRmsThreshold {
                    handle.drainSilenceSamples += count
                } else {
                    handle.drainSilenceSamples = 0
                }
                handle.drainSamplesUntilTimeout = max(0, handle.drainSamplesUntilTimeout - count)

                let requiredSilence = max(1, Int((nativeSampleRate * vadRequiredSilenceSeconds).rounded()))
                let reachedSilence = handle.drainSilenceSamples >= requiredSilence
                let reachedTimeout = handle.drainSamplesUntilTimeout == 0

                if reachedSilence || reachedTimeout {
                    handle.phase = .drained
                    logger.info("Drain complete: \(reachedSilence ? "silence" : "timeout")")
                }

                activeCaptures[id] = handle

            case .drained:
                // No longer accumulating
                break
            }
        }

        // Fill circular pre-buffer
        if state == .warm && preBufferCapacity > 0 {
            for sample in bufferPointer {
                preBuffer[preBufferWriteIndex % preBufferCapacity] = sample
                preBufferWriteIndex += 1
            }
        }

        lock.unlock()
    }

    private func computeRMS(_ samples: UnsafeBufferPointer<Float>) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sum: Float = 0
        for s in samples { sum += s * s }
        return sqrtf(sum / Float(samples.count))
    }

    // MARK: - Resampling

    static func resample(_ samples: [Float], from srcRate: Double) -> [Float] {
        let dstRate = targetSampleRate
        if srcRate == dstRate { return samples }
        guard !samples.isEmpty else { return [] }

        guard let srcFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: srcRate, channels: 1, interleaved: false
        ), let dstFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: dstRate, channels: 1, interleaved: false
        ), let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
            return samples
        }

        let srcCount = AVAudioFrameCount(samples.count)
        guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcCount) else {
            return samples
        }
        srcBuffer.frameLength = srcCount
        if let cd = srcBuffer.floatChannelData {
            samples.withUnsafeBufferPointer { cd[0].update(from: $0.baseAddress!, count: samples.count) }
        }

        let dstCount = AVAudioFrameCount(Double(srcCount) * dstRate / srcRate) + 1
        guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: dstCount) else {
            return samples
        }

        var consumed = false
        _ = converter.convert(to: dstBuffer, error: nil) { _, outStatus in
            if consumed { outStatus.pointee = .endOfStream; return nil }
            consumed = true
            outStatus.pointee = .haveData
            return srcBuffer
        }

        guard let cd = dstBuffer.floatChannelData else { return samples }
        return Array(UnsafeBufferPointer(start: cd[0], count: Int(dstBuffer.frameLength)))
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

    // MARK: - Internal

    private struct CaptureHandle {
        enum Phase { case capturing, draining, drained }

        var samples: [Float] = []
        var phase: Phase = .capturing
        var lastRms: Float = 0
        var drainSilenceSamples = 0
        var drainSamplesUntilTimeout = 0
    }
}

enum AudioEngineError: LocalizedError {
    case noMicrophone
    var errorDescription: String? { "No microphone available" }
}
