import AVFoundation
import Accelerate
import Foundation

/// Records microphone audio and returns raw float32 samples at 16kHz.
/// Supports "warm" mode where the engine runs continuously with a pre-buffer.
final class AudioRecorder: @unchecked Sendable {
    static let defaultMaximumDuration: TimeInterval = 90
    static let defaultPreBufferDuration: TimeInterval = 0.2  // 200ms pre-buffer
    static let spectrumBandCount = 6

    private var engine: AVAudioEngine?
    private var nativeSampleRate: Double = 0
    private let lock = NSLock()

    // Circular pre-buffer for warm mode
    private var preBuffer: [Float] = []
    private var preBufferWriteIndex = 0
    private var preBufferCapacity = 0

    // Captured samples during recording
    private var capturedSamples: [Float] = []
    private var isCapturing = false
    private var isWarm = false
    /// True while waiting to stop after a short post-keyup VAD tail window.
    private var stopCapturePending = false
    private var stopCaptureSignal: DispatchSemaphore?
    private var stopCaptureSilenceSamples = 0
    private var stopCaptureSamplesUntilTimeout = 0
    private let stopCaptureMaxWaitSeconds: TimeInterval = 0.5
    private let stopCaptureRequiredSilenceSeconds: TimeInterval = 0.12
    private let stopCaptureSilenceRmsThreshold: Float = 0.008
    private let stopCaptureBoundaryTimeoutSeconds: TimeInterval = 0.65

    private var onLevel: (@Sendable (Float) -> Void)?
    private var onSpectrum: (@Sendable ([Float]) -> Void)?

    // FFT setup
    private var smoothedBands = [Float](repeating: 0, count: spectrumBandCount)
    private let fftSize = 512
    private var fftSetup: vDSP_DFT_Setup?
    private var fftWindow: [Float] = []
    private var fftRealBuffer: [Float] = []
    private var fftImagBuffer: [Float] = []

    /// Target sample rate for the STT model.
    private let targetSampleRate: Double = 16000
    private let maximumDuration: TimeInterval
    private let preBufferDuration: TimeInterval

    init(
        maximumDuration: TimeInterval = AudioRecorder.defaultMaximumDuration,
        preBufferDuration: TimeInterval = AudioRecorder.defaultPreBufferDuration
    ) {
        self.maximumDuration = max(1, maximumDuration)
        self.preBufferDuration = preBufferDuration

        // Setup FFT
        fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftSize), .FORWARD)
        fftWindow = [Float](repeating: 0, count: fftSize)
        vDSP_hann_window(&fftWindow, vDSP_Length(fftSize), Int32(vDSP_HANN_NORM))
        fftRealBuffer = [Float](repeating: 0, count: fftSize)
        fftImagBuffer = [Float](repeating: 0, count: fftSize)
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Start the audio engine in "warm" mode - continuously recording to a circular buffer.
    /// Call `startCapture()` to begin actual capture, `stopCapture()` to get audio.
    func warmUp(
        onLevel: (@Sendable (Float) -> Void)? = nil,
        onSpectrum: (@Sendable ([Float]) -> Void)? = nil
    ) throws {
        lock.lock()
        guard !isWarm else {
            lock.unlock()
            return
        }
        lock.unlock()

        self.onLevel = onLevel
        self.onSpectrum = onSpectrum

        let engine = AVAudioEngine()
        self.engine = engine

        let inputNode = engine.inputNode
        let nativeFormat = inputNode.outputFormat(forBus: 0)

        guard nativeFormat.sampleRate > 0 else {
            throw AudioRecorderError.noMicrophone
        }

        nativeSampleRate = nativeFormat.sampleRate

        lock.lock()
        // Initialize circular pre-buffer
        preBufferCapacity = Int(nativeSampleRate * preBufferDuration)
        preBuffer = [Float](repeating: 0, count: preBufferCapacity)
        preBufferWriteIndex = 0
        capturedSamples.removeAll()
        isCapturing = false
        isWarm = true
        lock.unlock()

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nativeFormat) { [weak self] buffer, _ in
            self?.handleAudioBuffer(buffer)
        }

        try engine.start()
    }

    /// Stop warm mode and release audio resources.
    func coolDown() {
        lock.lock()
        isWarm = false
        isCapturing = false
        stopCapturePending = false
        stopCaptureSignal = nil
        stopCaptureSilenceSamples = 0
        stopCaptureSamplesUntilTimeout = 0
        lock.unlock()

        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        onLevel = nil

        lock.lock()
        preBuffer.removeAll()
        capturedSamples.removeAll()
        lock.unlock()
    }

    /// Begin capturing audio (including pre-buffer). Only works if warm.
    func startCapture() {
        lock.lock()
        defer { lock.unlock() }

        guard isWarm else { return }

        // Copy pre-buffer contents in correct order
        capturedSamples.removeAll()
        if preBufferCapacity > 0 {
            // The pre-buffer is circular, so we need to read from writeIndex to end, then start to writeIndex
            let samplesInBuffer = min(preBufferWriteIndex, preBufferCapacity)
            if preBufferWriteIndex >= preBufferCapacity {
                // Buffer has wrapped
                let startIndex = preBufferWriteIndex % preBufferCapacity
                capturedSamples.append(contentsOf: preBuffer[startIndex...])
                capturedSamples.append(contentsOf: preBuffer[..<startIndex])
            } else {
                // Buffer hasn't wrapped yet
                capturedSamples.append(contentsOf: preBuffer[..<preBufferWriteIndex])
            }
        }

        stopCapturePending = false
        stopCaptureSignal = nil
        stopCaptureSilenceSamples = 0
        stopCaptureSamplesUntilTimeout = 0
        isCapturing = true
    }

    /// Peek at current captured audio without stopping capture.
    /// Returns samples resampled to 16kHz mono float32.
    func peekCapture() -> [Float] {
        lock.lock()
        let captured = capturedSamples
        let capturedRate = nativeSampleRate
        lock.unlock()

        guard !captured.isEmpty else { return [] }

        if capturedRate == targetSampleRate {
            return captured
        }

        return resample(captured, from: capturedRate, to: targetSampleRate)
    }

    /// Stop capturing and return audio samples resampled to 16kHz mono float32.
    func stopCapture() -> [Float] {
        var waitSignal: DispatchSemaphore?

        lock.lock()
        if isCapturing {
            let signal = DispatchSemaphore(value: 0)
            stopCapturePending = true
            stopCaptureSignal = signal
            stopCaptureSilenceSamples = 0
            stopCaptureSamplesUntilTimeout = max(
                1,
                Int((nativeSampleRate * stopCaptureMaxWaitSeconds).rounded())
            )
            waitSignal = signal
        } else {
            stopCapturePending = false
            stopCaptureSignal = nil
            stopCaptureSilenceSamples = 0
            stopCaptureSamplesUntilTimeout = 0
        }
        lock.unlock()

        // Wait for VAD tail-stop (silence or timeout) so we avoid cutting
        // speech endings when key-up happens near the final words.
        if let waitSignal {
            _ = waitSignal.wait(timeout: .now() + stopCaptureBoundaryTimeoutSeconds)
        }

        lock.lock()
        isCapturing = false
        stopCapturePending = false
        stopCaptureSignal = nil
        stopCaptureSilenceSamples = 0
        stopCaptureSamplesUntilTimeout = 0
        let captured = capturedSamples
        let capturedRate = nativeSampleRate
        capturedSamples.removeAll()
        lock.unlock()

        guard !captured.isEmpty else { return [] }

        // Resample to 16kHz if needed
        if capturedRate == targetSampleRate {
            return captured
        }

        return resample(captured, from: capturedRate, to: targetSampleRate)
    }

    /// Legacy API: Start recording from cold (initializes engine).
    func start(onLevel: (@Sendable (Float) -> Void)? = nil) throws {
        try warmUp(onLevel: onLevel)
        startCapture()
    }

    /// Legacy API: Stop recording and return samples.
    func stop() -> [Float] {
        let samples = stopCapture()
        coolDown()
        return samples
    }

    /// Check if currently in warm mode.
    var isWarmedUp: Bool {
        lock.lock()
        defer { lock.unlock() }
        return isWarm
    }

    private func handleAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return }

        let bufferPointer = UnsafeBufferPointer(start: channelData[0], count: count)

            // Compute RMS level
            let rms = computeRMS(bufferPointer)
            if let onLevel = self.onLevel {
                let normalized = min(rms * 5.0, 1.0)
                onLevel(normalized)
            }

        // Compute FFT spectrum
        if let onSpectrum = self.onSpectrum, let setup = fftSetup, count >= fftSize {
            let bands = computeSpectrum(bufferPointer, setup: setup)
            onSpectrum(bands)
        }

        var stopSignalToFire: DispatchSemaphore?
        lock.lock()

        if isCapturing || stopCapturePending {
            // Actively capturing - append to captured samples
            let maxNativeSamples = Int(nativeSampleRate * maximumDuration)
            let remaining = maxNativeSamples - capturedSamples.count

            if remaining > 0 {
                if count <= remaining {
                    capturedSamples.append(contentsOf: bufferPointer)
                } else {
                    capturedSamples.append(contentsOf: bufferPointer.prefix(remaining))
                    isCapturing = false
                }
            } else {
                isCapturing = false
            }

            if stopCapturePending {
                if rms < stopCaptureSilenceRmsThreshold {
                    stopCaptureSilenceSamples += count
                } else {
                    stopCaptureSilenceSamples = 0
                }
                stopCaptureSamplesUntilTimeout = max(0, stopCaptureSamplesUntilTimeout - count)

                let requiredSilenceSamples = max(
                    1,
                    Int((nativeSampleRate * stopCaptureRequiredSilenceSeconds).rounded())
                )
                let reachedSilence = stopCaptureSilenceSamples >= requiredSilenceSamples
                let reachedTimeout = stopCaptureSamplesUntilTimeout == 0

                if reachedSilence || reachedTimeout {
                    stopCapturePending = false
                    isCapturing = false
                    stopSignalToFire = stopCaptureSignal
                    stopCaptureSignal = nil
                    stopCaptureSilenceSamples = 0
                    stopCaptureSamplesUntilTimeout = 0
                }
            }
        } else if isWarm && preBufferCapacity > 0 {
            // Warm but not capturing - fill circular pre-buffer
            for sample in bufferPointer {
                preBuffer[preBufferWriteIndex % preBufferCapacity] = sample
                preBufferWriteIndex += 1
            }
        }
        lock.unlock()

        // Signal outside the lock to avoid waking waiter into lock contention.
        stopSignalToFire?.signal()
    }

    private func computeSpectrum(_ samples: UnsafeBufferPointer<Float>, setup: vDSP_DFT_Setup) -> [Float] {
        // Copy and window the samples
        var windowed = [Float](repeating: 0, count: fftSize)
        for i in 0..<min(samples.count, fftSize) {
            windowed[i] = samples[i] * fftWindow[i]
        }

        // Perform FFT
        var realOut = [Float](repeating: 0, count: fftSize)
        var imagOut = [Float](repeating: 0, count: fftSize)
        var imagIn = [Float](repeating: 0, count: fftSize)

        vDSP_DFT_Execute(setup, windowed, imagIn, &realOut, &imagOut)

        // Compute magnitudes for first half (positive frequencies)
        let halfSize = fftSize / 2
        var magnitudes = [Float](repeating: 0, count: halfSize)
        for i in 0..<halfSize {
            magnitudes[i] = sqrtf(realOut[i] * realOut[i] + imagOut[i] * imagOut[i])
        }

        // Group into bands focused on voice frequencies (100Hz - 4kHz)
        let bandCount = Self.spectrumBandCount
        var bands = [Float](repeating: 0, count: bandCount)

        // Calculate bin indices for voice frequency range
        // Each bin represents nativeSampleRate/fftSize Hz
        let binHz = nativeSampleRate / Double(fftSize)
        let voiceMinHz = 200.0
        let voiceMaxHz = 4000.0
        let voiceMinBin = max(1, Int(voiceMinHz / binHz))
        let voiceMaxBin = min(halfSize - 1, Int(voiceMaxHz / binHz))
        let voiceBinRange = voiceMaxBin - voiceMinBin

        // Linear spacing within voice range for even distribution
        for band in 0..<bandCount {
            let lowBin = voiceMinBin + (band * voiceBinRange) / bandCount
            let highBin = voiceMinBin + ((band + 1) * voiceBinRange) / bandCount

            var sum: Float = 0
            let binCount = max(1, highBin - lowBin)
            for bin in lowBin..<highBin {
                sum += magnitudes[bin]
            }
            bands[band] = sum / Float(binCount)
        }

        // Convert to dB and normalize to 0..1 range.
        // This handles the wide dynamic range of audio naturally.
        for i in 0..<bandCount {
            let db = 20.0 * log10f(max(bands[i], 1e-10))
            bands[i] = max(0, min(1, (db + 50) / 40)) // map -50dB...-10dB → 0...1
        }

        // Temporal smoothing: fast attack, slow decay
        let attackRate: Float = 0.7
        let decayRate: Float = 0.12
        for i in 0..<bandCount {
            if bands[i] > smoothedBands[i] {
                smoothedBands[i] += (bands[i] - smoothedBands[i]) * attackRate
            } else {
                smoothedBands[i] += (bands[i] - smoothedBands[i]) * decayRate
            }
        }

        return smoothedBands
    }

    private func computeRMS(_ samples: UnsafeBufferPointer<Float>) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sumOfSquares: Float = 0
        for sample in samples {
            sumOfSquares += sample * sample
        }
        return sqrtf(sumOfSquares / Float(samples.count))
    }

    private func resample(_ samples: [Float], from srcRate: Double, to dstRate: Double) -> [Float] {
        guard let srcFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: srcRate,
            channels: 1,
            interleaved: false
        ),
        let dstFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: dstRate,
            channels: 1,
            interleaved: false
        ),
        let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
            return samples
        }

        let srcFrameCount = AVAudioFrameCount(samples.count)
        guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcFrameCount) else {
            return samples
        }
        srcBuffer.frameLength = srcFrameCount
        if let channelData = srcBuffer.floatChannelData {
            samples.withUnsafeBufferPointer { ptr in
                channelData[0].update(from: ptr.baseAddress!, count: samples.count)
            }
        }

        let ratio = dstRate / srcRate
        let dstFrameCount = AVAudioFrameCount(Double(srcFrameCount) * ratio) + 1
        guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: dstFrameCount) else {
            return samples
        }

        var consumed = false
        var convError: NSError?
        let status = converter.convert(to: dstBuffer, error: &convError) { _, outStatus in
            if consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return srcBuffer
        }

        if status == .error {
            return samples
        }

        guard let channelData = dstBuffer.floatChannelData else { return samples }
        let count = Int(dstBuffer.frameLength)
        let result = Array(UnsafeBufferPointer(start: channelData[0], count: count))
        return result
    }

    /// Check and request microphone permission.
    static func requestPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        default:
            return false
        }
    }
}

enum AudioRecorderError: LocalizedError {
    case noMicrophone

    var errorDescription: String? {
        switch self {
        case .noMicrophone:
            return "No microphone available"
        }
    }
}
