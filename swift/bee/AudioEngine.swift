import AVFoundation
import AudioToolbox
import CoreAudio
import Foundation
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "AudioEngine")

/// Shared audio infrastructure. Uses a raw AUHAL for capture so we can
/// select any input device, not just the system default.
///
/// Resamples to 16kHz and yields into per-session Channel 0 pipelines.
final class AudioEngine: @unchecked Sendable {
    enum State {
        case cold
        case warm
    }

    private(set) var state: State = .cold
    private let lock = NSLock()

    fileprivate var auhalUnit: AudioUnit?
    private(set) var nativeSampleRate: Double = 0
    static let targetSampleRate: Double = 16_000

    // Circular pre-buffer (~200ms at native rate)
    private let preBufferDuration: TimeInterval = 0.2
    private var preBuffer: [Float] = []
    private var preBufferWriteIndex = 0
    private var preBufferCapacity = 0

    // Render buffer for AUHAL callback
    private var renderBufferList: UnsafeMutableAudioBufferListPointer?
    private var renderBufferData: UnsafeMutablePointer<Float>?
    private var renderBufferCapacity: UInt32 = 4096

    // Cached resampler (created once per warmUp, reused across callbacks)
    private var resampler: AVAudioConverter?
    private var resamplerSrcFormat: AVAudioFormat?
    private var resamplerDstFormat: AVAudioFormat?
    // Accumulation buffer for resampling (collect native samples, resample in chunks)
    private var resampleAccumulator: [Float] = []
    private let resampleChunkSize = 4096  // resample in chunks for quality

    // Per-session raw audio pipelines (Channel 0)
    private var activePipelines: [UUID: RawAudioPipeline] = [:]

    // Device management
    var selectedDeviceUID: String?
    var deviceWarmPolicy: [String: Bool] = [:]

    // Audio level & stats (updated from audio callback)
    private(set) var currentLevel: Float = 0
    private(set) var totalBuffersReceived: UInt64 = 0
    private(set) var totalSamplesReceived: UInt64 = 0
    private(set) var activePipelineCount: Int = 0
    private(set) var channelCount: UInt32 = 0
    private(set) var currentRMS: Float = 0
    private(set) var peakLevel: Float = 0

    // MARK: - Engine Lifecycle

    func warmUp() throws {
        lock.lock()
        guard state == .cold else { lock.unlock(); return }
        lock.unlock()

        // 1. Create AUHAL
        var componentDesc = AudioComponentDescription(
            componentType: kAudioUnitType_Output,
            componentSubType: kAudioUnitSubType_HALOutput,
            componentManufacturer: kAudioUnitManufacturer_Apple,
            componentFlags: 0,
            componentFlagsMask: 0
        )
        guard let component = AudioComponentFindNext(nil, &componentDesc) else {
            beeLog("AUDIO: ⚠️ Could not find HALOutput component")
            throw AudioEngineError.noMicrophone
        }
        var unit: AudioUnit?
        try checkOSStatus(AudioComponentInstanceNew(component, &unit), "AudioComponentInstanceNew")
        guard let unit else { throw AudioEngineError.noMicrophone }

        // 2. Enable input on element 1
        var enableIO: UInt32 = 1
        try checkOSStatus(AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Input,
            1,
            &enableIO,
            UInt32(MemoryLayout<UInt32>.size)
        ), "EnableIO input")

        // 3. Disable output on element 0 (capture only)
        var disableIO: UInt32 = 0
        try checkOSStatus(AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_EnableIO,
            kAudioUnitScope_Output,
            0,
            &disableIO,
            UInt32(MemoryLayout<UInt32>.size)
        ), "DisableIO output")

        // 4. Set the input device
        let deviceID: AudioDeviceID
        if let uid = selectedDeviceUID, let resolved = Self.resolveAudioDeviceID(uid: uid) {
            deviceID = resolved
            beeLog("AUDIO: Using device \(uid) (AudioDeviceID: \(deviceID))")
        } else {
            deviceID = Self.defaultInputDeviceID()
            beeLog("AUDIO: Using system default input (AudioDeviceID: \(deviceID))")
        }

        var devID = deviceID
        try checkOSStatus(AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &devID,
            UInt32(MemoryLayout<AudioDeviceID>.size)
        ), "SetCurrentDevice")

        // 5. Query the device's native format (input scope, element 1)
        var deviceFormat = AudioStreamBasicDescription()
        var formatSize = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        try checkOSStatus(AudioUnitGetProperty(
            unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Input,
            1,
            &deviceFormat,
            &formatSize
        ), "GetStreamFormat (device)")

        let nativeRate = deviceFormat.mSampleRate
        let nativeChannels = deviceFormat.mChannelsPerFrame
        beeLog("AUDIO: Device format: \(nativeRate) Hz, \(nativeChannels) ch, \(deviceFormat.mBitsPerChannel) bit")

        guard nativeRate > 0 else { throw AudioEngineError.noMicrophone }

        // 6. Set our client format on output scope, element 1 (mono float32 at native rate)
        var clientFormat = AudioStreamBasicDescription(
            mSampleRate: nativeRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagsNativeFloatPacked | kAudioFormatFlagIsNonInterleaved,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )
        try checkOSStatus(AudioUnitSetProperty(
            unit,
            kAudioUnitProperty_StreamFormat,
            kAudioUnitScope_Output,
            1,
            &clientFormat,
            UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        ), "SetStreamFormat (client)")

        // 7. Allocate render buffer
        let bufferData = UnsafeMutablePointer<Float>.allocate(capacity: Int(renderBufferCapacity))
        renderBufferData = bufferData

        // 8. Set input callback
        var callbackStruct = AURenderCallbackStruct(
            inputProc: auhalInputCallback,
            inputProcRefCon: Unmanaged.passUnretained(self).toOpaque()
        )
        try checkOSStatus(AudioUnitSetProperty(
            unit,
            kAudioOutputUnitProperty_SetInputCallback,
            kAudioUnitScope_Global,
            0,
            &callbackStruct,
            UInt32(MemoryLayout<AURenderCallbackStruct>.size)
        ), "SetInputCallback")

        // 9. Set up cached resampler
        if nativeRate != Self.targetSampleRate {
            let srcFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: nativeRate, channels: 1, interleaved: false)!
            let dstFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Self.targetSampleRate, channels: 1, interleaved: false)!
            resamplerSrcFormat = srcFmt
            resamplerDstFormat = dstFmt
            resampler = AVAudioConverter(from: srcFmt, to: dstFmt)
        } else {
            resampler = nil
            resamplerSrcFormat = nil
            resamplerDstFormat = nil
        }
        resampleAccumulator.removeAll()

        // 10. Initialize and start
        try checkOSStatus(AudioUnitInitialize(unit), "AudioUnitInitialize")

        lock.lock()
        self.nativeSampleRate = nativeRate
        self.channelCount = 1 // we request mono
        preBufferCapacity = Int(nativeRate * preBufferDuration)
        preBuffer = [Float](repeating: 0, count: preBufferCapacity)
        preBufferWriteIndex = 0
        state = .warm
        lock.unlock()

        try checkOSStatus(AudioOutputUnitStart(unit), "AudioOutputUnitStart")
        self.auhalUnit = unit
        beeLog("AUDIO: ✓ Engine warm: \(nativeRate) Hz from device \(deviceID)")
    }

    func coolDown() {
        beeLog("AUDIO: Cooling down")
        if let unit = auhalUnit {
            AudioOutputUnitStop(unit)
            AudioUnitUninitialize(unit)
            AudioComponentInstanceDispose(unit)
            auhalUnit = nil
        }
        renderBufferData?.deallocate()
        renderBufferData = nil
        resampler = nil
        resamplerSrcFormat = nil
        resamplerDstFormat = nil
        resampleAccumulator.removeAll()

        lock.lock()
        state = .cold
        preBuffer.removeAll()
        preBufferCapacity = 0
        preBufferWriteIndex = 0
        currentLevel = 0
        currentRMS = 0
        peakLevel = 0
        totalBuffersReceived = 0
        totalSamplesReceived = 0
        activePipelineCount = 0
        lock.unlock()
    }

    var isWarm: Bool {
        lock.withLock { state == .warm }
    }

    // MARK: - AUHAL Callback

    /// Called by the AUHAL on the audio I/O thread.
    fileprivate func handleInputCallback(
        unit: AudioUnit,
        ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
        inTimeStamp: UnsafePointer<AudioTimeStamp>,
        inBusNumber: UInt32,
        inNumberFrames: UInt32
    ) {
        // Ensure buffer is large enough
        if inNumberFrames > renderBufferCapacity {
            renderBufferData?.deallocate()
            renderBufferCapacity = inNumberFrames * 2
            renderBufferData = UnsafeMutablePointer<Float>.allocate(capacity: Int(renderBufferCapacity))
        }
        guard let bufferData = renderBufferData else { return }

        // Set up AudioBufferList for mono float32
        var bufferList = AudioBufferList(
            mNumberBuffers: 1,
            mBuffers: AudioBuffer(
                mNumberChannels: 1,
                mDataByteSize: inNumberFrames * 4,
                mData: UnsafeMutableRawPointer(bufferData)
            )
        )

        // Pull audio from the AUHAL (always element 1 = input bus)
        let status = AudioUnitRender(unit, ioActionFlags, inTimeStamp, 1, inNumberFrames, &bufferList)
        guard status == noErr else { return }

        let count = Int(inNumberFrames)
        let samples = UnsafeBufferPointer(start: bufferData, count: count)

        lock.lock()

        // Fill circular pre-buffer (native rate)
        if state == .warm && preBufferCapacity > 0 {
            for sample in samples {
                preBuffer[preBufferWriteIndex % preBufferCapacity] = sample
                preBufferWriteIndex += 1
            }
        }

        // Resample to 16kHz and send to pipelines
        if !activePipelines.isEmpty {
            let resampled = resampleCached(Array(samples))
            for (_, pipeline) in activePipelines {
                pipeline.send(resampled)
            }
        }

        // Compute RMS level and stats
        var sumSquares: Float = 0
        var peak: Float = 0
        for sample in samples {
            sumSquares += sample * sample
            let abs = Swift.abs(sample)
            if abs > peak { peak = abs }
        }
        let rms = sqrtf(sumSquares / Float(count))
        currentRMS = rms
        currentLevel = min(1, rms * 5)
        peakLevel = peak
        totalBuffersReceived += 1
        totalSamplesReceived += UInt64(count)
        activePipelineCount = activePipelines.count

        lock.unlock()
    }

    // MARK: - Capture API

    /// Register a Channel 0 pipeline. Copies the pre-buffer (resampled)
    /// as the first chunk, then forwards all subsequent audio.
    func startCapture(for sessionID: UUID, pipeline: RawAudioPipeline) {
        lock.lock()
        defer { lock.unlock() }

        // Collect pre-buffer (native rate)
        var preBufferSamples: [Float] = []
        if preBufferCapacity > 0 {
            if preBufferWriteIndex >= preBufferCapacity {
                let startIndex = preBufferWriteIndex % preBufferCapacity
                preBufferSamples.append(contentsOf: preBuffer[startIndex...])
                preBufferSamples.append(contentsOf: preBuffer[..<startIndex])
            } else {
                preBufferSamples.append(contentsOf: preBuffer[..<preBufferWriteIndex])
            }
        }

        // Resample pre-buffer and send as first chunk
        if !preBufferSamples.isEmpty {
            let resampled = resampleCached(preBufferSamples)
            if !resampled.isEmpty {
                pipeline.send(resampled)
            }
        }

        activePipelines[sessionID] = pipeline
    }

    /// Stop forwarding audio to this session's Channel 0.
    func stopCapture(for sessionID: UUID) {
        lock.lock()
        let pipeline = activePipelines.removeValue(forKey: sessionID)
        lock.unlock()
        pipeline?.finish()
    }

    // MARK: - Resampling

    /// Resample using the cached converter (created once per warmUp).
    /// Uses a fresh converter instance per call to avoid cross-boundary state issues,
    /// but reuses the pre-created formats.
    private func resampleCached(_ samples: [Float]) -> [Float] {
        guard let srcFormat = resamplerSrcFormat,
              let dstFormat = resamplerDstFormat else {
            return samples // no conversion needed (native == target)
        }

        let srcCount = AVAudioFrameCount(samples.count)
        guard let srcBuffer = AVAudioPCMBuffer(pcmFormat: srcFormat, frameCapacity: srcCount) else {
            return samples
        }
        srcBuffer.frameLength = srcCount
        if let cd = srcBuffer.floatChannelData {
            samples.withUnsafeBufferPointer { cd[0].update(from: $0.baseAddress!, count: samples.count) }
        }

        let dstCount = AVAudioFrameCount(Double(srcCount) * dstFormat.sampleRate / srcFormat.sampleRate) + 1
        guard let dstBuffer = AVAudioPCMBuffer(pcmFormat: dstFormat, frameCapacity: dstCount) else {
            return samples
        }

        guard let converter = AVAudioConverter(from: srcFormat, to: dstFormat) else {
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

    // MARK: - Device Management

    func selectDevice(uid: String) {
        beeLog("AUDIO: Device selected: \(uid)")
        selectedDeviceUID = uid
    }

    static func resolveAudioDeviceID(uid: String) -> AudioDeviceID? {
        var deviceID = AudioDeviceID(0)
        var cfUID = uid as CFString
        var translation = AudioValueTranslation(
            mInputData: &cfUID,
            mInputDataSize: UInt32(MemoryLayout<CFString>.size),
            mOutputData: &deviceID,
            mOutputDataSize: UInt32(MemoryLayout<AudioDeviceID>.size)
        )
        var size = UInt32(MemoryLayout<AudioValueTranslation>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDeviceForUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size, &translation
        )
        return status == noErr && deviceID != 0 ? deviceID : nil
    }

    private static func defaultInputDeviceID() -> AudioDeviceID {
        var deviceID = AudioDeviceID(0)
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &address, 0, nil, &size, &deviceID
        )
        return deviceID
    }

    private func checkOSStatus(_ status: OSStatus, _ label: String) throws {
        guard status == noErr else {
            beeLog("AUDIO: ⚠️ \(label) failed: OSStatus \(status)")
            throw AudioEngineError.osStatus(status, label)
        }
    }
}

// MARK: - AUHAL C Callback

private func auhalInputCallback(
    inRefCon: UnsafeMutableRawPointer,
    ioActionFlags: UnsafeMutablePointer<AudioUnitRenderActionFlags>,
    inTimeStamp: UnsafePointer<AudioTimeStamp>,
    inBusNumber: UInt32,
    inNumberFrames: UInt32,
    ioData: UnsafeMutablePointer<AudioBufferList>?
) -> OSStatus {
    let engine = Unmanaged<AudioEngine>.fromOpaque(inRefCon).takeUnretainedValue()
    engine.handleInputCallback(
        unit: engine.auhalUnit!,
        ioActionFlags: ioActionFlags,
        inTimeStamp: inTimeStamp,
        inBusNumber: inBusNumber,
        inNumberFrames: inNumberFrames
    )
    return noErr
}

// MARK: - Errors

enum AudioEngineError: LocalizedError {
    case noMicrophone
    case osStatus(OSStatus, String)

    var errorDescription: String? {
        switch self {
        case .noMicrophone: return "No microphone available"
        case .osStatus(let code, let label): return "\(label) failed (OSStatus \(code))"
        }
    }
}
