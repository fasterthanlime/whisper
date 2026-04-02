import AVFoundation
import CoreAudio
import Foundation

enum AudioDiagnostics {
    static func dumpAllDevices() -> String {
        var output = ""

        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .external],
            mediaType: .audio,
            position: .unspecified
        )
        let defaultUID = AVCaptureDevice.default(for: .audio)?.uniqueID

        for device in discovery.devices {
            output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            output += "Device: \(device.localizedName)\n"
            output += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            output += "  UID:           \(device.uniqueID)\n"
            output += "  Model ID:      \(device.modelID)\n"
            output += "  Manufacturer:  \(device.manufacturer)\n"
            output += "  Device Type:   \(device.deviceType.rawValue)\n"
            output += "  Is Default:    \(device.uniqueID == defaultUID)\n"
            output += "  Is Connected:  \(device.isConnected)\n"
            output += "  Is Suspended:  \(device.isSuspended)\n"

            // CoreAudio properties via AudioDeviceID
            if let deviceID = resolveDeviceID(uid: device.uniqueID) {
                output += "\n  --- CoreAudio Properties ---\n"
                output += "  AudioDeviceID: \(deviceID)\n"

                if let transport = getStringProperty(deviceID, selector: kAudioDevicePropertyTransportType) {
                    output += "  Transport:     \(transport)\n"
                }
                if let transportRaw = getUInt32Property(deviceID, selector: kAudioDevicePropertyTransportType) {
                    output += "  Transport Raw: 0x\(String(transportRaw, radix: 16)) (\(fourCC(transportRaw)))\n"
                }
                if let name = getCFStringProperty(deviceID, selector: kAudioObjectPropertyName) {
                    output += "  CA Name:       \(name)\n"
                }
                if let manufacturer = getCFStringProperty(deviceID, selector: kAudioObjectPropertyManufacturer) {
                    output += "  CA Manufacturer: \(manufacturer)\n"
                }
                if let modelUID = getCFStringProperty(deviceID, selector: kAudioDevicePropertyModelUID) {
                    output += "  Model UID:     \(modelUID)\n"
                }
                if let icon = getCFStringProperty(deviceID, selector: kAudioDevicePropertyIcon) {
                    output += "  Icon:          \(icon)\n"
                }
                if let clockDomain = getUInt32Property(deviceID, selector: kAudioDevicePropertyClockDomain) {
                    output += "  Clock Domain:  \(clockDomain)\n"
                }
                if let isAlive = getUInt32Property(deviceID, selector: kAudioDevicePropertyDeviceIsAlive) {
                    output += "  Is Alive:      \(isAlive != 0)\n"
                }
                if let isRunning = getUInt32Property(deviceID, selector: kAudioDevicePropertyDeviceIsRunning) {
                    output += "  Is Running:    \(isRunning != 0)\n"
                }
                if let canBeDefault = getUInt32Property(deviceID, selector: kAudioDevicePropertyDeviceCanBeDefaultDevice, scope: kAudioDevicePropertyScopeInput) {
                    output += "  Can Be Default (input): \(canBeDefault != 0)\n"
                }

                // Input stream info
                let inputChannels = getChannelCount(deviceID, scope: kAudioDevicePropertyScopeInput)
                output += "  Input Channels: \(inputChannels)\n"

                if let nominalRate = getFloat64Property(deviceID, selector: kAudioDevicePropertyNominalSampleRate) {
                    output += "  Sample Rate:   \(nominalRate) Hz\n"
                }

                // Available sample rates
                if let rates = getAvailableSampleRates(deviceID) {
                    let rateStrs = rates.map { r in
                        if r.mMinimum == r.mMaximum {
                            return "\(Int(r.mMinimum))"
                        }
                        return "\(Int(r.mMinimum))-\(Int(r.mMaximum))"
                    }
                    output += "  Available Rates: \(rateStrs.joined(separator: ", "))\n"
                }

                if let latency = getUInt32Property(deviceID, selector: kAudioDevicePropertyLatency, scope: kAudioDevicePropertyScopeInput) {
                    output += "  Input Latency: \(latency) frames\n"
                }
                if let bufferSize = getUInt32Property(deviceID, selector: kAudioDevicePropertyBufferFrameSize) {
                    output += "  Buffer Size:   \(bufferSize) frames\n"
                }
            }

            output += "\n"
        }

        if discovery.devices.isEmpty {
            output = "No audio input devices found.\n"
        }

        return output
    }

    // MARK: - CoreAudio helpers

    private static func resolveDeviceID(uid: String) -> AudioDeviceID? {
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

    private static func getUInt32Property(_ id: AudioDeviceID, selector: AudioObjectPropertySelector, scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal) -> UInt32? {
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value)
        return status == noErr ? value : nil
    }

    private static func getFloat64Property(_ id: AudioDeviceID, selector: AudioObjectPropertySelector) -> Float64? {
        var value: Float64 = 0
        var size = UInt32(MemoryLayout<Float64>.size)
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value)
        return status == noErr ? value : nil
    }

    private static func getCFStringProperty(_ id: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        var value: CFString?
        var size = UInt32(MemoryLayout<CFString?>.size)
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyData(id, &address, 0, nil, &size, &value)
        return status == noErr ? value as String? : nil
    }

    private static func getStringProperty(_ id: AudioDeviceID, selector: AudioObjectPropertySelector) -> String? {
        guard let raw = getUInt32Property(id, selector: selector) else { return nil }
        return transportName(raw)
    }

    private static func getChannelCount(_ id: AudioDeviceID, scope: AudioObjectPropertyScope) -> Int {
        var size: UInt32 = 0
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyStreamConfiguration, mScope: scope, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
        guard status == noErr, size > 0 else { return 0 }

        let bufferListPtr = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { bufferListPtr.deallocate() }
        let status2 = AudioObjectGetPropertyData(id, &address, 0, nil, &size, bufferListPtr)
        guard status2 == noErr else { return 0 }

        let bufferList = bufferListPtr.assumingMemoryBound(to: AudioBufferList.self).pointee
        var channels = 0
        withUnsafePointer(to: bufferList.mBuffers) { ptr in
            for i in 0..<Int(bufferList.mNumberBuffers) {
                let buf = UnsafeRawPointer(ptr).advanced(by: i * MemoryLayout<AudioBuffer>.stride)
                    .assumingMemoryBound(to: AudioBuffer.self).pointee
                channels += Int(buf.mNumberChannels)
            }
        }
        return channels
    }

    private static func getAvailableSampleRates(_ id: AudioDeviceID) -> [AudioValueRange]? {
        var size: UInt32 = 0
        var address = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyAvailableNominalSampleRates, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
        let status = AudioObjectGetPropertyDataSize(id, &address, 0, nil, &size)
        guard status == noErr, size > 0 else { return nil }

        let count = Int(size) / MemoryLayout<AudioValueRange>.size
        var ranges = [AudioValueRange](repeating: AudioValueRange(), count: count)
        let status2 = AudioObjectGetPropertyData(id, &address, 0, nil, &size, &ranges)
        return status2 == noErr ? ranges : nil
    }

    private static func transportName(_ raw: UInt32) -> String {
        switch raw {
        case kAudioDeviceTransportTypeBuiltIn: return "Built-in"
        case kAudioDeviceTransportTypeUSB: return "USB"
        case kAudioDeviceTransportTypeBluetooth: return "Bluetooth"
        case kAudioDeviceTransportTypeBluetoothLE: return "Bluetooth LE"
        case kAudioDeviceTransportTypeVirtual: return "Virtual"
        case kAudioDeviceTransportTypeAggregate: return "Aggregate"
        case kAudioDeviceTransportTypeFireWire: return "FireWire"
        case kAudioDeviceTransportTypePCI: return "PCI"
        case kAudioDeviceTransportTypeThunderbolt: return "Thunderbolt"
        default: return "Unknown (0x\(String(raw, radix: 16)))"
        }
    }

    private static func fourCC(_ value: UInt32) -> String {
        let chars = [
            Character(UnicodeScalar((value >> 24) & 0xFF)!),
            Character(UnicodeScalar((value >> 16) & 0xFF)!),
            Character(UnicodeScalar((value >> 8) & 0xFF)!),
            Character(UnicodeScalar(value & 0xFF)!),
        ]
        return String(chars)
    }
}
