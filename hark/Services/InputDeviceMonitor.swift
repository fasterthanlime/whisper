import AudioToolbox
import CoreAudio
import Foundation
import os

struct InputAudioDevice: Identifiable, Hashable, Sendable {
    let uid: String
    let name: String
    let isBuiltIn: Bool
    let isDefault: Bool

    var id: String { uid }
}

struct InputDeviceSnapshot: Sendable {
    let devices: [InputAudioDevice]

    var activeDevice: InputAudioDevice? {
        devices.first(where: \.isDefault)
    }
}

/// Monitors system input devices and notifies when the default input or device list changes.
final class InputDeviceMonitor: @unchecked Sendable {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "hark",
        category: "InputDeviceMonitor"
    )

    private var onDeviceChange: (@Sendable (InputDeviceSnapshot) -> Void)?
    private var defaultInputListenerBlock: AudioObjectPropertyListenerBlock?
    private var devicesListenerBlock: AudioObjectPropertyListenerBlock?
    private let lock = NSLock()

    /// Start monitoring for input device and default-route changes.
    func start(onDeviceChange: @escaping @Sendable (InputDeviceSnapshot) -> Void) {
        stop()

        lock.lock()
        self.onDeviceChange = onDeviceChange
        lock.unlock()

        var defaultAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let defaultBlock: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.notifySnapshot()
        }
        defaultInputListenerBlock = defaultBlock

        let defaultStatus = AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &defaultAddress,
            nil,
            defaultBlock
        )
        if defaultStatus != noErr {
            Self.logger.error(
                "Failed to add default-input listener: status=\(defaultStatus, privacy: .public)"
            )
        }

        var devicesAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let devicesBlock: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.notifySnapshot()
        }
        devicesListenerBlock = devicesBlock

        let devicesStatus = AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &devicesAddress,
            nil,
            devicesBlock
        )
        if devicesStatus != noErr {
            Self.logger.error("Failed to add devices listener: status=\(devicesStatus, privacy: .public)")
        }

        notifySnapshot()
    }

    /// Stop monitoring.
    func stop() {
        if let defaultInputListenerBlock {
            var defaultAddress = AudioObjectPropertyAddress(
                mSelector: kAudioHardwarePropertyDefaultInputDevice,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )

            AudioObjectRemovePropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject),
                &defaultAddress,
                nil,
                defaultInputListenerBlock
            )
        }

        if let devicesListenerBlock {
            var devicesAddress = AudioObjectPropertyAddress(
                mSelector: kAudioHardwarePropertyDevices,
                mScope: kAudioObjectPropertyScopeGlobal,
                mElement: kAudioObjectPropertyElementMain
            )

            AudioObjectRemovePropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject),
                &devicesAddress,
                nil,
                devicesListenerBlock
            )
        }

        defaultInputListenerBlock = nil
        devicesListenerBlock = nil

        lock.lock()
        onDeviceChange = nil
        lock.unlock()
    }

    /// Current snapshot of available input devices.
    func getCurrentSnapshot() -> InputDeviceSnapshot {
        let defaultDeviceID = getDefaultInputDeviceID()
        let pairedDevices = getAllAudioDeviceIDs().compactMap { deviceID -> (AudioDeviceID, InputAudioDevice)? in
            guard supportsInput(deviceID: deviceID) else { return nil }
            guard let uid = getDeviceUID(deviceID: deviceID) else { return nil }
            guard let name = getDeviceName(deviceID: deviceID), !name.isEmpty else { return nil }
            let isDefault = defaultDeviceID.map { $0 == deviceID } ?? false
            let device = InputAudioDevice(
                uid: uid,
                name: name,
                isBuiltIn: isBuiltIn(deviceID: deviceID),
                isDefault: isDefault
            )
            return (deviceID, device)
        }
        let sorted = pairedDevices
            .map(\.1)
            .sorted { lhs, rhs in
                if lhs.isDefault != rhs.isDefault {
                    return lhs.isDefault && !rhs.isDefault
                }
                return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
            }
        return InputDeviceSnapshot(devices: sorted)
    }

    /// Set the system default input device by UID.
    @discardableResult
    func setDefaultInputDevice(uid: String) -> Bool {
        let pairedDevices = getAllAudioDeviceIDs().compactMap { deviceID -> (AudioDeviceID, String)? in
            guard supportsInput(deviceID: deviceID), let deviceUID = getDeviceUID(deviceID: deviceID) else {
                return nil
            }
            return (deviceID, deviceUID)
        }
        guard let targetID = pairedDevices.first(where: { $0.1 == uid })?.0 else {
            Self.logger.error("Unable to set input device: unknown uid=\(uid, privacy: .public)")
            return false
        }

        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var mutableTargetID = targetID
        let status = AudioObjectSetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            UInt32(MemoryLayout<AudioDeviceID>.size),
            &mutableTargetID
        )
        guard status == noErr else {
            Self.logger.error(
                "Failed to set default input device uid=\(uid, privacy: .public) status=\(status, privacy: .public)"
            )
            return false
        }

        notifySnapshot()
        return true
    }

    /// Current default input device name (compat helper).
    func getCurrentInputDeviceName() -> String? {
        getCurrentSnapshot().activeDevice?.name
    }

    private func notifySnapshot() {
        let snapshot = getCurrentSnapshot()
        Self.logger.info(
            "Input devices refreshed: count=\(snapshot.devices.count, privacy: .public), active=\(snapshot.activeDevice?.name ?? "none", privacy: .public)"
        )

        lock.lock()
        let callback = onDeviceChange
        lock.unlock()
        callback?(snapshot)
    }

    private func getDefaultInputDeviceID() -> AudioDeviceID? {
        var deviceID = AudioDeviceID(0)
        var deviceIDSize = UInt32(MemoryLayout.size(ofValue: deviceID))

        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &deviceIDSize,
            &deviceID
        )

        guard status == noErr, deviceID != kAudioObjectUnknown else {
            return nil
        }

        return deviceID
    }

    private func getDeviceName(deviceID: AudioDeviceID) -> String? {
        var nameRef: Unmanaged<CFString>?
        var nameSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)

        var nameAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceNameCFString,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let status = AudioObjectGetPropertyData(
            deviceID,
            &nameAddress,
            0,
            nil,
            &nameSize,
            &nameRef
        )

        guard status == noErr, let name = nameRef?.takeUnretainedValue() else { return nil }
        return name as String
    }

    private func getDeviceUID(deviceID: AudioDeviceID) -> String? {
        var uidRef: Unmanaged<CFString>?
        var uidSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)

        var uidAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let status = AudioObjectGetPropertyData(
            deviceID,
            &uidAddress,
            0,
            nil,
            &uidSize,
            &uidRef
        )
        guard status == noErr, let uid = uidRef?.takeUnretainedValue() else { return nil }
        return uid as String
    }

    private func getAllAudioDeviceIDs() -> [AudioDeviceID] {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        var size: UInt32 = 0
        let sizeStatus = AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &size
        )
        guard sizeStatus == noErr else {
            Self.logger.error(
                "Failed to read device list size: status=\(sizeStatus, privacy: .public)"
            )
            return []
        }

        let count = Int(size) / MemoryLayout<AudioDeviceID>.size
        guard count > 0 else { return [] }
        var deviceIDs = [AudioDeviceID](repeating: 0, count: count)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            0,
            nil,
            &size,
            &deviceIDs
        )
        guard status == noErr else {
            Self.logger.error("Failed to read device list: status=\(status, privacy: .public)")
            return []
        }
        return deviceIDs
    }

    private func supportsInput(deviceID: AudioDeviceID) -> Bool {
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreams,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        let status = AudioObjectGetPropertyDataSize(
            deviceID,
            &propertyAddress,
            0,
            nil,
            &size
        )
        return status == noErr && size >= UInt32(MemoryLayout<AudioStreamID>.size)
    }

    private func isBuiltIn(deviceID: AudioDeviceID) -> Bool {
        var transportType: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyTransportType,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(
            deviceID,
            &propertyAddress,
            0,
            nil,
            &size,
            &transportType
        )
        return status == noErr && transportType == kAudioDeviceTransportTypeBuiltIn
    }

    deinit {
        stop()
    }
}
