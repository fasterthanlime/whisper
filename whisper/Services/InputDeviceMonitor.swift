import AudioToolbox
import CoreAudio
import Foundation
import os

/// Monitors the default audio input device and notifies when it changes.
final class InputDeviceMonitor: @unchecked Sendable {
    private static let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "whisper",
        category: "InputDeviceMonitor"
    )

    private var onDeviceChange: (@Sendable (String?) -> Void)?
    private var listenerBlock: AudioObjectPropertyListenerBlock?
    private let lock = NSLock()

    /// Start monitoring for input device changes.
    func start(onDeviceChange: @escaping @Sendable (String?) -> Void) {
        lock.lock()
        self.onDeviceChange = onDeviceChange
        lock.unlock()

        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        let block: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            guard let self else { return }
            let name = self.getCurrentInputDeviceName()
            Self.logger.info("Input device changed to: \(name ?? "none", privacy: .public)")
            self.lock.lock()
            let callback = self.onDeviceChange
            self.lock.unlock()
            callback?(name)
        }

        listenerBlock = block
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            nil,
            block
        )

        // Report initial device
        let initialName = getCurrentInputDeviceName()
        Self.logger.info("Initial input device: \(initialName ?? "none", privacy: .public)")
        onDeviceChange(initialName)
    }

    /// Stop monitoring.
    func stop() {
        guard let block = listenerBlock else { return }

        var propertyAddress = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )

        AudioObjectRemovePropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject),
            &propertyAddress,
            nil,
            block
        )

        listenerBlock = nil
        lock.lock()
        onDeviceChange = nil
        lock.unlock()
    }

    /// Get the name of the current default input device.
    func getCurrentInputDeviceName() -> String? {
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

        return getDeviceName(deviceID: deviceID)
    }

    private func getDeviceName(deviceID: AudioDeviceID) -> String? {
        var name: CFString = "" as CFString
        var nameSize = UInt32(MemoryLayout<CFString>.size)

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
            &name
        )

        guard status == noErr else { return nil }
        return name as String
    }

    deinit {
        stop()
    }
}
