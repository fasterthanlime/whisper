import Carbon.HIToolbox.Events
import CoreGraphics
import os

private let logger = Logger(subsystem: "fasterthanlime.bee", category: "Hotkey")

// Right Option key code
private let kRightOption = UInt16(kVK_RightOption)
private let kRightCommand = UInt16(kVK_RightCommand)

// NX_DEVICE mask for right option in CGEventFlags
private let kRightOptionDeviceMask: UInt64 = 0x0000_0040
private let kRightCommandDeviceMask: UInt64 = 0x0000_0010

/// Monitors for Right Option via a CGEvent tap and dispatches to AppState.
/// Runs on the main thread (CGEvent tap runs on the main run loop).
final class HotkeyMonitor: @unchecked Sendable {
    nonisolated(unsafe) var appState: AppState?

    fileprivate var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var rOptHeld = false

    func start() {
        guard eventTap == nil else { return }

        // We need flagsChanged (for ROpt, RCmd) and keyDown (for Escape, Enter, other keys)
        let mask: CGEventMask =
            (1 << CGEventType.flagsChanged.rawValue) |
            (1 << CGEventType.keyDown.rawValue)

        let refcon = Unmanaged.passUnretained(self).toOpaque()

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap, // we need to be able to swallow events
            eventsOfInterest: mask,
            callback: hotkeyCallback,
            userInfo: refcon
        ) else {
            logger.error("Failed to create CGEvent tap — accessibility permission needed")
            return
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetMain(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        logger.info("CGEvent tap started")
    }

    func stop() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        rOptHeld = false
        logger.info("CGEvent tap stopped")
    }

    /// Called from the C callback on the main run loop.
    /// Returns nil to swallow the event, or the event to pass through.
    /// Returns true if the event should be swallowed.
    /// Called from the CGEvent tap callback on the main run loop.
    fileprivate nonisolated func handleEvent(_ type: CGEventType, _ event: CGEvent) -> Bool {
        guard let appState else { return false }
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let flags = event.flags.rawValue
        let isRepeat = event.getIntegerValueField(.keyboardEventAutorepeat) != 0

        return MainActor.assumeIsolated {
            switch type {
            case .flagsChanged:
                return handleFlagsChanged(keyCode: keyCode, flags: flags, appState: appState)
            case .keyDown:
                return handleKeyDown(keyCode: keyCode, isRepeat: isRepeat, appState: appState)
            default:
                return false
            }
        }
    }

    @MainActor
    private func handleFlagsChanged(keyCode: UInt16, flags: UInt64, appState: AppState) -> Bool {
        if keyCode == kRightOption {
            let isDown = (flags & kRightOptionDeviceMask) != 0
            if isDown && !rOptHeld {
                rOptHeld = true
                let swallow = appState.handleROptDown()
                logger.info("ROpt down → swallow=\(swallow)")
                return swallow
            } else if !isDown && rOptHeld {
                rOptHeld = false
                let swallow = appState.handleROptUp()
                logger.info("ROpt up → swallow=\(swallow)")
                return swallow
            }
        }

        if keyCode == kRightCommand {
            let isDown = (flags & kRightCommandDeviceMask) != 0
            if isDown {
                return appState.handleRCmdDown()
            }
        }

        return false
    }

    @MainActor
    private func handleKeyDown(keyCode: UInt16, isRepeat: Bool, appState: AppState) -> Bool {
        if isRepeat { return false }

        switch Int(keyCode) {
        case kVK_Escape:
            return appState.handleEscape()
        case kVK_Return, kVK_ANSI_KeypadEnter:
            return appState.handleEnter()
        default:
            return appState.handleOtherKey(keyCode: keyCode)
        }
    }
}

// C callback — must be a free function
private func hotkeyCallback(
    proxy: CGEventTapProxy,
    type: CGEventType,
    event: CGEvent,
    refcon: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    // Re-enable tap if it was disabled by the system (too slow)
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        if let refcon {
            let monitor = Unmanaged<HotkeyMonitor>.fromOpaque(refcon).takeUnretainedValue()
            if let tap = monitor.eventTap {
                CGEvent.tapEnable(tap: tap, enable: true)
            }
        }
        return Unmanaged.passUnretained(event)
    }

    guard let refcon else {
        return Unmanaged.passUnretained(event)
    }

    let monitor = Unmanaged<HotkeyMonitor>.fromOpaque(refcon).takeUnretainedValue()
    let swallow = monitor.handleEvent(type, event)
    return swallow ? nil : Unmanaged.passUnretained(event)
}
