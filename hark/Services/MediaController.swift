import AppKit
import CoreAudio

/// Pauses/resumes system media playback during dictation.
@MainActor
struct MediaController {
    private static var didPauseMedia = false

    /// Pause system media if any process (other than ourselves) is driving audio output.
    static func pauseIfPlaying() {
        didPauseMedia = false

        let myPID = ProcessInfo.processInfo.processIdentifier
        let (active, bundle, pid) = findActiveAudioOutput(excludingPID: myPID)
        print("[media] Audio output active: \(active)\(active ? " (pid=\(pid) bundle=\(bundle ?? "?"))" : "")")

        guard active else { return }
        mediaRemoteSendCommand(1) // kMRPause
        didPauseMedia = true
        print("[media] Sent pause command")
    }

    /// If we paused media earlier, resume it.
    static func resumeIfPaused() {
        guard didPauseMedia else { return }
        mediaRemoteSendCommand(0) // kMRPlay
        didPauseMedia = false
        print("[media] Sent play command")
    }

    /// Setting stored in UserDefaults.
    static var isEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "pauseMediaWhileDictating") }
        set { UserDefaults.standard.set(newValue, forKey: "pauseMediaWhileDictating") }
    }

    // MARK: - Core Audio Process Detection

    /// Check if any process (excluding the given PID) is currently producing audio output.
    private static func findActiveAudioOutput(excludingPID: pid_t) -> (active: Bool, bundle: String?, pid: pid_t) {
        for obj in getProcessObjects() {
            let pid = getProcessPID(obj)
            guard pid != excludingPID else { continue }
            if isProcessRunningOutput(obj) {
                return (true, getProcessBundleID(obj), pid)
            }
        }
        return (false, nil, 0)
    }

    private static func getProcessObjects() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size
        ) == noErr else { return [] }

        let count = Int(size) / MemoryLayout<AudioObjectID>.size
        var objects = [AudioObjectID](repeating: 0, count: count)
        guard AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &objects
        ) == noErr else { return [] }
        return objects
    }

    private static func getProcessPID(_ obj: AudioObjectID) -> pid_t {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyPID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var pid: pid_t = 0
        var size = UInt32(MemoryLayout<pid_t>.size)
        AudioObjectGetPropertyData(obj, &address, 0, nil, &size, &pid)
        return pid
    }

    private static func getProcessBundleID(_ obj: AudioObjectID) -> String? {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyBundleID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var cfStr: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(obj, &address, 0, nil, &size, &cfStr) == noErr,
              let str = cfStr?.takeUnretainedValue() else { return nil }
        return str as String
    }

    private static func isProcessRunningOutput(_ obj: AudioObjectID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyIsRunningOutput,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var running: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        AudioObjectGetPropertyData(obj, &address, 0, nil, &size, &running)
        return running != 0
    }

    // MARK: - MediaRemote (for sending pause/play commands)

    private static let mrHandle: UnsafeMutableRawPointer? = {
        dlopen("/System/Library/PrivateFrameworks/MediaRemote.framework/MediaRemote", RTLD_LAZY)
    }()

    private static func mediaRemoteSendCommand(_ command: UInt32) {
        guard let handle = mrHandle,
              let sym = dlsym(handle, "MRMediaRemoteSendCommand") else {
            print("[media] sendCommand: no handle or sym")
            return
        }
        typealias Fn = @convention(c) (UInt32, UnsafeRawPointer?) -> Bool
        let ok = unsafeBitCast(sym, to: Fn.self)(command, nil)
        print("[media] sendCommand(\(command)) returned \(ok)")
    }
}
