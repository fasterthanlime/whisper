import AppKit
import os

@MainActor
final class SoundEffects {
    static let shared = SoundEffects()

    private enum Effect: CaseIterable {
        case recordingStarted
        case startFailure
        case commit
        case commitSubmit
        case cancel

        var fileName: String {
            switch self {
            case .recordingStarted: "bee-recording-started"
            case .startFailure: "bee-start-failure"
            case .commit: "bee-commit"
            case .commitSubmit: "bee-commit-submit"
            case .cancel: "bee-cancel"
            }
        }
    }

    private static let logger = Logger(subsystem: "fasterthanlime.bee", category: "SoundEffects")

    private var recordingStartedSound: NSSound?
    private var startFailureSound: NSSound?
    private var commitSound: NSSound?
    private var commitSubmitSound: NSSound?
    private var cancelSound: NSSound?
    private var didWarmUp = false

    private init() {
        preload()
    }

    private func preload() {
        for effect in Effect.allCases {
            switch effect {
            case .recordingStarted:
                if recordingStartedSound == nil {
                    recordingStartedSound = load(effect: effect)
                }
            case .startFailure:
                if startFailureSound == nil {
                    startFailureSound = load(effect: effect)
                }
            case .commit:
                if commitSound == nil {
                    commitSound = load(effect: effect)
                }
            case .commitSubmit:
                if commitSubmitSound == nil {
                    commitSubmitSound = load(effect: effect)
                }
            case .cancel:
                if cancelSound == nil {
                    cancelSound = load(effect: effect)
                }
            }
        }
    }

    /// Play a silent sound to warm up the audio playback path.
    /// Avoids latency on the first real sound.
    func warmUp() {
        guard !didWarmUp else { return }
        didWarmUp = true

        preload()
        guard let sound = recordingStartedSound else { return }
        let originalVolume = sound.volume
        sound.stop()
        sound.currentTime = 0
        sound.volume = 0
        sound.play()
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.05) {
            sound.stop()
            sound.volume = originalVolume
        }
    }

    func playRecordingStarted() {
        play(recordingStartedSound, volume: systemVolume())
    }

    func playCommit() {
        play(commitSound, volume: systemVolume())
    }

    func playStartFailure() {
        play(startFailureSound, volume: systemVolume())
    }

    func playCancel() {
        play(cancelSound, volume: systemVolume())
    }

    func playCommitSubmit() {
        play(commitSubmitSound, volume: systemVolume())
    }

    private func play(_ sound: NSSound?, volume: Float) {
        guard let sound else { return }
        sound.stop()
        sound.currentTime = 0
        sound.volume = volume
        sound.play()
    }

    private func systemVolume() -> Float {
        // Use the alert volume from UserDefaults (0.0–1.0)
        let vol = UserDefaults.standard.float(forKey: "com.apple.sound.beep.volume")
        return vol > 0 ? vol : 0.5
    }

    private func load(effect: Effect) -> NSSound? {
        // Prefer Sounds/ subdirectory in bundle, fallback to bundle root for convenience.
        let url = Bundle.main.url(forResource: effect.fileName, withExtension: "wav", subdirectory: "Sounds")
            ?? Bundle.main.url(forResource: effect.fileName, withExtension: "wav")

        guard let url else {
            Self.logger.notice("Missing custom SFX file: \(effect.fileName, privacy: .public).wav")
            return nil
        }

        guard let sound = NSSound(contentsOf: url, byReference: false) else {
            Self.logger.error("Failed to load custom SFX file: \(url.path, privacy: .public)")
            return nil
        }

        sound.loops = false
        _ = sound.duration
        return sound
    }
}
