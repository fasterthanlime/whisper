import AppKit

@MainActor
final class SoundEffects {
    static let shared = SoundEffects()

    private var tinkSound: NSSound?
    private var popSound: NSSound?
    private var didWarmUp = false

    private init() {
        preload()
    }

    private func preload() {
        if tinkSound == nil, let sound = NSSound(named: "Tink") {
            sound.loops = false
            _ = sound.duration
            tinkSound = sound
        }
        if popSound == nil, let sound = NSSound(named: "Pop") {
            sound.loops = false
            _ = sound.duration
            popSound = sound
        }
    }

    /// Play a silent sound to warm up the audio playback path.
    /// Avoids latency on the first real sound.
    func warmUp() {
        guard !didWarmUp else { return }
        didWarmUp = true

        preload()
        guard let sound = tinkSound else { return }
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
        play(tinkSound, volume: systemVolume() * 0.15)
    }

    func playCommit() {
        play(tinkSound, volume: systemVolume() * 0.1)
    }

    func playCancel() {
        play(popSound, volume: systemVolume() * 0.15)
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
}
