# Hark

A macOS menu bar app for on-device speech-to-text. Hold a hotkey, speak, release — transcribed text is pasted into the active application automatically.

All processing runs locally using [Qwen3 ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) models via a vendored Rust inference engine ([qwen3-asr-rs](qwen3-asr-rs/)) with Metal GPU acceleration. No audio leaves your machine.

<p>
<em>Streaming overlay — partial transcription appears in real time as you speak</em><br>
<img width="600" alt="Streaming overlay showing real-time transcription" src="https://github.com/user-attachments/assets/63039c9b-6de2-4947-adf8-5d3671b11e73" />
</p>

<p>
<em>Menu bar dropdown — model picker, per-app language & vocab, permissions, settings</em><br>
<img width="400" alt="Menu bar dropdown with settings" src="https://github.com/user-attachments/assets/c70eaf96-96bd-4bc3-85e3-78c036f81909" />
</p>

## Features

### Push-to-talk with streaming transcription

Hold your hotkey and speak — partial results appear in the overlay in real time as you talk. When you release the key, the final transcript is pasted into whatever app you're using.

### Toggle mode

Quick-press the hotkey (< 300ms) to lock recording on hands-free. Press again to stop and paste. You can also hold the hotkey and press Command to lock without releasing.

### Voice commands

- Say **"over"** to paste the current sentence and keep recording
- Say **"over and out"** to paste and stop recording entirely

### Per-app language and vocabulary

Set a different language per application — Auto, English, French, Spanish, German, or Polish. Each app also gets its own vocabulary prompt to help the model with domain-specific terms (e.g. "serde, candle, GGUF" for your code editor).

### Multiple models

Choose between Qwen3 ASR 0.6B and 1.7B in full-precision or Q4_K quantized formats. Models are downloaded on demand from HuggingFace and cached locally.

| Model | Format | Download |
|-------|--------|----------|
| Qwen3 ASR 0.6B | safetensors | ~1.2 GB |
| Qwen3 ASR 0.6B | Q8_0 GGUF | ~1.0 GB |
| Qwen3 ASR 0.6B | Q4_K GGUF | ~605 MB |
| Qwen3 ASR 1.7B | safetensors | ~3.4 GB |
| Qwen3 ASR 1.7B | Q8_0 GGUF | ~2.5 GB |
| Qwen3 ASR 1.7B | Q4_K GGUF | ~1.3 GB |

### Smart paste

Transcribed text is written to the pasteboard, Cmd+V is simulated via the Accessibility API, and the original clipboard contents are restored afterward. A space is automatically prepended when the cursor follows non-whitespace. In terminal apps (iTerm2, Ghostty), Enter is sent automatically after pasting.

### Visual feedback

An animated floating overlay with a MeshGradient responds to real-time audio level and FFT spectrum. Partial transcription text streams into the overlay as you speak. A spinner appears during final inference.

### Other features

- **Configurable hotkey** — any key combination, left/right modifier aware
- **Audio device hot-swap** — seamlessly switches when you plug/unplug microphones
- **Pause media while dictating** — optionally pauses playback during recording
- **Transcription history** — last 20 transcriptions in the menu, click to copy
- **Run on startup** — uses SMAppService for native login item support
- **Escape to cancel** — press Escape (or hotkey + Escape in toggle mode) to cancel without pasting
- **Return to submit** — press Return while recording to stop and force-submit with Enter

## Requirements

- macOS on Apple Silicon
- Xcode (for building from source)
- Microphone permission
- Accessibility permission (for simulating paste keystrokes and detecting cursor context)

## Building

The app links against `libqwen3_asr_ffi`, a static library built from the vendored `qwen3-asr-rs/` Rust crate.

1. Build the Rust library:
   ```
   cd qwen3-asr-rs && cargo build --release
   ```

2. Open `hark.xcodeproj` in Xcode. Go to the **hark** target → **Signing & Capabilities**, enable **Automatically manage signing**, and select your Team.

3. Build and install:
   ```
   xcodebuild -project hark.xcodeproj -scheme hark -configuration Release -derivedDataPath build clean build
   cp -R "build/Build/Products/Release/hark.app" /Applications/
   open /Applications/hark.app
   ```

4. Grant **Microphone** and **Accessibility** permissions when prompted.

> **If macOS blocks launch** — right-click the app and choose Open, or:
> ```
> xattr -dr com.apple.quarantine /Applications/hark.app
> ```

## Architecture

```
harkApp.swift             App entry, hotkey wiring, streaming loop, state machine
AppState.swift            Observable state (idle/recording/transcribing/pasting/error)

Services/
  TranscriptionService    Rust FFI wrapper — streaming sessions + single-shot fallback
  AudioRecorder           AVAudioEngine capture, RMS level, FFT spectrum, 16 kHz resampling
  PasteController         Pasteboard snapshot/restore, Cmd+V simulation
  InputDeviceMonitor      Audio device connect/disconnect handling
  MediaController         Pause/resume media during recording

Views/
  MenuBarView             Dropdown menu (models, permissions, per-app settings)
  RecordingOverlay        Animated MeshGradient circle + partial transcript
  OverlayManager          Per-screen overlay lifecycle
  OverlayPanel            Non-activating transparent NSPanel

Models/
  STTModelDefinition      Model registry (name, repo, quantization)

Hotkey/
  HotkeyDefinitions       CGEvent tap, modifier-aware key combos, persistence

qwen3-asr-rs/             Vendored Rust ASR engine (Candle + Metal)
  qwen3-asr-ffi/          C FFI layer consumed by the Swift app
```

## License

See [LICENSE](LICENSE) for details.
