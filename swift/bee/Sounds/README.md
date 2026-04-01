# Bee custom sound effects

Place custom WAV files in this folder with exactly these names:

- `bee-recording-started.wav`
- `bee-start-failure.wav`
- `bee-commit.wav`
- `bee-commit-submit.wav`
- `bee-cancel.wav`

These files are loaded at runtime from the app bundle by `SoundEffects`.
If a file is missing, Bee skips that sound effect.
