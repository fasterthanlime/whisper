# Bee State Management Spec

Bee is a push-to-talk dictation tool for macOS. It runs entirely on-device,
captures audio, streams it through an ASR model, and inserts the resulting
text into the frontmost application via a custom IME.

> h[spec.source-of-truth]
> This file is the source of truth for Bee state management and IME behavior.
> Any older IME design notes are non-authoritative and MUST NOT override this
> spec.

## Architecture

There are two layers:

- **UI layer** — handles hotkey events, shows a status indicator at the
  cursor, decides when to create sessions and how they end. Runs on the
  main thread.
- **Session** — a self-contained unit of work created for each dictation
  attempt. Owns its capture buffer, ASR handle, and IME handle. Multiple
  sessions can coexist (e.g., the previous one finalizing while a new one
  is streaming).

The UI layer sends exactly three messages to end a session:

- **Abort** — immediate teardown, no trace. It never happened.
- **Cancel** — finalize in the background, create a history entry, but
  clear the IME (don't insert text).
- **Commit(submit: bool)** — finalize, insert text via IME. If submit is
  true, simulate a Return keystroke after insertion.

Once the UI layer ends a session, it is back to Idle and ready for the
next activation. The session finishes its work independently.

## Audio Engine

The audio engine is shared infrastructure. It is not part of any session.
It runs continuously when warm, capturing audio into a circular pre-buffer
that is discarded unless a session taps into it.

### Device management

> h[audio.device.warm-policy]
> Each device has a user-configurable warm/cold policy, persisted by device
> UID. Warm devices have lower activation latency but keep the microphone
> indicator lit and use CPU.

> h[audio.device.active-selection]
> The user may select which input device Bee uses. The selection is
> independent of the system default — Bee MUST NOT change the system
> default recording device.

> h[audio.device.hotplug]
> When input devices appear or disappear, the device list is rebuilt from
> Core Audio. If the selected device disappeared, Bee falls back to the
> system default.

### Internal format

> h[audio.internal-format]
> The audio engine always outputs 16kHz mono float32 samples, regardless
> of the input device's native format. Resampling happens inside the
> engine. Sessions and the ASR layer never deal with sample rates or
> channel counts.

> h[audio.resample-on-device-change]
> When the active device changes (including mid-recording), the engine
> reconfigures its resampling pipeline for the new device's native format.
> The output format remains 16kHz mono float32.

### Engine states

The audio engine has two states: Cold and Warm.

> h[audio.cold]
> When cold, no AVAudioEngine resources are allocated. If a session is
> created while the engine is cold, there will be a brief startup delay.

> h[audio.warm]
> When warm, the AVAudioEngine is running with a tap installed. Incoming
> audio fills a circular pre-buffer (~200ms) that continuously overwrites
> itself. This keeps the microphone ready for instant capture.

Events while Cold:

| Event | Effect |
|---|---|
| Warm-up requested | → Warm (allocate engine, install tap) |
| Device appeared/disappeared | Rebuild device list, stay Cold |

Events while Warm:

| Event | Effect |
|---|---|
| Cool-down requested | → Cold (stop engine, release resources) |
| Device disappeared (active) | Rebuild list, restart engine on fallback device |
| Device appeared/disappeared (other) | Rebuild list, stay Warm |

## UI Layer

The UI layer handles keyboard events and manages the overlay. It creates
sessions and tells them how to end.

### Key event handling

Bee observes all key events via CGEvent tap. By default, events pass
through to the app — Bee reacts but doesn't prevent the app from seeing
them. Some events are **swallowed**: Bee prevents the app from receiving
them entirely.

> h[ui.swallow-policy]
> Key events are swallowed only when explicitly marked below. All other
> events pass through to the app even if Bee reacts to them.

### States

```
Idle → Pending → PushToTalk → Idle (commit/cancel)
               → Locked ⇄ LockedOptionHeld → Idle (commit/cancel)
               → Idle (abort)
```

#### Idle

Not recording, no active session.

> h[ui.idle-to-pending]
> On ROpt down: create a new session, transition to Pending.

| Event | Swallowed? | Effect |
|---|---|---|
| ROpt down | no | create session, → Pending |
| everything else | no | ignored |

#### Pending

ROpt is down. A session has been created and is capturing audio, but the
UI doesn't know yet whether this is a real activation.

> h[ui.pending-to-ptt]
> If ROpt is still held after ~300ms with no other keys pressed: transition
> to PushToTalk.

> h[ui.pending-to-locked]
> If ROpt is released in under ~300ms with no other keys pressed: transition
> to Locked.

> h[ui.pending-abort]
> If any other key is pressed while in Pending: abort the session,
> transition to Idle. The other key MUST be passed through to the app.

> h[ui.pending-ime-confirmation]
> Pending is also the IME-start handshake window. Bee may capture audio while
> pending, but recording is not considered confirmed until the IME start
> acknowledgement is received for the current session.

> h[ui.pending-ime-timeout]
> If IME start acknowledgement is not received within the configured startup
> timeout budget, Bee MUST abort the session, transition to Idle, and play the
> start-failure sound. It MUST NOT continue dictating in an unconfirmed state.

| Event | Swallowed? | Effect |
|---|---|---|
| ~300ms timer fires | — | → PushToTalk |
| ROpt up (clean) | no | → Locked |
| P (no other modifiers) | **yes** | paste last history entry, abort session, → Idle |
| any other key down | no | abort session, → Idle |

#### PushToTalk

ROpt is held. Recording is confirmed.

> h[ui.ptt-commit]
> On ROpt up while in PushToTalk: commit the session, transition to Idle.

> h[ui.ptt-to-locked]
> On RCmd down while in PushToTalk: transition to Locked. The next ROpt up
> MUST NOT trigger a commit.

> h[ui.ptt-cancel]
> On Escape while in PushToTalk: cancel the session, transition to Idle.

| Event | Swallowed? | Effect |
|---|---|---|
| ROpt up | **yes** | commit(submit: false), → Idle |
| RCmd down | **yes** | → Locked |
| Escape | **yes** | cancel, → Idle |
| max duration (300s) | — | commit(submit: false), → Idle |
| all other keys | no | passthrough |

#### Locked

Hands-free recording. ROpt is not held. The user may switch apps freely.

> h[ui.locked-option-down]
> On ROpt down while in Locked: transition to LockedOptionHeld.

> h[ui.locked-enter]
> On Enter while in Locked: commit the session with submit, transition to
> Idle. The Enter MUST be swallowed — the session will simulate Return
> after text insertion.

> h[ui.locked-esc-passthrough]
> On Escape while in Locked: passthrough to the app. Not intercepted.

| Event | Swallowed? | Effect |
|---|---|---|
| ROpt down | **yes** | → LockedOptionHeld |
| Enter | **yes** | commit(submit: true), → Idle |
| Escape | no | passthrough |
| max duration (300s) | — | commit(submit: false), → Idle |
| all other keys | no | passthrough |

> h[ui.locked-app-switch]
> In Locked and LockedOptionHeld states, the user may switch to other
> applications. Recording continues. The overlay indicates tethering.

#### LockedOptionHeld

In locked mode, ROpt is currently being held.

> h[ui.locked-option-held-commit]
> On ROpt up while in LockedOptionHeld: commit the session, transition to
> Idle.

> h[ui.locked-option-held-cancel]
> On Escape while in LockedOptionHeld: cancel the session, transition to
> Idle. The Escape MUST be swallowed.

| Event | Swallowed? | Effect |
|---|---|---|
| ROpt up | **yes** | commit(submit: false), → Idle |
| Escape | **yes** | cancel, → Idle |
| max duration (300s) | — | commit(submit: false), → Idle |
| all other keys | no | passthrough |

### Max duration

> h[ui.max-duration]
> After 300s of recording, the UI layer commits the current session
> regardless of current state.

### Status indicator

A small status indicator is shown at the IME cursor position during
recording. It does not display the transcript (that's the IME marked
text) — it shows the session's state so the user knows what's happening.

> h[ui.indicator.position]
> The status indicator is positioned at the IME cursor location.

> h[ui.indicator.states]
> The indicator shows the current state: recording (audio is being
> captured and streamed), finalizing (waiting for the final transcript
> from a previous session), or listening (a new session has started and
> is capturing audio while the previous one finalizes).

> h[ui.indicator.tether]
> In locked mode, if the user switches to a different app, the indicator
> shows that recording is tethered to the original app.

## Session

A session is created by the UI layer on ROpt down. It has a unique
identifier and owns three internal layers: Capture, ASR, and IME. Each
layer has its own state machine.

### Session endings

> h[session.abort]
> On abort: all three layers are torn down immediately. No finalization,
> no history entry, no visible side effects. The session never existed
> from the user's perspective.

> h[session.cancel]
> On cancel: capture drains (delivers tail audio), ASR finalizes in the
> background, a history entry is created with the final transcript, but
> the IME clears its marked text without committing. No text is inserted.

> h[session.commit]
> On commit: capture drains (delivers tail audio), ASR finalizes, IME
> commits the final transcript (inserts text). If submit is true, a
> Return keystroke is simulated after insertion.

### Capture layer

The capture layer taps into the shared audio engine to collect audio for
this session.

> h[capture.start]
> When the session is created, the capture layer copies the audio engine's
> pre-buffer and begins accumulating incoming audio. This preserves audio
> from just before the hotkey was pressed.

> h[capture.buffering]
> While buffering, incoming audio samples are appended to the session's
> capture buffer. RMS levels are computed per buffer for the status
> indicator.

> h[capture.drain]
> On commit or cancel, the capture layer enters draining mode. It monitors
> incoming audio for silence (VAD). Drain completes when RMS stays below
> the silence threshold for the required duration, or the drain timeout
> is reached.

> h[capture.drain-delivers]
> When drain completes, all captured samples are delivered to the ASR
> layer. The audio engine remains warm.

> h[capture.abort-discard]
> On abort, the capture layer discards all buffered audio immediately.
> No drain, no delivery.

### ASR layer

The ASR layer processes audio from the capture layer and produces text.

> h[asr.streaming]
> During capture, audio samples are fed to the ASR session incrementally.
> Each feed may return a streaming update with the current transcript.

#### Checkpointing

Without checkpointing, the ASR encoder must re-process an ever-growing
audio buffer on every inference pass, eventually falling behind real-time.
Checkpointing solves this by locking in stable text and resetting the
internal audio/encoder state.

When a checkpoint fires, the checkpointed text is permanent (it will not
change), the internal streaming state is fully reset (audio accumulator,
encoder cache, decoder state), and a new sub-session starts with only the
remaining text as context.

The IME layer does NOT commit on checkpoint — all text (including
checkpointed portions) remains as marked text. Only session commit/cancel
affects the IME.

> h[asr.checkpoint.clause-boundary]
> Checkpoints MUST only occur at clause boundaries: commas, periods,
> exclamation marks, question marks, and equivalent punctuation. A
> checkpoint MUST NOT split in the middle of a clause.

> h[asr.checkpoint.stability]
> A clause boundary is eligible for checkpoint only after the text up to
> that boundary has been identical across multiple consecutive inference
> passes (stability threshold). This prevents checkpointing text that the
> model is still revising.

> h[asr.checkpoint.minimum-length]
> A checkpoint candidate must contain a minimum number of words and
> characters to avoid checkpointing trivial fragments.

> h[asr.checkpoint.no-time-based]
> There MUST NOT be a time-based forced rotation. Checkpoints are
> triggered only by stability at clause boundaries. Time-based rotation
> causes mid-sentence cuts and inserts spurious punctuation.

> h[asr.checkpoint.reset]
> When a checkpoint fires: the checkpointed text moves to the permanent
> transcript, the remainder becomes the new pending prefix, and the
> internal streaming state (audio accumulator, encoder cache, decoder
> state, token IDs) is fully reset. The new sub-session receives the
> trailing text (up to 200 characters) as initial context.

> h[asr.finalize]
> After capture delivers its final samples, the ASR layer runs
> finalization to produce the definitive transcript.

> h[asr.finalize-background]
> Finalization runs in the background. It MUST NOT block the UI layer
> or prevent new sessions from being created.

> h[asr.tail-audio]
> Samples captured after the last streaming feed but before drain
> completion (tail audio) MUST be included in the finalization feed.
> Dropping them causes truncated transcripts.

> h[asr.fallback]
> If finalization produces an empty or suspiciously short result while
> meaningful audio exists, a full-audio batch transcription runs as a
> fallback. The longer result wins.

> h[asr.chunk-size]
> The streaming chunk size (how often audio is fed to the ASR) is
> configurable. Smaller chunks give faster partial updates but use more
> CPU. This is exposed as a user-facing setting.

> h[asr.streaming-signals]
> The ASR layer detects special voice commands in the transcript:
> "Over" triggers a commit, "Over and out" triggers a commit and stops
> recording. These allow hands-free control during locked mode.

### IME layer

#### macOS InputMethodKit background

macOS input methods run as separate processes. The framework provides
`IMKServer` (one per input method process) and `IMKInputController` (one
per input session). The system manages controller lifecycles automatically.

**Controller lifecycle.** The OS creates one `IMKInputController` per
"input session." In practice, this means roughly one controller per
application — not per window, not per text field. Controllers are created
lazily on first activation and persist in memory for the lifetime of the
app's connection. They are never explicitly destroyed by the input method.

**activateServer(sender).** Called by the OS when the input method becomes
active for a particular client. This happens when: (a) the user switches
to this input source via `TISSelectInputSource`, (b) a text field gains
focus while this input source is already selected, or (c) the user switches
between windows that use different text fields. The `sender` parameter and
`self.client()` both refer to the `IMKTextInput` proxy for the text field
that now has focus.

**deactivateServer(sender).** Called when the input method is deactivated
for a client. This happens when: (a) focus moves to a different text field,
(b) the user switches to a different input source, or (c) the app loses
focus. If marked text exists at deactivation time, the client typically
auto-commits it (standard macOS behavior for marked text on focus loss).

**Important timing characteristics:**
- `TISSelectInputSource` returns immediately but `activateServer` fires
  asynchronously — there is a 15–150ms gap between the two.
- Distributed notifications also arrive asynchronously on the main queue.
- When switching between windows of the **same app**, the OS may reuse
  the existing controller without calling `deactivateServer` +
  `activateServer`. The controller's `client()` silently changes to point
  to the new window's text field.
- When switching between **different apps**, the old controller gets
  `deactivateServer` and the new app's controller gets `activateServer`.
- Multiple controllers can be live simultaneously (one per app that has
  used the input method). Only one is "active" at a time.

**Race conditions.** If `TISSelectInputSource` is called while focus is
shifting between windows/apps (e.g., the user pressed the hotkey while a
window transition was in progress), the OS may route `activateServer` to
a different app/window than intended. The input method process has no
mechanism to specify "activate for app X" — it can only react to what
the OS gives it.

#### Bee's IME design

Bee uses a custom input method (beeInput) for text insertion. The bee app
and beeInput are separate processes. The control-plane transport for start/
ack lifecycle is XPC. (Distributed notifications may still be used for legacy
or transitional event wiring, but authoritative session-start coordination is
XPC request/reply.)

The IME layer manages text insertion via the beeInput InputMethodKit IME.

> h[ime.activate]
> When the session is created, Bee saves the current input source and
> calls `TISSelectInputSource` to switch to the beeInput IME. Because
> `activateServer` fires asynchronously (15–150ms later), the initial
> marked text (🐝 cursor) is queued and delivered when the controller
> activates.

> h[ime.start-two-phase]
> IME startup is a two-phase handshake:
> 1) app prepares session on the broker (session ID only — no target PID),
> 2) app selects beeInput source,
> 3) IME claims the prepared session and replies with start acknowledgement.
> The session is not active until step (3) succeeds for the current session.
> No PID matching is performed — the IME dictates into whatever app is
> focused. Target tracking for app-switch parking is handled separately
> by the app layer.

> h[ime.start-no-early-effects]
> Before IME start acknowledgement:
> - no recording-confirmed sound,
> - no transition from Pending to PushToTalk/Locked,
> - no user-visible IME rendering beyond queued placeholder behavior.

> h[ime.per-app-controller]
> InputMethodKit creates one controller per application, not per window.
> When the user switches between windows of the same app, the OS may
> reuse the existing controller without calling deactivateServer +
> activateServer. The controller's `client()` may still point to the
> previous window's text field. This means that if the user has two
> windows of the same app open, text may be delivered to the wrong
> window even when the correct window has had focus for a long time.
> This is a fundamental limitation of InputMethodKit's per-app model.

> h[ime.marked-text]
> During recording, ASR streaming updates are sent to the IME as marked
> text. The text appears underlined in the input field to indicate it is
> provisional. The full transcript (including checkpointed portions)
> remains as marked text — checkpoints do not cause IME commits.

> h[ime.morph-animation]
> When the transcript changes, the displayed text morphs into the new
> text via randomized in-place character changes and appends (Matrix
> style), rather than snapping or using delete-then-insert. Each step
> randomly picks between: fixing a wrong character in-place, appending
> the next correct character, or trimming excess characters. The
> animation is interrupted immediately when a new partial arrives or
> when the session ends (.done event skips animation entirely).

> h[ime.commit]
> On session commit, the marked text is replaced with the final transcript
> via insertText, making it permanent. A trailing space is appended.

> h[ime.clear-on-cancel]
> On session cancel, the marked text is cleared without committing. No
> text is inserted.

> h[ime.deactivate]
> After commit or cancel, the beeInput IME is deactivated and the
> previous input source is restored.

> h[ime.submit]
> When commit has submit: true, a Return keystroke is simulated after a
> short delay following IME deactivation.

> h[ime.abort-teardown]
> On session abort, the IME is deactivated immediately. No commit, no
> clear — the session was never visible to the user.

### Focus loss and parking (IME)

> h[ime.focus-loss-autocommit]
> When the target field loses focus during recording, macOS auto-commits
> the current marked text (standard IME behavior). The IME controller
> saves this text as an auto-committed prefix.

> h[ime.prefix-dedup]
> When streaming resumes after focus return, incoming text is matched
> against the auto-committed prefix. If the text starts with the prefix,
> the prefix is stripped to avoid duplication. If the text has diverged,
> the prefix is discarded.

> h[ime.parking]
> In locked mode, when the user switches away from the target app, the
> IME session is parked: the controller stays alive and isDictating
> remains true so the session resumes on return.

> h[ime.key-intercept]
> While isDictating is true, the IME intercepts Enter (triggers submit)
> and Escape (triggers cancel) via XPC through the broker back to the
> main app.

> h[ime.communication]
> All communication between the main app and the IME goes through the
> broker process via XPC:
> - Session lifecycle: prepare, claim, attach, clear.
> - App → IME text commands: `setMarkedText`, `commitText`, `cancelInput`,
>   `stopDictating`.
> - IME → App events: `imeSessionStarted`, `imeSubmit`, `imeCancel`,
>   `imeUserTyped`, `imeContextLost`.
> The broker also notifies the IME of newly prepared sessions so it can
> claim directly if it already has an active controller.

## Hotkey

> h[hotkey.right-option]
> The hotkey is the Right Option key, detected via CGEvent tap. Key-down
> and key-up events on ROpt drive all UI layer transitions.

## Media

> h[media.pause-on-record]
> When media pause is enabled, Bee detects active audio output from other
> apps and sends a pause command before recording starts.

> h[media.resume-after-record]
> After recording ends, Bee resumes media playback only if it was the one
> that paused it.

## Language Detection

Language is always auto-detected. There are no overrides.

> h[lang.detect-from-ax]
> At session creation, Bee walks the AX element tree of the focused
> window (up to 2000 elements), collects text, takes the last 500
> characters, and runs NLLanguageRecognizer.

> h[lang.confidence-threshold]
> English requires 50% confidence. Non-English languages require 80%
> confidence. Below the threshold, no language hint is passed to the model.

> h[lang.lock-during-streaming]
> If no language was determined at session start, the first streaming
> update that reports a detected language locks the session to that
> language for the remainder of the recording.

## History

> h[history.paste-last]
> ROpt then P (no other modifiers, while in Pending) pastes the last
> history entry into the current app. The P is swallowed and the session
> is aborted.

## IME Safety

The beeInput IME is a separate process. Bee switches to it at session
start and away from it at session end. But things can go wrong: the app
can crash, the user can quit, or the IME can end up selected with no
active session. These cases must be handled gracefully.

> h[ime.safety.restore-on-quit]
> When Bee quits (cleanly), it MUST restore the previous input source
> if beeInput is currently active.

> h[ime.safety.restore-on-crash]
> If Bee crashes while beeInput is the active input source, the IME
> itself MUST detect that no session is active and switch away to the
> previous input source.

> h[ime.safety.no-session-typing]
> If the user starts typing while beeInput is the active input source
> but no session is active, the IME MUST immediately switch to the
> previous input source and let the keystrokes pass through.

## Sounds

> h[sounds.recording-started]
> A sound plays when recording is confirmed (transition from Pending to
> PushToTalk or Locked). No sound plays if the activation is aborted.

> h[sounds.start-failure]
> If IME start association fails or times out, Bee plays a distinct
> start-failure sound and returns to Idle.

> h[sounds.commit]
> A sound plays when a session commit completes (text has been inserted).

> h[sounds.cancel]
> A distinct sound plays when a session is cancelled.

## Menu Bar

Bee is a menu bar app. The menu bar icon and popover are the only
persistent UI surface.

> h[menubar.status]
> The menu bar shows the current status: ready, recording, or finalizing.

> h[menubar.history]
> The menu bar popover shows recent transcription history. Clicking an
> entry pastes it into the current app.

> h[menubar.model]
> The menu bar popover allows selecting, downloading, and deleting ASR
> models.

> h[menubar.input-device]
> The menu bar popover shows the current input device and allows
> selecting a different one.

> h[menubar.warm-toggle]
> The menu bar popover shows the warm/cold setting for the current
> device and allows toggling it.

> h[menubar.run-on-startup]
> The menu bar popover allows toggling run-on-startup.

> h[menubar.pause-media]
> The menu bar popover allows toggling pause-media-while-dictating.

> h[menubar.quit]
> The menu bar popover has a quit button. Quitting MUST restore the
> previous input source (see `ime.safety.restore-on-quit`).

## Forensics

> h[forensics.html-dump]
> When enabled, each session generates an HTML timeline dump with
> embedded audio players, event traces (Swift and Rust), and timing
> data. Dumps are written to a local directory.

> h[forensics.session-retention]
> Old forensics sessions are automatically cleaned up. Only the most
> recent N sessions are retained.

## Coordination

> h[coord.drain-before-finalize]
> Capture drain MUST complete and deliver all samples before ASR
> finalization begins.

> h[coord.reactivate-locked-app]
> On commit in locked mode, if the user is in a different app, Bee
> brings the original app to the front and waits before inserting text.

## Deprecated Features (non-normative)

The following features existed in the old app and are explicitly removed.
This section is not normative — it exists to document what we're no longer
doing and why.

**Paste and AX insertion strategies.** The old app had three text insertion
strategies (paste via clipboard + Cmd+V, direct AX text field manipulation,
and IME). Each could be selected per-app. We now use IME exclusively. The
paste strategy had clipboard corruption issues and timing-sensitive Cmd+V
simulation. AX had fragile element capture that broke across app updates.
IME is the only strategy that works with the text input system rather than
against it.

**Per-app settings.** The old app stored language, insertion strategy,
auto-submit, and vocabulary prompt per application bundle ID. All of these
are removed. Language is always auto-detected. There is no insertion
strategy choice. Auto-submit is replaced by explicit Enter in locked mode.
Vocabulary prompts are removed.

**Configurable hotkey.** The old app supported arbitrary modifier+key
combinations as the hotkey, with a capture UI in the menu bar. The hotkey
is now always Right Option, hardcoded.

**Shift submit-arm.** The old app let users press Shift during recording to
toggle a submit-arm flag. Submit is now only triggered by Enter in locked
mode.

**Floating overlay panel.** The old app had a large floating overlay
(540×210) showing the live transcript, a 6-band spectrum analyzer,
contextual hint text, and connector lines to the focused element. This
is replaced by a small status indicator at the IME cursor — the transcript
itself is visible as IME marked text in the actual input field.

**Complex overlay anchoring.** The old overlay had a 4-level fallback
chain (caret → element frame → window frame → mouse), editor pane
detection heuristics, focus highlight panels, connector line drawing, and
a 60ms follow loop. The new status indicator simply sits at the IME cursor.

**Accidental double-press detection.** The old app had a 0.5s ignore window
after a very short press to prevent rapid re-triggers. This is superseded
by the Pending state's ~300ms mode-determination window, which naturally
absorbs accidental taps.

**TranscriptionLogger.** The old app logged every transcription to a JSONL
file for training data collection. This is removed from the app spec (it
may live elsewhere in the pipeline).

**Spectrum visualization.** The old overlay had a real-time 6-band FFT
spectrum analyzer. This is removed along with the floating overlay.

**Unused legacy XPC stubs.** The old app carried partial XPC scaffolding
that was not the active startup contract. The current spec makes XPC the
authoritative control-plane transport for IME session start/ack.
