# Bee IME Dictation Spec

Status: Draft (source of truth for rewrite)
Last updated: 2026-04-01

## 1. Goal

Define deterministic behavior for Bee dictation with IME integration so dictation:

- always targets the correct text client,
- never leaks to the wrong app,
- supports park/resume across app switches,
- and hands control back to manual typing immediately.

This document is product behavior, not implementation detail.

## 2. Core Principles

1. Target ownership is strict.
2. Wrong-target insertion is a hard failure.
3. Session lifecycle is explicit and state-driven.
4. Parking pauses IME rendering only (not capture/transcription).
5. Manual typing has priority over ongoing dictation.

## 3. Definitions

- Target client: the exact text input client that would receive a normal key when dictation is triggered.
- Active: session is running and IME renders marked text into target client.
- Parked: session is running, IME does not render into any text field.
- Resume: return from parked to active on the original target.
- Immediate commit: commit current text snapshot without finalization.

## 4. Functional Requirements

### 4.1 Start / Targeting

1. Triggering dictation must bind to the exact current target client.
2. Dictation text must appear only in that target client while active.
3. If target binding cannot be established, session must not start rendering.

### 4.2 Wrong-target Safety

4. Dictation text must never be inserted/marked in any non-target app/client.
5. Any ambiguous routing condition must fail safe (no write).

### 4.3 Park / Resume

6. If user switches away from target app/client, session transitions to parked.
7. While parked, IME rendering to text fields is disabled.
8. While parked, live dictation text must remain visible in an on-screen overlay.
9. Overlay must include:
   - target app icon,
   - target app name,
   - current dictated text.
10. Returning to target app/client resumes active rendering into the same target.

### 4.4 Manual Typing Priority

11. If user types manually in target context (letters, space, backspace, etc.), Bee must:
   - immediately commit current text snapshot (no finalization),
   - stop dictation updates,
   - hand control back to user typing.
12. Immediate commit must be fast and deterministic, including if animation is mid-update.
13. After immediate commit, Bee must switch away from `beeInput` right away.

### 4.5 End Conditions

14. Enter/Escape behavior must be deterministic (submit/cancel).
15. If target disappears while parked (window/tab/client gone), session must cancel.
16. Quitting Bee must always leave keyboard input on a non-`beeInput` source.
17. Manual activation of `beeInput` outside active session is invalid and should auto-exit/switch away.

### 4.6 Capture / Transcription

18. Parking must pause IME rendering only.
19. Audio capture/transcription may continue while parked.

## 5. Session State Machine

States:

- `idle`
- `active`
- `parked`
- terminal: `committed`, `cancelled`, `aborted`

Transitions:

1. `idle -> active`: start and target bound.
2. `active -> parked`: user leaves target app/client.
3. `parked -> active`: user returns to target app/client.
4. `active -> committed`: normal commit/submit or manual typing immediate commit.
5. `active -> cancelled`: explicit cancel.
6. `parked -> cancelled`: target disappears or explicit cancel.
7. `active|parked -> aborted`: unrecoverable internal failure.

Constraints:

- No writes to text fields in `parked`.
- No transition from terminal states.

## 6. Immediate Commit Semantics

On manual typing:

1. Take current displayed text snapshot.
2. Stop further dictation rendering.
3. Insert snapshot as final committed text in target.
4. Switch away from `beeInput`.
5. Do not run ASR finalization pass.

Notes:

- Current product behavior may append trailing space after commit; if changed, update this spec.

## 7. Non-Goals

1. This spec does not define transport/protocol mechanics (notifications vs XPC, etc.).
2. This spec does not define animation style, only behavioral outcomes.
3. This spec does not define visual design details of overlay.

## 8. Observability Requirements

Each significant event must log:

- session ID,
- target identity,
- active/parked state,
- reason for transition,
- write/drop decision,
- commit mode (normal vs immediate).

## 9. Acceptance Criteria

1. Start dictation in app A, remain in app A: continuous correct IME updates.
2. Start in app A, switch to app B: no IME text appears in B; overlay appears with app A metadata.
3. Return to app A: IME updates resume in A without new session.
4. Type a key in app A during active dictation: immediate commit of current snapshot; dictation stops; manual typing continues.
5. While parked, close/destroy target context: session cancels.
6. Quit Bee during active/parked session: input source is not `beeInput` afterward.
7. Manually select `beeInput` when idle: it auto-switches away and does not trap user.
