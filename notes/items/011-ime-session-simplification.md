# IME Session Simplification

## Problem

The current `prepareSession` / `claimSession` / `imeAttach` dance exists to work around an older world where IME activation was unreliable and we had to survive duplicate triggers and inconsistent activation timing.

That is no longer the world we are in.

Today this protocol is mostly complexity:

- Dictation state is split across Bee, the IME bridge, and Vox RPC state.
- The app treats IME confirmation as if it were required to start dictation.
- A transient IME activation race can abort an otherwise healthy dictation session.
- We keep patching handoff bugs (`pendingSessionId`, stale controller behavior, reconnect races) instead of removing the handoff itself.

## Intended Model

Bee owns dictation. The IME is only a rendering and commit surface.

- Hotkey down starts dictation immediately in Bee.
- Bee records the target window / app context where dictation started.
- Start sound plays immediately when dictation starts, not after an IME round-trip.
- Bee owns the full transcript for the lifetime of dictation.
- The IME is attached opportunistically when the focused target matches the dictation target.
- If focus leaves the target, we stop using the IME and show the overlay instead.
- If focus returns to the target, we create a fresh IME-local session and replay the full current transcript into it.

This means the IME session is disposable. The dictation session is not.

## Surface Rules

There are two presentation surfaces:

1. IME
2. Overlay

Selection rules:

- If the focused target matches the dictation target and the IME is available, show the transcript in the IME.
- Otherwise, show the transcript in the overlay.
- Switching away from the target clears IME-local state.
- Switching back creates a new IME-local session and replays the transcript from Bee state.

## Commit Rules

On finish:

- If the IME is attached to the original target, commit through the IME.
- Otherwise, if focus is still in the original target and the IME is unavailable, use accessibility APIs to insert the final text.
- Otherwise, do not inject text blindly into a different target.

Accessibility insertion is a final-commit fallback only. It should not be part of the normal streaming path.

## Consequences

The following should stop being core protocol concepts:

- app-side pending dictation session handoff
- IME-side claim of a prepared session
- treating IME confirmation as a prerequisite for dictation start
- coupling start sound to IME readiness

The following should become core state instead:

- Bee-owned dictation session
- target window / app identity
- current transcript snapshot
- current presentation surface

## Migration Direction

1. Fix the immediate `pendingSessionId` regression so current behavior works again.
2. Move “dictation has started” feedback earlier so it does not wait on IME confirmation.
3. Make overlay the normal fallback when the target is not currently IME-attached.
4. Replace the prepare/claim protocol with fresh IME attachment plus transcript replay.
5. Keep reconnect / resumption policy inside Vox, not Bee.

## Design Constraint

Bee should not grow its own transport recovery layer.

- Session resumption
- fresh-session fallback
- reconnect behavior
- client invalidation after transport failure

All of that belongs in Vox.

Bee should only decide:

- what the current dictation target is
- what transcript to present
- which presentation surface to use
- how to commit final text
