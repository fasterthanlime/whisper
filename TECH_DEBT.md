# Tech Debt

## IME activation timeout started from hotkey down, not TIS selection

**Location:** `AppState.swift` `startIMEAckTimeoutIfNeeded`

The 2s timeout is measured from when `startIMEAckTimeoutIfNeeded` is called (hotkey down path), not from when `TISSelectInputSource` actually succeeds. Session setup takes ~180ms, so the effective wait after TIS is shorter than intended.

**Proper fix:** Start the timeout from when `TISSelectInputSource` succeeds. Requires `activate()` in `BeeInputClient` to signal back to `AppState` so the timeout can start from the right moment.

## Sleeps used as proxy for acknowledgement in finishIME

**Location:** `Session.swift` `finishIME`

After `commitText`, we sleep 50ms before calling `deactivate()`, hoping the commit has propagated. Same before `simulateReturn()`. These sleeps are fragile:
- On the Session actor, `await Task.sleep` makes the actor re-entrant — other methods can interleave and mutate state during the sleep.
- If the task gets cancelled, the sleep throws `CancellationError`.

**Proper fix:** Replace sleeps with explicit acknowledgement. `commitText` should be an async operation that resolves when the IME confirms the text was inserted, then deactivate. The broker already has the plumbing for this — add a reply to the commit XPC call.

## Actor reentrancy across all awaits in Session

**Location:** `Session` actor, all methods with `await`

Every `await` in the Session actor (sleeps, MainActor.run, XPC calls) allows other methods to enter and mutate state. This means:
- `abort()` can fire during `finishIME()`'s sleep
- `routeDidBecomeActive()` can fire during `start()`'s `await activate()`
- State checks before an `await` may be stale after it

**Proper fix:** Critical state transitions should be atomic (no awaits between check and mutation). For multi-step operations, use a serial queue of operations or a state machine that rejects invalid transitions rather than relying on timing.

## Parked IME sessions do not actually resume after focus returns

**Location:** Locked-mode focus loss / parking flow, around IME park-resume handling

The spec says locked sessions are parked on app switch and should resume when the
target app regains focus. In practice, if dictation blurs to another window/app,
the IME session is lost, and when focus returns Bee does not meaningfully resume
the parked session. The current behavior is not useful: recording may continue,
but the dictation session does not reattach and resume marked-text updates as
expected.

This regresses the intended locked-mode parking behavior described in
`spec/state.md` under `h[ime.parking]`, `h[ime.focus-loss-autocommit]`, and
`h[ime.prefix-dedup]`.

**Proper fix:** Treat focus return as a real resume path, not just a bookkeeping
event. On reactivation, re-establish the IME session/controller, restore any
auto-committed prefix bookkeeping, and resume streaming updates into the target
field so parked locked-mode dictation behaves like a continuous session again.

## Enter during push-to-talk finalization is not converted into commit-plus-submit

**Location:** Push-to-talk release path and hotkey/key interception during session finalization

After releasing push-to-talk, Bee can spend roughly 0.5s finalizing capture/ASR
before the commit is ready. If the user presses Enter during that window, Bee
should swallow the key, upgrade the in-flight `commit(submit: false)` into
"pending commit + submit", wait for finalization to finish, and then inject
Return after text insertion.

That is the correct complement to the existing hands-free submit flow: users
should be able to hold the hotkey, dictate, release it, and press Enter instead
of having to re-enter locked mode or press the hotkey again.

**Proper fix:** Add an explicit "finalizing" submit-latch state for push-to-talk
commits. While finalization is in progress, intercept Enter, record a pending
submit intent, and convert the eventual IME commit into submit behavior once the
final transcript is inserted. Do not pass the original Enter through to the app.

## Finishing dictation in another window needs a notification + clipboard fallback

**Location:** Locked-mode app/window switching, especially commit while focus is no longer on the original text field

There should be an explicit recovery path for the case where a dictation starts
in one window/text field, but by commit time the user is in another window and
Bee can no longer reliably deliver the final text back into the intended field.
Instead of silently doing nothing useful or inserting into the wrong place, Bee
should preserve the final transcript, copy it to the clipboard, and show a
clickable notification telling the user the dictation is ready to paste.

This is closely related to the app/window focus hazards described in
`spec/state.md` around IME controller reuse, cross-window ambiguity, and parked
session behavior, but the fallback UX itself is not currently written down as a
concrete requirement.

**Proper fix:** When Bee cannot safely resume or commit back into the intended
field, finalize the transcript anyway, put it on the clipboard, and present a
notification with clear affordance text so the user can return and paste it.
That fallback should be deterministic and user-visible, not a silent failure.
