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
