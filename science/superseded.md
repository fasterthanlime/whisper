# Superseded / Implementation-Specific Findings

This file exists so `science/facts.md` can stay focused on behaviors that are plausibly stable across macOS and client apps.

If you find yourself needing one of these items again, the right fix is usually to convert it into:
- a real “universal-ish” fact about ordering/proxy behavior (belongs in `science/facts.md`), or
- a concrete bee bug / design constraint (belongs in code comments or an issue), not a “fact”.

## S-001-pendingCleanup-wrong-bundle-id

In E-004, cleanup did not fire for Codex because bee recorded the wrong bundle ID for “the app that lost focus”.

This was not a macOS behavior: it was a bug in how bee identified the outgoing app in `deactivateServer`, interacting with the real ordering fact that `activateServer` for the incoming app can run before `deactivateServer` for the outgoing app (see F-013).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-004-cleanup-on-reactivation.md`

## S-002-cleanup-cleared-once-then-never-again

In E-004, a `setMarkedText("")` cleanup appeared to clear Messages once; later experiments (E-006, E-007, E-008) contradicted this as a reliable approach.

Treat this as “a fluke / timing artifact”, not as evidence that the mechanism is dependable.

**Evidence**:
- `~/bearcove/bee-experiments/experiments/E-004-cleanup-on-reactivation.md`
- `~/bearcove/bee-experiments/experiments/E-006-deferred-cleanup.md`
- `~/bearcove/bee-experiments/experiments/E-007-isolated-cleanup.md`
- `~/bearcove/bee-experiments/experiments/E-008-cleanup-every-activate.md`

