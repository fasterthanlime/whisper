# IME Facts (macOS InputMethodKit)

These are empirically observed behaviors intended to be stable enough to build a working mental model of macOS IMEs.

**Scope** (current evidence):
- Experiments were run on 2026-04-10 using the test apps listed in [science/README.md](./README.md).
- Evidence lives in `~/bearcove/bee-experiments/experiments/` and `~/bearcove/bee-experiments/logs/`.
- Anything described as “normal macOS behavior” below means “observed with Apple’s built-in Japanese IME in these apps”, not “proven for all apps / all macOS versions”.

**Terminology**:
- “marked text” is the client-side composition UI (underline/highlight) managed through the `NSTextInputClient` contract.
- “proxy” refers to what bee sees via InputMethodKit (e.g. `markedRange`, `selectedRange`, `attributedSubstring(from:)`).
- Logs print `NSNotFound` as `{9223372036854775807, 9223372036854775807}`; the experiments write `{∅}` for readability.

## F-001-marked-text-persists-on-switch

Switching apps does **not** automatically clear existing marked text in the previously-focused app. In the tested apps, the marked text remained visible after Cmd+Tab in all cases; in 4/5 apps it remained visibly marked (underline/highlight), while Codex committed it (see F-002/F-005).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-001-heartbeat-app-switch.md` (+ per-app logs)

## F-002-codex-commits-on-deactivate

**App-specific**: Codex (`com.openai.codex`, Electron/Chromium) commits marked text on focus loss (deactivate): the composing text becomes regular text (no marked UI).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-001-heartbeat-app-switch.md`

## F-003-super-cancel-composition-insufficient

Calling `cancelComposition()` from `deactivateServer` is not sufficient to ensure the client clears the visible marked text on app switch (in the tested apps).

This is intentionally scoped to the behavior we need to assume when implementing a robust IME: “don’t rely on deactivate-time cancel to clean up client UI”.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-001-heartbeat-app-switch.md`

## F-004-marked-text-persists-is-normal

In the tested apps, Apple’s built-in Japanese IME also leaves marked text visible across app switches. This strongly suggests that “marked text persisting across Cmd+Tab” is not a bee-specific bug.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-002-japanese-ime-app-switch.md`

## F-005-codex-commits-all-imes

**App-specific**: Codex commits marked text on deactivate regardless of which IME is active (confirmed with bee and the built-in Japanese IME).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-001-heartbeat-app-switch.md`, `~/bearcove/bee-experiments/experiments/E-002-japanese-ime-app-switch.md`

## F-006-marked-range-survives-round-trip (ACTIVE COMPOSITION ONLY)

While a composition is still “live” (the IME session stays active), switching away and back can yield a non-`NSNotFound` `markedRange` again in all tested apps.

Important nuance:
- This does **not** mean the proxy state is trustworthy (Codex can report a `markedRange` that refers to unrelated document text; see F-009).
- This does **not** hold once the IME session is ended before returning (see F-018).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-003-reactivation-state.md`

## F-007-full-text-probe-blocked

`attributedSubstring(from:)` for a large range returned `nil` in all tested apps. Treat bulk text reads through the proxy as unavailable (design for zero context or small local windows only).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-003-reactivation-state.md`

## F-008-selected-range-after-marked

On reactivation, `selectedRange` position relative to `markedRange` varies by app (some report the caret at the start of the marked region, others at the end).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-003-reactivation-state.md`

## F-009-codex-marked-range-lies

**App-specific**: In Codex, the proxy-reported `markedRange`/`markedText` on reactivation can be stale or unrelated to the IME’s actual marked text. In the E-003 setup, Codex reported `markedText=\"ba\"` at `markedRange={3,2}` while the IME’s emoji had been visually committed.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-003-reactivation-state.md`, `~/bearcove/bee-experiments/logs/E-003-reactivation-state-codex.txt`

## F-010-zed-double-activate

**App-specific**: Zed can trigger multiple `activateServer` calls when returning to an app, with the proxy state changing between calls (“two-phase reactivation”).

**Evidence**: `~/bearcove/bee-experiments/experiments/E-003-reactivation-state.md`

## F-013-activate-before-deactivate-ordering

On app switch, `activateServer` for the incoming app can run **before** `deactivateServer` for the outgoing app. IME code must not assume “deactivate happens first”.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-004-cleanup-on-reactivation.md`, `~/bearcove/bee-experiments/logs/E-004-cleanup-on-reactivation-codex.txt`

## F-014-first-activateServer-may-see-no-markedRange

On reactivation, the first `activateServer` call can have `markedRange == NSNotFound` / unavailable. Treat the first activation callback as “proxy may not be ready”.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-005-cleanup-with-stored-bundle.md`

## F-015-markedRange-can-appear-on-a-later-activateServer

A later `activateServer` call shortly after the first can expose a valid `markedRange`. If you need to read proxy state, you may need to wait for a subsequent activation callback where the proxy is “connected enough” to report non-`NSNotFound` values.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-005-cleanup-with-stored-bundle.md`

## F-016-setMarkedText-empty-does-not-clear

`setMarkedText(\"\")` during activation/reactivation is not a reliable “cleanup” mechanism: the proxy can report the range as cleared, while the client UI remains unchanged (Messages) or you get duplicated content (Codex).

**Note**: This was observed while `TISDeselectInputSource` was still being called. With TIS deselect removed (F-019), `setMarkedText("")` successfully cleared marked text in Messages (E-010). May need re-evaluation.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-006-deferred-cleanup.md`

## F-017-proxy-is-not-authoritative

The proxy’s post-action state (e.g. `markedRange={loc,0}` after `setMarkedText(\"\")`) is not a reliable indicator of what the client actually shows. Treat proxy reads as hints, not truth.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-006-deferred-cleanup.md`

## F-018-markedRange-may-be-unavailable-after-session-stop

If the IME session is ended before returning to the app (stop bee session while the client still shows leftover marked text), the proxy may report `markedRange == NSNotFound` on *all* subsequent `activateServer` calls. That makes `markedRange` unusable for detecting and cleaning up stale leftovers after deactivation.

**Note**: This was observed while `TISDeselectInputSource` was still being called. With TIS deselect removed (F-019), the proxy stays alive and markedRange may be available. Needs re-verification.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-008-cleanup-every-activate.md`

## F-019-tis-deselect-kills-proxy

Calling `TISDeselectInputSource` after ending a bee session destroys the proxy connection to the client app. Any subsequent `setMarkedText("")` (whether via `sender`, `self.client()`, or IPC) cannot reach the actual client — the text input channel is gone. Removing `TISDeselectInputSource` keeps the proxy alive, allowing cleanup calls to forward to the real client.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-010-no-tis-deselect.md`

## F-020-unconditional-clear-too-aggressive

An unconditional `setMarkedText("")` on every `activateServer` (when not dictating) clears marked text in apps that were never bee targets. The clear must be scoped to only fire when the IME knows it left stale bee marked text in a specific app.

**Evidence**: `~/bearcove/bee-experiments/experiments/E-010-no-tis-deselect.md`

## Superseded / Not Universal

Some earlier writeups were really “implementation postmortems” (bee’s own bookkeeping) rather than macOS/app behavior. Those should not be treated as universal facts:
- Former “cleanup missed Codex due to ordering” is an app-switch ordering fact (kept as F-013) plus an implementation bug (moved out).
- “cleanup works for normal apps” was not reproducible (superseded by F-016/F-018).
