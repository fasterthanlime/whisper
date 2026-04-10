# Established Facts

## F-001-marked-text-persists-on-switch

With stripped-down overrides (only init, activateServer, deactivateServer) + `super.cancelComposition()` in deactivateServer, marked text is NOT cleared on app switch in any of the five test apps. It persists as marked text in 4/5 apps.

**Evidence**: [E-001-heartbeat-app-switch](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-001-heartbeat-app-switch.md)
**Commit**: `c053ce6`

## F-002-codex-commits-on-deactivate

Codex (Electron/Chromium) commits marked text as real text on deactivate (no underline — the emoji becomes permanent text), while all other tested apps preserve it as marked text (with underline/highlight).

**Evidence**: [E-001-heartbeat-app-switch](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-001-heartbeat-app-switch.md)
**Commit**: `c053ce6`

## F-003-super-cancel-composition-insufficient

`super.cancelComposition()` called inside `deactivateServer` does not trigger IMK's internal cleanup that clears marked text on the client.

**Evidence**: [E-001-heartbeat-app-switch](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-001-heartbeat-app-switch.md)
**Commit**: `c053ce6`

## F-004-marked-text-persists-is-normal

Marked text persisting on app switch is **normal macOS behavior**. The built-in Japanese IME exhibits the same behavior as bee — marked text stays in all tested apps. This is not a bug in bee or in our IME implementation.

**Evidence**: [E-002-japanese-ime-app-switch](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-002-japanese-ime-app-switch.md)

## F-005-codex-commits-all-imes

Codex (Electron/Chromium) instantly commits marked text on deactivate regardless of which IME is active. This is an Electron bug, not an IME bug. Confirmed with both bee and the built-in Japanese IME.

**Evidence**: [E-001-heartbeat-app-switch](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-001-heartbeat-app-switch.md), [E-002-japanese-ime-app-switch](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-002-japanese-ime-app-switch.md)

## F-006-marked-range-survives-round-trip

On reactivation (Cmd+Tab back), all 5 apps report a valid `markedRange` via the IMK proxy — even Codex, which visually committed the text. The markedRange location matches where bee started composing. The IMK proxy remembers the marked range across deactivate/reactivate.

**Evidence**: [E-003-reactivation-state](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-003-reactivation-state.md)

## F-007-full-text-probe-blocked

`attributedSubstring(from:)` with a large range returns nil in all apps. The IMK proxy does not allow bulk text reads.

**Evidence**: [E-003-reactivation-state](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-003-reactivation-state.md)

## F-008-selected-range-after-marked

On reactivation, `selectedRange` position varies by app: Notes/Zed/Codex report cursor at end of marked region (loc+len), while ime-spy/Messages report cursor at start. App-dependent behavior.

**Evidence**: [E-003-reactivation-state](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-003-reactivation-state.md)

## F-009-codex-marked-range-lies

Codex reports `markedRange={3, 2}` with `markedText="ba"` on reactivation — but the emoji was visually committed. The markedRange points at residual text from the document, not at bee's emoji. The proxy's markedRange is stale/wrong for Codex.

**Evidence**: [E-003-reactivation-state](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-003-reactivation-state.md)

## F-010-zed-double-activate

Zed fires two `activateServer` calls on return: first showing the emoji in markedText, then showing underlying document text. Zed's text engine processes reactivation in two phases.

**Evidence**: [E-003-reactivation-state](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-003-reactivation-state.md)

## F-011-cleanup-works-for-normal-apps (UNRELIABLE)

`setMarkedText("")` on reactivation cleared leftover marked text in Messages in E-004 (user confirmed). However, E-006 with the same approach did NOT clear in Messages. This result is not reliably reproducible — may depend on timing of which `activateServer` call the cleanup fires on.

**Evidence**: [E-004-cleanup-on-reactivation](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-004-cleanup-on-reactivation.md), [E-006-deferred-cleanup](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-006-deferred-cleanup.md)

## F-012-cleanup-missed-codex-due-to-ordering

Cleanup did not fire for Codex because the pendingCleanup recorded the wrong bundle ID (`dev.zed.Zed` instead of `com.openai.codex`). Root cause: `deactivate` reads the client's bundle from the controller, but by that time the controller's client already points at the new app (due to F-013).

**Evidence**: [E-004-cleanup-on-reactivation](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-004-cleanup-on-reactivation.md)

## F-013-activate-before-deactivate-ordering

`activateServer` for the new app fires BEFORE `deactivateServer` for the old app. Confirmed with timestamps: Zed's activateServer at 13:56:41.102 precedes Codex's deactivateServer at 13:56:41.121 (19ms later).

**Evidence**: [E-004-cleanup-on-reactivation](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-004-cleanup-on-reactivation.md)

## F-014-cleanup-fires-too-early

Cleanup fires on the first `activateServer` call, but the proxy's `markedRange` is still `{∅}` at that point. The `setMarkedText("")` goes to a proxy that hasn't reconnected to the real client, so it has no effect.

**Evidence**: [E-005-cleanup-with-stored-bundle](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-005-cleanup-with-stored-bundle.md)

## F-015-proxy-reconnects-on-second-activate

The IMK proxy doesn't expose valid markedRange on the first `activateServer` — it shows `{∅}`. A second `activateServer` follows shortly after with the real markedRange. Cleanup must be deferred until the proxy reports a valid markedRange.

**Evidence**: [E-005-cleanup-with-stored-bundle](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-005-cleanup-with-stored-bundle.md)

## F-016-setMarkedText-empty-does-not-clear

`setMarkedText("")` on reactivation does NOT visually clear the marked text in either Messages or Codex, despite the proxy reporting `markedRange` going from `{3, N}` to `{3, 0}`. User confirmed: text unchanged in both apps.

**Evidence**: [E-006-deferred-cleanup](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-006-deferred-cleanup.md)

## F-017-proxy-lies-about-cleanup

The IMK proxy reports `markedRange={3, 0}` after `setMarkedText("")`, suggesting the marked range was cleared. This is a lie — the actual text in the client app is unaffected. The proxy's markedRange is not a reliable indicator of the client's actual state.

**Evidence**: [E-006-deferred-cleanup](https://github.com/fasterthanlime/bee-experiments/blob/main/experiments/E-006-deferred-cleanup.md)
