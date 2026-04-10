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
