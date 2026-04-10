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
