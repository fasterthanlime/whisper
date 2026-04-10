# IME Facts (established through debugging)

## IMK lifecycle

- `activateServer` for the **new** app fires **before** `deactivateServer` for the old app.
- By the time `deactivateServer` runs, `IMKInputController.client()` (i.e. the instance method on the controller) may already point to the new app's client proxy.
- However, the `sender` parameter in `deactivateServer` still refers to the old app's client proxy.
- The `sender` proxy can report `markedRange`, `selectedRange`, `bundleIdentifier`, etc. for the old app.
- Calling `setMarkedText("")` or `insertText("", replacementRange:)` on the sender proxy does **not** reach the real client app. The proxy IDs rotate on every call and the calls are no-ops from the client's perspective.
- IMK creates new `IMKInputController` instances freely — multiple inits can fire during a single app switch. Controllers are transient; do not store state on them.

## Palette IME specifics

- Bee is configured as `InputMethodType=palette` in Info.plist.
- `recognizedEvents` is never called by the system for palette IMEs.
- `mouseDown(onCharacterIndex:...)` is never called for palette IMEs.
- `handle(_:client:)` is never called for palette IMEs (key events don't route through the IME).
- `composedString` is **not** queried by IMK during deactivation for palette IMEs.
- `super.deactivateServer(sender)` alone does **not** clear marked text on the client for palette IMEs.

## Marked text behavior across apps

- **ime-spy** (custom NSTextInputClient): Marked text rotates correctly during heartbeat. On app switch, the marked text stays visible in current builds. However, we **did** manage to clear it at one point (and Notes too) — the exact combination of overrides that made it work was lost during refactoring.
- **Notes**: Marked text **does** disappear on app switch.
- **Messages**: Marked text **does not** disappear on app switch (stays stuck).
- **Zed**: Marked text rotates correctly; switching away and back picks up where it left off (not committed).
- **Codex** (Electron): Marked text gets **committed as real text** on deactivate. `setMarkedText("")` is treated as "unmark and keep" rather than "discard." This is an Electron/Chromium bug.

## What we've tried and the results

- `setMarkedText("")` on sender proxy in `deactivateServer`: no-op, doesn't reach client.
- `insertText("", replacementRange: markedRange)` on sender proxy: no-op, doesn't reach client.
- Stashing `lastUsedClient` from `handleSetMarkedText`: proxy IDs rotate, stashed reference is stale.
- `NSTextInputContext.current` / captured `inputContext`: always nil in IME process.
- `composedString` returning `""` always: IMK never calls it during deactivation for palette IMEs, so irrelevant.
- `super.cancelComposition()` inside `deactivateServer`: triggers a cascade of two deactivateServer calls (one for old app, one for new), kills the session, and a fresh activate creates a new session. Unclear if it actually clears marked text on the old client.

## Open questions

- Does `super.cancelComposition()` actually send `setMarkedText("")` to the real client? Need to verify with ime-spy console output.
- ~~Would switching from `palette` to `keyboard` InputMethodType change IMK's deactivation behavior?~~ **NO.** Keyboard IME is a beehive — we'd be in the input source rotation, deal with "maybe" activation, etc. Palette is the right choice.
- Is there a way to send a message directly to a client app's NSTextInputClient without going through the IMK proxy?
