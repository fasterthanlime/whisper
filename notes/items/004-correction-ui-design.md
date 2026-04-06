# 004: Correction UI in the bee app

## Goal

Design and implement the user-facing correction experience in the
dictation IME.

## Depends on

- 003 (bee-ffi correction API available from Swift)

## Architecture: Option C (correct-on-insert with undo)

1. User dictates → ASR produces transcript
2. Correction engine processes transcript → decision set
3. Insert corrected text (not raw transcript) via `insertText`
4. Track the insertion range (cursor position + length)
5. Brief subtle notification: "2 corrections applied" (or nothing if zero)
6. User hits ROpt-C → correction review panel opens
7. Panel shows sentence with correction ranges, each accept/reject-able
8. Accept/reject feeds back to `bee_correct_teach`

## IME text replacement: confirmed feasible

`insertText(_:replacementRange:)` on the IMK client supports replacing
specific character ranges in the host app's text. This is the standard
mechanism CJK IMEs use for re-input (e.g., replacing "かな" with "kana").

Works in apps that properly implement `NSTextInputClient` (TextEdit,
Xcode, Safari, most Cocoa apps). May not work in Electron or
cross-platform apps that half-implement the protocol — same constraint
all IMEs live with.

### What to track per insertion

```swift
struct InsertionRecord {
    let range: NSRange           // where we inserted
    let rawTranscript: String    // what ASR produced
    let correctedText: String    // what we actually inserted
    let corrections: [CorrectionRange]  // individual corrections applied
    let timestamp: Date
}

struct CorrectionRange {
    let originalSpan: Range<Int>     // character range in raw transcript
    let replacementText: String      // what we replaced it with
    let aliasId: UInt32              // which vocab term
    let term: String                 // canonical term name
    let confidence: Float            // gate_prob * ranker_prob
}
```

### Invalidation

If the user moves the cursor away from the insertion point or edits the
text manually, the insertion record is invalidated — we can no longer
safely replace. The correction panel should show "text has been modified,
corrections can no longer be applied" or just not open.

## Correction review panel

Triggered by ROpt-C (Right Option + C). Shows:

### Main view

The sentence with correction ranges highlighted. Each correction shows:
- Original text (struck through or dimmed)
- Replacement text (highlighted)
- Accept/reject toggle per correction

Example:

```
I was working with [Sir Day Jason → serde_json] and [Tokyo → tokio]
to parse the [Jason → JSON] file.

[✓] serde_json  (was: "Sir Day Jason")
[✓] tokio       (was: "Tokyo")
[✓] JSON        (was: "Jason")

[Apply] [Revert All] [Dismiss]
```

### Corrections are range-based, not word-based

"Sir Day Jason" → "serde_json" maps a 3-word span to a 1-word term.
The UI operates on character ranges, not word boundaries.

### Actions

| Action | Effect | Teaching signal |
|--------|--------|-----------------|
| Accept (keep checked) | Correction stays | Positive for gate + ranker |
| Reject (uncheck) | Revert to original text | Negative for gate |
| Revert All | Revert entire sentence to raw transcript | Negative for all |
| Dismiss | Keep whatever is checked, close panel | Mixed signal per correction |

### "Mark as correct" (original was fine)

If no corrections were applied but the user thinks there should have been
one, or if a correction was wrong: the panel should have a way to say
"the original was right." This is a negative gate signal.

### "Add term" flow

If the user knows a term that should be in the vocabulary:

1. User selects a range in the transcript (the misheard text)
2. Types the correct term
3. System adds it to the phonetic index
4. This is vocab expansion — creates a new alias entry

This is a stretch goal. The core flow (accept/reject existing corrections)
should ship first.

## Multiple corrections in one utterance

The panel shows all corrections at once. Each is independently
accept/reject-able. The "Apply" button commits the final mix of
accepted and rejected corrections by replacing the insertion range
with the appropriately modified text.

## Operating point

Default: conservative (GT=0.5, RT=0.2) — zero false positives in eval.
This means the user should almost never see a wrong correction. The cost
is missing some valid corrections, but the user can always re-dictate.

## Prototyping plan

1. **HTML prototype** in beeml web UI — nail the interaction design
   with mock data, no backend needed
2. **SwiftUI panel** — real panel using AppKit/SwiftUI, connected to
   bee-ffi correction API
3. **IME integration** — wire the panel to ROpt-C, track insertion
   ranges, handle replacement

## Open questions

- Should the notification ("2 corrections applied") be visual or audio?
- How long does the insertion record stay valid? Until next dictation?
  Until cursor moves? Time-based expiry?
- Should there be a global history of corrections (not just current
  insertion)?
- What about corrections across multiple dictation segments in the
  same document?
