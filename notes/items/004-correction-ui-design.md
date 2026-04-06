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
5. User hits ROpt-C → correction review panel opens
6. Panel shows the **track view**: sentence splits at corrections,
   original on top, correction on bottom
7. User clicks lanes to resolve each correction
8. Apply replaces the inserted text via `insertText(_, replacementRange:)`
9. Teaching events sent to `bee_correct_teach` with stable session/edit IDs

## The panel always shows both

Per consultant feedback: always show raw transcript AND corrected text,
not just the corrected sentence. The teaching signal depends on the
contrast. The track view achieves this naturally — both are visible
at split points.

## IME text replacement: confirmed feasible

`insertText(_:replacementRange:)` on the IMK client supports replacing
specific character ranges. Standard mechanism used by CJK IMEs.

Works in apps that properly implement `NSTextInputClient`. May not work
in Electron or cross-platform apps — same constraint all IMEs live with.

### Invalidation

If the user moves the cursor or edits the text manually, the insertion
record is invalidated. The panel should indicate corrections can no
longer be applied in-place (but teaching signals can still be sent).

## Operating point

Default: conservative (GT=0.5, RT=0.2) — zero false positives in eval.
The product lives or dies on false positives.

## Teaching signals

| User action | Teaching signal |
|-------------|-----------------|
| Accept correction (click bottom lane) | Positive gate + positive ranker |
| Reject correction (click top lane) | Negative gate |
| Accept all (Enter) | Positive for all pending |
| User adds correction (select + type) | Vocabulary expansion signal |
| Dismiss (Esc) | No signal (ambiguous) |

Teaching references stable edit IDs from the process result (see 003),
not raw character offsets. This avoids drift if reprocessed.

## Prototyping

1. HTML prototype in beeml web UI (done — track/lane metaphor validated)
2. SwiftUI panel with mock data
3. SwiftUI panel with real bee-ffi correction API
4. IME integration: ROpt-C triggers panel, Apply does replacement

## Open questions

- How long does the insertion record stay valid? Until next dictation?
  Until cursor moves? Time-based expiry?
- Should there be a global history of corrections?
- What about corrections across multiple dictation segments?
- "Add term" flow: when the user types a new term in the panel, what
  happens? Index update? Alias creation? Verification? This needs its
  own work item once the core flow ships.
