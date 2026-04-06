# 007: Online learning strategy for production

## Goal

Define how the two-stage judge learns from user feedback in production,
without destabilizing the trained weights.

## Phased approach

### Phase 1: Fixed weights + memory only

- Ship with offline-trained gate + ranker weights (frozen)
- Memory updates immediately (TermMemory tracks accept/reject per term)
- Memory features flow into both gate and ranker
- Log all correction events for later offline retraining
- This is safe — no weight updates, only memory counters change

### Phase 2: Cautious online updates

Only after Phase 1 is validated in production:

- Gate updates: user accepts correction → positive, user reverts → negative
- Ranker updates: user picks candidate → positive for chosen, negative for
  what the ranker had picked (if different)
- Use low learning rate (α=0.1 instead of 0.5)
- Only update on high-confidence teaching signals (explicit user action)
- Never update on "user did nothing" — ambiguous signal

### Phase 3: Periodic offline retraining

- Collect correction events from production
- Add to eval corpus
- Retrain offline with k-fold validation
- Ship updated weights in app update
- This is the most reliable path to improvement

## Key principle

Online learning is a product risk. The offline eval proves the architecture
works. Don't rush online updates — get the fixed-weight version right first,
then add learning cautiously.

## Depends on

- 002 (TwoStageJudge in production)
- 003 (bee-ffi teaches corrections back)
