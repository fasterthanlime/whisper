# 006: Establish three permanent scoreboards

## Goal

Always report three separate metrics to prevent denominator confusion.

## Scoreboards

### 1. End-to-end

Denominator: all canonical cases (106).

| Metric | Definition |
|--------|------------|
| Canonical corrected | Gold correction applied / all canonical |
| Counterexample abstained | No replacement made / all counterexamples |
| False positive rate | Wrong replacement made / all counterexamples |

### 2. Judge-stage

Denominator: reachable cases only (86 canonical, 113 cx).

| Metric | Definition |
|--------|------------|
| Gate open/close accuracy | Balanced accuracy of gate alone |
| Ranker top-1 | Gold is highest-scored candidate / reachable canonical |
| Composed balanced accuracy | (can_acc + cx_acc) / 2 at chosen thresholds |

### 3. Upstream opportunity set

Shows where cases are lost before the judge.

| Metric | Definition |
|--------|------------|
| Gold retrieved | Gold term appears in shortlist / all canonical |
| Gold verified | Gold candidate passes verification / all canonical |
| Gold reaches judge | Gold in judge-visible decision set / all canonical |

## Implementation

- Update `--offline-eval` output to print all three scoreboards
- Update the web UI (OfflineJudgeEvalPanel) to show all three
- Update eval-results.md template to use this structure

## Depends on

- No code dependencies — reformats existing data
