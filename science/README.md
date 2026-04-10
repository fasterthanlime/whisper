# IME Science

Systematic experiments to understand macOS IME behavior for bee (a palette-type input method).

## Methodology

### Facts

Each fact has:
- **ID**: `F-NNN-descriptive-slug` (number for ordering, slug for meaning)
- **Statement**: A precise, falsifiable claim
- **Evidence**: Which experiment(s) established it
- **Commit**: The git SHA of the code that was running when the evidence was collected

Facts live in `facts.md`.

### Experiments

Each experiment has:
- **ID**: `E-NNN-descriptive-slug` (number for ordering, slug for meaning)
- **Commit**: The exact git SHA of the code under test
- **Setup**: What code is deployed, what apps are open
- **Procedure**: Step-by-step what the operator does
- **Predictions**: What we expect to happen (if any)
- **Raw logs**: The full parsed log output, stored in `logs/E-NNN.txt`
- **Observations**: What actually happened (including screenshots if relevant)
- **Conclusions**: What facts are established or refuted

Experiments live in `experiments/E-NNN-slug.md`. Logs live in `logs/E-NNN-slug.txt`.

### Test apps

Experiments should be run against multiple client apps to distinguish app-specific behavior from universal behavior:

| App | Bundle ID | Notes |
|-----|-----------|-------|
| ime-spy | com.fasterthanlime.ime-spy | Custom NSTextInputClient, logs everything |
| Notes | com.apple.Notes | Standard Apple app, NSTextView-based |
| Messages | com.apple.MobileSMS | Known to hold onto marked text |
| Zed | dev.zed.Zed | Custom text engine |
| Codex | com.openai.codex | Electron/Chromium |

### Running an experiment

Use `science/run-experiment.py`:

```bash
python3 science/run-experiment.py baseline-app-switch \
  -d "Test whether marked text clears on app switch with stripped-down overrides" \
  -p "1. Open ime-spy\n2. Start bee session (right-option)\n3. Wait 3 heartbeats\n4. Cmd+Tab to Zed\n5. Observe ime-spy window" \
  --prediction "Marked text stays in ime-spy"
```

The script:
1. Verifies git is clean (all code committed)
2. Records the commit SHA
3. Truncates bee.log
4. Waits for you to perform the experiment
5. Collects the full log to `logs/E-NNN-slug.txt`
6. Prompts for observations
7. Writes the experiment file to `experiments/E-NNN-slug.md`
