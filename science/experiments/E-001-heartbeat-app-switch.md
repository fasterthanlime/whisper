    # E-001-heartbeat-app-switch

    **Date**: 2026-04-10 12:55:59
    **Commit**: `94212d57df9557dba391c7a5069acabd9e44466d` (`94212d5`)
    **Description**: Baseline: does heartbeat marked text clear when switching apps? (stripped-down overrides + super.cancelComposition in deactivateServer)

    ## Procedure (per app)

       1. Focus the target app, click in a text field
   2. Start bee session (hold right-option)
   3. Wait for 2-3 heartbeat emojis to appear
   4. Cmd+Tab to switch to a different app
   5. Observe: does the emoji disappear from the target app?

    ## Prediction

    (none)

    ## Test apps

    - ime-spy (`com.fasterthanlime.ime-spy`)
- Notes (`com.apple.Notes`)
- Messages (`com.apple.MobileSMS`)
- Zed (`dev.zed.Zed`)
- Codex (`com.openai.codex`)

    ## Results

    (run `python3 science/run-experiment.py run E-001-heartbeat-app-switch` to collect)

    ## Conclusions

    (to be filled in after analysis)
