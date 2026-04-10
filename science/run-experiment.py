#!/usr/bin/env python3
"""Run an IME science experiment with proper bookkeeping."""

import argparse
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCIENCE_DIR = REPO_ROOT / "science"
EXPERIMENTS_DIR = SCIENCE_DIR / "experiments"
LOGS_DIR = SCIENCE_DIR / "logs"
BEE_LOG = Path.home() / "Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee/bee.log"

TEST_APPS = [
    ("ime-spy", "com.fasterthanlime.ime-spy", "Custom NSTextInputClient, logs everything"),
    ("Notes", "com.apple.Notes", "Standard Apple app, NSTextView-based"),
    ("Messages", "com.apple.MobileSMS", "Known to hold onto marked text"),
    ("Zed", "dev.zed.Zed", "Custom text engine"),
    ("Codex", "com.openai.codex", "Electron/Chromium"),
]


def run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, **kwargs)


def git_is_clean():
    r = run(["git", "status", "--porcelain"])
    lines = [
        l
        for l in r.stdout.strip().splitlines()
        if l and not l.lstrip("? ").startswith("science/")
    ]
    return len(lines) == 0, r.stdout.strip()


def git_sha():
    return run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def git_sha_full():
    return run(["git", "rev-parse", "HEAD"]).stdout.strip()


def next_experiment_number():
    existing = list(EXPERIMENTS_DIR.glob("E-*.md"))
    numbers = []
    for p in existing:
        try:
            numbers.append(int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            pass
    return max(numbers, default=0) + 1


def truncate_log():
    if BEE_LOG.exists():
        BEE_LOG.write_text("")
        return True
    return False


def collect_log(dest):
    if BEE_LOG.exists():
        shutil.copy2(BEE_LOG, dest)
        line_count = sum(1 for _ in open(dest))
        return line_count
    else:
        dest.write_text("(no log file found)\n")
        return 0


def prompt_observations(app_name):
    print(f"    Observations for {app_name}? (one line, or empty to skip)")
    line = input("    > ").strip()
    return line if line else None


def main():
    parser = argparse.ArgumentParser(description="Run an IME science experiment")
    parser.add_argument("slug", help="Descriptive slug (e.g. baseline-app-switch)")
    parser.add_argument(
        "--description", "-d", required=True,
        help="One-line description of what we're testing",
    )
    parser.add_argument(
        "--procedure", "-p", required=True,
        help="Step-by-step procedure to perform in each app (use \\n for newlines)",
    )
    parser.add_argument("--prediction", default="", help="What we expect to happen")
    parser.add_argument(
        "--skip-clean-check", action="store_true",
        help="Skip the git clean check",
    )
    parser.add_argument(
        "--apps", nargs="*",
        help="Only test specific apps (by short name). Default: all five.",
    )
    args = parser.parse_args()

    # 1. Check git is clean
    if not args.skip_clean_check:
        clean, status = git_is_clean()
        if not clean:
            print(
                "ERROR: Repository has uncommitted changes. Commit first.\n",
                file=sys.stderr,
            )
            print(status, file=sys.stderr)
            sys.exit(1)

    # 2. Determine experiment ID
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    num = next_experiment_number()
    experiment_id = f"E-{num:03d}-{args.slug}"
    sha = git_sha()
    sha_full = git_sha_full()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Filter apps if requested
    if args.apps:
        apps = [a for a in TEST_APPS if a[0].lower() in [x.lower() for x in args.apps]]
        if not apps:
            print(f"ERROR: No matching apps. Available: {[a[0] for a in TEST_APPS]}", file=sys.stderr)
            sys.exit(1)
    else:
        apps = TEST_APPS

    print(f"\n  Experiment: {experiment_id}")
    print(f"  Commit:     {sha}")
    print(f"  Time:       {timestamp}")
    print(f"  Description: {args.description}")
    print(f"  Apps:       {', '.join(a[0] for a in apps)}")
    print()

    procedure_lines = args.procedure.replace("\\n", "\n")

    # 3. Run against each app
    app_results = []
    for app_name, bundle_id, app_notes in apps:
        print(f"  ┌─ {app_name} ({bundle_id})")
        print(f"  │  {app_notes}")
        print(f"  │")

        # Truncate log
        if truncate_log():
            print(f"  │  [ok] Truncated bee.log")
        else:
            print(f"  │  [warn] bee.log not found")

        # Show procedure
        print(f"  │")
        print(f"  │  Procedure:")
        for line in procedure_lines.splitlines():
            print(f"  │    {line}")
        print(f"  │")
        print(f"  │  Perform the test in {app_name} now. Press ENTER when done.")
        input(f"  │  > ")

        # Collect log
        log_file = LOGS_DIR / f"{experiment_id}-{app_name.lower()}.txt"
        line_count = collect_log(log_file)
        print(f"  │  [ok] Collected {line_count} log lines")

        # Observations
        obs = prompt_observations(app_name)

        app_results.append({
            "name": app_name,
            "bundle_id": bundle_id,
            "log_file": log_file.name,
            "line_count": line_count,
            "observation": obs,
        })

        print(f"  └─ done\n")

    # 4. Write experiment file
    procedure_formatted = "\n".join(f"   {line}" for line in procedure_lines.splitlines())

    results_section = ""
    for r in app_results:
        obs = r["observation"] or "(no observation recorded)"
        results_section += f"### {r['name']} (`{r['bundle_id']}`)\n\n"
        results_section += f"- **Log**: [`logs/{r['log_file']}`](../logs/{r['log_file']})"
        results_section += f" ({r['line_count']} lines)\n"
        results_section += f"- **Observation**: {obs}\n\n"

    experiment_md = textwrap.dedent(f"""\
    # {experiment_id}

    **Date**: {timestamp}
    **Commit**: `{sha_full}` (`{sha}`)
    **Description**: {args.description}

    ## Procedure (per app)

    {procedure_formatted}

    ## Prediction

    {args.prediction or "(none)"}

    ## Results

    {results_section}
    ## Conclusions

    (to be filled in after analysis)
    """)

    experiment_path = EXPERIMENTS_DIR / f"{experiment_id}.md"
    experiment_path.write_text(experiment_md)
    print(f"  [ok] Wrote {experiment_path.relative_to(REPO_ROOT)}")
    print(f"\n  Done. Fill in conclusions, then update science/facts.md.\n")


if __name__ == "__main__":
    main()
