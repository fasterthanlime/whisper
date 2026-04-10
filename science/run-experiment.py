#!/usr/bin/env python3
"""
IME science experiment tool.

Two phases:
  prepare  - creates the experiment file (commit, procedure, predictions)
  run      - interactive: cycles through apps, collects logs & observations
"""

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_REPO = Path.home() / "bearcove/bee-experiments"
EXPERIMENTS_DIR = EXPERIMENTS_REPO / "experiments"
LOGS_DIR = EXPERIMENTS_REPO / "logs"
BEE_LOG = Path.home() / "Library/Group Containers/B2N6FSRTPV.group.fasterthanlime.bee/bee.log"

TEST_APPS = [
    ("ime-spy", "com.fasterthanlime.ime-spy", "Custom NSTextInputClient, logs everything"),
    ("Notes", "com.apple.Notes", "Standard Apple app, NSTextView-based"),
    ("Messages", "com.apple.MobileSMS", "Known to hold onto marked text"),
    ("Zed", "dev.zed.Zed", "Custom text engine"),
    ("Codex", "com.openai.codex", "Electron/Chromium"),
]


def sh(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT, **kwargs)


def git_is_clean():
    r = sh(["git", "status", "--porcelain"])
    lines = [
        l for l in r.stdout.strip().splitlines()
        if l and not l.lstrip("? ").startswith("science/")
    ]
    return len(lines) == 0, r.stdout.strip()


def git_sha():
    return sh(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def git_sha_full():
    return sh(["git", "rev-parse", "HEAD"]).stdout.strip()


def next_experiment_number():
    existing = list(EXPERIMENTS_DIR.glob("E-*.md"))
    numbers = []
    for p in existing:
        try:
            numbers.append(int(p.stem.split("-")[1]))
        except (IndexError, ValueError):
            pass
    return max(numbers, default=0) + 1


def resolve_apps(app_names):
    if not app_names:
        return TEST_APPS
    apps = [a for a in TEST_APPS if a[0].lower() in [x.lower() for x in app_names]]
    if not apps:
        print(f"ERROR: No matching apps. Available: {[a[0] for a in TEST_APPS]}", file=sys.stderr)
        sys.exit(1)
    return apps


# ── prepare ──────────────────────────────────────────────────────────

def cmd_prepare(args):
    # Check git is clean
    clean, status = git_is_clean()
    if not clean:
        print("ERROR: Repository has uncommitted changes. Commit first.\n", file=sys.stderr)
        print(status, file=sys.stderr)
        sys.exit(1)

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    num = next_experiment_number()
    experiment_id = f"E-{num:03d}-{args.slug}"
    sha = git_sha()
    sha_full = git_sha_full()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    apps = resolve_apps(args.apps)

    procedure_lines = args.procedure.replace("\\n", "\n")
    procedure_formatted = "\n".join(f"   {line}" for line in procedure_lines.splitlines())

    app_list = "\n".join(f"- {name} (`{bid}`)" for name, bid, _ in apps)

    experiment_md = textwrap.dedent(f"""\
    # {experiment_id}

    **Date**: {timestamp}
    **Commit**: `{sha_full}` (`{sha}`)
    **Description**: {args.description}

    ## Procedure (per app)

    {procedure_formatted}

    ## Prediction

    {args.prediction or "(none)"}

    ## Test apps

    {app_list}

    ## Results

    (run `python3 science/run-experiment.py run {experiment_id}` to collect)

    ## Conclusions

    (to be filled in after analysis)
    """)

    experiment_path = EXPERIMENTS_DIR / f"{experiment_id}.md"
    experiment_path.write_text(experiment_md)

    # Also write a machine-readable sidecar for the run phase
    meta = {
        "id": experiment_id,
        "sha": sha,
        "sha_full": sha_full,
        "timestamp": timestamp,
        "description": args.description,
        "procedure": procedure_lines,
        "prediction": args.prediction or "",
        "apps": [{"name": a[0], "bundle_id": a[1], "notes": a[2]} for a in apps],
    }
    meta_path = EXPERIMENTS_DIR / f"{experiment_id}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\n  Prepared: {experiment_id}")
    print(f"  Commit:   {sha}")
    print(f"  Apps:     {', '.join(a[0] for a in apps)}")
    print(f"  Files:")
    print(f"    {experiment_path.relative_to(EXPERIMENTS_REPO)}")
    print(f"    {meta_path.relative_to(EXPERIMENTS_REPO)}")
    print(f"\n  To run:  python3 science/run-experiment.py run {experiment_id}\n")


# ── run ──────────────────────────────────────────────────────────────

def cmd_run(args):
    # Find the experiment metadata
    meta_path = EXPERIMENTS_DIR / f"{args.experiment_id}.json"
    if not meta_path.exists():
        # Try glob in case they gave a partial ID
        candidates = list(EXPERIMENTS_DIR.glob(f"{args.experiment_id}*.json"))
        if len(candidates) == 1:
            meta_path = candidates[0]
        elif len(candidates) > 1:
            print(f"ERROR: Ambiguous ID. Matches: {[c.stem for c in candidates]}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"ERROR: No experiment found for '{args.experiment_id}'", file=sys.stderr)
            print(f"  Available:", file=sys.stderr)
            for p in sorted(EXPERIMENTS_DIR.glob("E-*.json")):
                print(f"    {p.stem}", file=sys.stderr)
            sys.exit(1)

    meta = json.loads(meta_path.read_text())
    experiment_id = meta["id"]
    apps = meta["apps"]
    procedure = meta["procedure"]

    # Record the actual commit we're running at
    current_sha = git_sha()
    current_sha_full = git_sha_full()
    if current_sha != meta["sha"]:
        print(f"  Note: Running at {current_sha} (prepared at {meta['sha']})")
    meta["run_sha"] = current_sha
    meta["run_sha_full"] = current_sha_full

    print(f"\n  Running: {experiment_id}")
    print(f"  Description: {meta['description']}")
    print()

    # Build and install bee
    print("  ┌─ Building bee...")
    build = subprocess.run(
        ["just", "install-debug"],
        cwd=REPO_ROOT,
        capture_output=False,
    )
    if build.returncode != 0:
        print("  │  ERROR: build failed")
        sys.exit(1)
    print("  └─ [ok] bee installed\n")

    # Wait a moment for the IME to be ready
    import time
    print("  Waiting 2s for IME to load...")
    time.sleep(2)
    print()

    app_results = []
    for app in apps:
        app_name = app["name"]
        bundle_id = app["bundle_id"]
        notes = app["notes"]

        print(f"  ┌─ {app_name} ({bundle_id})")
        print(f"  │  {notes}")
        print(f"  │")

        # Truncate log
        if BEE_LOG.exists():
            BEE_LOG.write_text("")
            print(f"  │  [ok] Truncated bee.log")
        else:
            print(f"  │  [warn] bee.log not found")

        # Show procedure
        print(f"  │")
        print(f"  │  Procedure:")
        for line in procedure.splitlines():
            print(f"  │    {line}")
        print(f"  │")
        print(f"  │  Perform the test in {app_name}. Press ENTER when done.")
        input(f"  │  > ")

        # Collect log
        log_file = LOGS_DIR / f"{experiment_id}-{app_name.lower()}.txt"
        if BEE_LOG.exists():
            shutil.copy2(BEE_LOG, log_file)
            line_count = sum(1 for _ in open(log_file))
        else:
            log_file.write_text("(no log file found)\n")
            line_count = 0
        print(f"  │  [ok] Collected {line_count} log lines")

        # Observation
        print(f"  │")
        print(f"  │  Observation for {app_name}? (one line, or empty to skip)")
        obs = input(f"  │  > ").strip() or None

        app_results.append({
            "name": app_name,
            "bundle_id": bundle_id,
            "log_file": log_file.name,
            "line_count": line_count,
            "observation": obs,
        })

        print(f"  └─ done\n")

    # Update the experiment markdown with results
    results_section = ""
    for r in app_results:
        obs = r["observation"] or "(no observation recorded)"
        results_section += f"### {r['name']} (`{r['bundle_id']}`)\n\n"
        results_section += f"- **Log**: [`logs/{r['log_file']}`](../logs/{r['log_file']})"
        results_section += f" ({r['line_count']} lines)\n"
        results_section += f"- **Observation**: {obs}\n\n"

    experiment_path = EXPERIMENTS_DIR / f"{experiment_id}.md"
    md = experiment_path.read_text()
    md = md.replace(
        f"(run `python3 science/run-experiment.py run {experiment_id}` to collect)",
        results_section.rstrip(),
    )
    experiment_path.write_text(md)

    # Also save results to the JSON
    meta["results"] = app_results
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"  [ok] Updated {experiment_path.relative_to(EXPERIMENTS_REPO)}")
    print(f"\n  Done. Fill in conclusions, then update science/facts.md.\n")


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="IME science experiment tool")
    sub = parser.add_subparsers(dest="command")

    p_prepare = sub.add_parser("prepare", help="Create an experiment (no interaction)")
    p_prepare.add_argument("slug", help="Descriptive slug (e.g. baseline-app-switch)")
    p_prepare.add_argument("-d", "--description", required=True, help="What we're testing")
    p_prepare.add_argument("-p", "--procedure", required=True, help="Steps per app (\\n for newlines)")
    p_prepare.add_argument("--prediction", default="", help="What we expect")
    p_prepare.add_argument("--apps", nargs="*", help="Subset of apps to test")

    p_run = sub.add_parser("run", help="Run a prepared experiment (interactive)")
    p_run.add_argument("experiment_id", help="Experiment ID (e.g. E-001-baseline-app-switch)")

    args = parser.parse_args()
    if args.command == "prepare":
        cmd_prepare(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
