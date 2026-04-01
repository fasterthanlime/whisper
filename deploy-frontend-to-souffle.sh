#!/usr/bin/env bash
set -euo pipefail

REMOTE=souffle
REMOTE_ROOT=~/bearcove/hark/beeml/crates/synth-dashboard
LOCAL_ROOT=/Users/amos/bearcove/hark/beeml/crates/synth-dashboard

echo "=== Syncing dashboard frontend to souffle ==="
cd "$LOCAL_ROOT"
rsync -av \
  static/index.html \
  static/fonts \
  "$REMOTE:$REMOTE_ROOT/static/"

echo
echo "Frontend files synced."
echo "The running dashboard serves index.html from disk, so no rebuild or restart is required."
