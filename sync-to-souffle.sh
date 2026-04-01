#!/usr/bin/env bash
set -euo pipefail

# Dry run first
echo "=== Dry run ==="
rsync -avn --delete \
  --exclude target/ \
  --exclude build/ \
  --exclude build-bee-release/ \
  --exclude .venv/ \
  --exclude venv/ \
  --exclude .git/ \
  --exclude '*.xcodeproj/' \
  --exclude '*.profraw' \
  --exclude corpus.db --exclude corpus.db-shm --exclude corpus.db-wal \
  --exclude training/ \
  --exclude data/ \
  --exclude audio/ \
  --exclude models/ \
  --exclude voices/ \
  /Users/amos/bearcove/hark/ \
  souffle:bearcove/hark/

echo ""
read -p "Proceed? [y/N] " -n 1 -r
echo ""
[[ $REPLY =~ ^[Yy]$ ]] || exit 0

rsync -av --delete \
  --exclude target/ \
  --exclude build/ \
  --exclude build-bee-release/ \
  --exclude .venv/ \
  --exclude venv/ \
  --exclude .git/ \
  --exclude '*.xcodeproj/' \
  --exclude '*.profraw' \
  --exclude corpus.db --exclude corpus.db-shm --exclude corpus.db-wal \
  --exclude training/ \
  --exclude data/ \
  --exclude audio/ \
  --exclude models/ \
  --exclude voices/ \
  /Users/amos/bearcove/hark/ \
  souffle:bearcove/hark/
