#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$(pwd)/build-release"

echo "Building hark (Release)..."
xcodebuild -project hark.xcodeproj -scheme hark -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Installing to /Applications/hark.app..."
rsync -a --delete "$BUILD_DIR/hark.app/" /Applications/hark.app/

echo "Done."
