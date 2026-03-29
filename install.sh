#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$(pwd)/build-release"
INPUT_METHOD_DIR="$HOME/Library/Input Methods"

echo "Building hark (Release)..."
xcodebuild -project hark.xcodeproj -scheme hark -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Building harkInput (Release)..."
xcodebuild -project hark.xcodeproj -scheme harkInput -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Installing hark to /Applications/hark.app..."
rsync -a --delete "$BUILD_DIR/hark.app/" /Applications/hark.app/

echo "Installing harkInput to $INPUT_METHOD_DIR/harkInput.app..."
mkdir -p "$INPUT_METHOD_DIR"
rsync -a --delete "$BUILD_DIR/harkInput.app/" "$INPUT_METHOD_DIR/harkInput.app/"

echo "Done."
