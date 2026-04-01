#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$(pwd)/build-bee-release"
INPUT_METHOD_DIR="$HOME/Library/Input Methods"

echo "Building MLX Rust FFI (release)..."
(cd rust && cargo build --release -p bee-ffi 2>&1 | tail -3)

echo "Generating Xcode project..."
xcodegen generate --spec bee-project.yml

echo "Building bee (Release)..."
xcodebuild -project bee.xcodeproj -scheme bee -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Building beeInput (Release)..."
xcodebuild -project bee.xcodeproj -scheme beeInput -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Installing bee to /Applications/bee.app..."
rsync -a --delete "$BUILD_DIR/bee.app/" /Applications/bee.app/

echo "Installing beeInput to $INPUT_METHOD_DIR/beeInput.app..."
mkdir -p "$INPUT_METHOD_DIR"
rsync -a --delete "$BUILD_DIR/beeInput.app/" "$INPUT_METHOD_DIR/beeInput.app/"

echo "Done. Launch bee from /Applications/bee.app"
