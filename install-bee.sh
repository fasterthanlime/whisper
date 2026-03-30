#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$(pwd)/build-bee-release"
INPUT_METHOD_DIR="$HOME/Library/Input Methods"

echo "Building Rust FFI (release)..."
(cd qwen3-asr-rs && cargo build --release -p qwen3-asr-ffi 2>&1 | tail -3)

echo "Building bee (Release)..."
xcodebuild -project bee.xcodeproj -scheme bee -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Building bee-input (Release)..."
xcodebuild -project bee.xcodeproj -scheme bee-input -configuration Release \
    CONFIGURATION_BUILD_DIR="$BUILD_DIR" build 2>&1 | tail -3

echo "Installing bee to /Applications/bee.app..."
rsync -a --delete "$BUILD_DIR/bee.app/" /Applications/bee.app/

echo "Installing bee-input to $INPUT_METHOD_DIR/bee-input.app..."
mkdir -p "$INPUT_METHOD_DIR"
rsync -a --delete "$BUILD_DIR/bee-input.app/" "$INPUT_METHOD_DIR/bee-input.app/"

echo "Done. Launch bee from /Applications/bee.app"
