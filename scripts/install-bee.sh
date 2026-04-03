#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build-bee-release"
INPUT_METHOD_DIR="$HOME/Library/Input Methods"

SWIFT_DIR="$PROJECT_ROOT/swift"
XCODE_PROJECT="$SWIFT_DIR/bee.xcodeproj"
XCODE_SPEC="$SWIFT_DIR/bee-project.yml"
ENVRC_PATH="$PROJECT_ROOT/.envrc"

if [ -t 1 ]; then
  RED=$'\033[0;31m'
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[0;33m'
  BLUE=$'\033[0;34m'
  MAGENTA=$'\033[0;35m'
  CYAN=$'\033[0;36m'
  BOLD=$'\033[1m'
  RESET=$'\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  MAGENTA=''
  CYAN=''
  BOLD=''
  RESET=''
fi

if [ -f "$ENVRC_PATH" ]; then
  # Source project environment from repo root so paths using $PWD resolve correctly.
  pushd "$PROJECT_ROOT" >/dev/null
  # shellcheck disable=SC1090
  source "$ENVRC_PATH"
  popd >/dev/null
fi

banner() {
  local title="$1"
  local width=72
  local inner_width=$((width - 4))
  local pad=$(( (inner_width - ${#title}) / 2 ))
  local right_pad=$((inner_width - ${#title} - pad))
  local left_pad=$(printf '%*s' "$pad" '')
  local right_pad_str=$(printf '%*s' "$right_pad" '')

  printf '\n%s\n' "${CYAN}${BOLD}$(printf '%*s' "$width" '' | tr ' ' '=')${RESET}"
  printf '%s\n' "${CYAN}${BOLD}${left_pad} ${title} ${right_pad_str}${RESET}"
  printf '%s\n\n' "${CYAN}${BOLD}$(printf '%*s' "$width" '' | tr ' ' '=')${RESET}"
}

run_step() {
  local title="$1"
  shift
  banner "$title"
  if ! bash -lc "$*"; then
    printf '%s\n' "${RED}${BOLD}Step failed: $title${RESET}"
    exit 1
  fi
  printf '%s\n' "${GREEN}${BOLD}Step complete: $title${RESET}"
}

run_step "Building MLX Rust FFI (release)" "cd \"$PROJECT_ROOT/rust\" && cargo build --release -p bee-ffi"
run_step "Generating Xcode project" "cd \"$SWIFT_DIR\" && xcodegen generate --spec \"$XCODE_SPEC\""
banner "Building bee (+ embedded beeInput) (Release)"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
# beeInput is a dependency of bee and gets embedded automatically; no separate scheme needed.
schemes=(bee)
pids=()
logs=()
for scheme in "${schemes[@]}"; do
  logfile="$BUILD_DIR/$scheme-build.log"
  logs+=("$logfile")
  (cd "$SWIFT_DIR" && xcodebuild -project "$XCODE_PROJECT" -scheme "$scheme" -configuration Release CONFIGURATION_BUILD_DIR="$BUILD_DIR" -derivedDataPath "$BUILD_DIR/DerivedData-$scheme" build >"$logfile" 2>&1) &
  pids+=($!)
done
failed=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    failed=1
    printf '%s\n' "${RED}${BOLD}Build failed: ${schemes[$i]}${RESET}"
    printf '%s\n' "${YELLOW}Log: ${logs[$i]}${RESET}"
    grep -E '^.*error:' "${logs[$i]}" | head -20 || tail -20 "${logs[$i]}"
  else
    printf '%s\n' "${GREEN}Built: ${schemes[$i]}${RESET}"
  fi
done
if [ "$failed" -ne 0 ]; then
  printf '%s\n' "${RED}${BOLD}Swift build failed${RESET}"
  exit 1
fi
printf '%s\n' "${GREEN}${BOLD}All Swift targets built${RESET}"

run_step "Installing bee to /Applications/bee.app" "rsync -a --delete \"$BUILD_DIR/bee.app/\" /Applications/bee.app/"
# beeInput is now embedded in bee.app/Contents/Library/Input Methods/ and registered
# by bee on first launch. Remove the old standalone installation if present.
run_step "Removing old standalone beeInput from ~/Library/Input Methods" \
  "rm -rf \"$INPUT_METHOD_DIR/beeInput.app\""

run_step "Restarting beeInput" "pkill beeInput || true"
run_step "Killing running bee" "pkill bee || true"
sleep 1
run_step "Launching bee" "open -a /Applications/bee.app"
