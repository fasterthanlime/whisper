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
if [[ "${BEE_REGEN:-}" == "1" ]]; then
  run_step "Generating Xcode project" "cd \"$SWIFT_DIR\" && xcodegen generate --spec \"$XCODE_SPEC\""
fi
banner "Building bee (+ embedded beeInput) (Release)"
mkdir -p "$BUILD_DIR"
# beeInput is a dependency of bee and gets embedded automatically; no separate scheme needed.
xcodebuild_args=(
  -project "$XCODE_PROJECT"
  -scheme bee
  -configuration Release
  CONFIGURATION_BUILD_DIR="$BUILD_DIR"
  -derivedDataPath "$BUILD_DIR/DerivedData"
  -clonedSourcePackagesDirPath "$BUILD_DIR/SourcePackages"
  -parallelizeTargets
  -jobs "$(sysctl -n hw.ncpu)"
  -skipPackageUpdates
  build
)
if [[ "${BEE_BUILD_TIMING:-}" == "1" ]]; then
  xcodebuild_args+=(-showBuildTimingSummary)
  if ! (cd "$SWIFT_DIR" && xcodebuild "${xcodebuild_args[@]}" 2>&1); then
    printf '%s\n' "${RED}${BOLD}Swift build failed${RESET}"
    exit 1
  fi
else
  if ! (cd "$SWIFT_DIR" && xcodebuild "${xcodebuild_args[@]}" 2>&1 | xcbeautify); then
    printf '%s\n' "${RED}${BOLD}Swift build failed${RESET}"
    exit 1
  fi
fi
printf '%s\n' "${GREEN}${BOLD}All Swift targets built${RESET}"

run_step "Installing bee to /Applications/bee.app" "rsync -a --delete \"$BUILD_DIR/bee.app/\" /Applications/bee.app/"
run_step "Copying mlx.metallib into app bundle" "cp \"$MLX_SYS_PREFIX/lib/mlx.metallib\" /Applications/bee.app/Contents/Frameworks/mlx.metallib"
run_step "Installing beeInput to ~/Library/Input Methods" \
  "rm -rf \"$INPUT_METHOD_DIR/beeInput.app\" && cp -r \"$BUILD_DIR/beeInput.app\" \"$INPUT_METHOD_DIR/\""

run_step "Restarting beeInput" "pkill beeInput || true"
run_step "Killing running bee" "pkill bee || true"
sleep 1
run_step "Launching bee" "open -a /Applications/bee.app"
