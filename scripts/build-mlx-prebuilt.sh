#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLX_C_DIR="$REPO_ROOT/mlx-rs/mlx-sys/src/mlx-c"

PREFIX="${1:-$REPO_ROOT/build/mlx-prebuilt}"
BUILD_DIR="${BUILD_DIR:-$PREFIX/_build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
MLX_BUILD_METAL="${MLX_BUILD_METAL:-ON}"
MLX_BUILD_ACCELERATE="${MLX_BUILD_ACCELERATE:-ON}"
JOBS="${JOBS:-}"

echo "Configuring mlx-c with Ninja..."
cmake -S "$MLX_C_DIR" -B "$BUILD_DIR" -G Ninja \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DMLX_BUILD_METAL="$MLX_BUILD_METAL" \
  -DMLX_BUILD_ACCELERATE="$MLX_BUILD_ACCELERATE"

echo "Building mlx-c..."
if [[ -n "$JOBS" ]]; then
  cmake --build "$BUILD_DIR" --parallel "$JOBS"
else
  cmake --build "$BUILD_DIR" --parallel
fi

echo "Installing mlx-c..."
cmake --install "$BUILD_DIR"

mkdir -p "$PREFIX/lib"

ensure_static_lib() {
  local lib_name="$1"
  local target="$PREFIX/lib/$lib_name"

  if [[ -f "$target" ]]; then
    return
  fi

  local found=""
  while IFS= read -r path; do
    found="$path"
    break
  done < <(find "$BUILD_DIR" -type f -name "$lib_name")

  if [[ -z "$found" ]]; then
    echo "error: could not locate $lib_name in $BUILD_DIR" >&2
    exit 1
  fi

  cp "$found" "$target"
}

ensure_static_lib "libmlx.a"
ensure_static_lib "libmlxc.a"

echo
echo "Prebuilt MLX artifacts are ready:"
echo "  PREFIX=$PREFIX"
echo
echo "Use them with:"
echo "  MLX_SYS_PREFIX=\"$PREFIX\" cargo check"
echo "  MLX_SYS_PREFIX=\"$PREFIX\" cargo run"
