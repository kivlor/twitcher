#!/usr/bin/env bash
# Cross-compile the twitcher binary for linux/arm64 (Raspberry Pi 3B, aarch64).
#
# Runs the build inside an ubuntu:24.04 container with the aarch64 cross
# toolchain (see EXTRACTION_PLAN.md §N4). The prebuilt TFLite C library in
# build/arm64 is linked directly; no TFLite source compile needed.
#
# Output: build/arm64/twitcher
set -euo pipefail

WORK="$(cd "$(dirname "$0")/.." && pwd)"
TF_SRC="$WORK/.cache/m0/tensorflow"
OUT="$WORK/build/arm64"

if [ ! -f "$TF_SRC/tensorflow/lite/c/c_api.h" ]; then
  echo "ERROR: TFLite headers not found at $TF_SRC (run scripts/m0_cross_tflite.sh step 1 or clone tensorflow v2.17.1 sparsely)" >&2
  exit 1
fi
if [ ! -f "$OUT/libtensorflowlite_c.so" ]; then
  echo "ERROR: $OUT/libtensorflowlite_c.so missing (prebuilt tflite_c v2.17.1 linux-arm64)" >&2
  exit 1
fi

echo "==> building twitcher for linux/arm64 in ubuntu:24.04 container"
docker run --rm \
  -v "$WORK":/work -w /work \
  -e CGO_ENABLED=1 -e GOOS=linux -e GOARCH=arm64 \
  -e CC=aarch64-linux-gnu-gcc \
  -e CGO_CFLAGS="-I/work/.cache/m0/tensorflow" \
  -e CGO_LDFLAGS="-L/work/build/arm64 -ltensorflowlite_c -ldl -lrt" \
  ubuntu:24.04 bash -eux -c '
    apt-get update -qq &&
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      gcc-aarch64-linux-gnu libc6-dev-arm64-cross golang-go ca-certificates pkg-config &&
    go build -trimpath -o /work/build/arm64/twitcher ./cmd/twitcher
  '

file "$OUT/twitcher" || true
echo "==> RESULT: $OUT/twitcher built OK"
