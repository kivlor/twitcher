#!/usr/bin/env bash
# M0 spike: cross-compile the TensorFlow Lite C library (libtensorflowlite_c.so)
# for 32-bit ARMv7 (Raspberry Pi 3B target), then build the go-tflite spike binary
# for GOARCH=arm GOARM=7.
#
# Designed to run INSIDE a debian/ubuntu container (amd64) with the repo mounted
# at /work. Requires: bash, cmake, make, git, curl, build-essential,
# gcc-arm-linux-gnueabihf, golang (>= 1.24 for #cgo noescape/nocallback pragmas).
set -euo pipefail

TFLITE_VERSION=v2.17.1          # same pin as upstream birdnet-go Taskfile.yml
WORK=/work
CACHE="$WORK/.cache/m0"
TF_SRC="$CACHE/tensorflow"
BUILD="$CACHE/build-armv7"
OUT="$WORK/build/armv7"

mkdir -p "$CACHE" "$OUT"

# --- 1. TensorFlow sources (sparse: everything cmake build needs) -------------
if [ ! -d "$TF_SRC/.git" ]; then
  echo "==> cloning tensorflow $TFLITE_VERSION (depth 1)"
  git clone --branch "$TFLITE_VERSION" --depth 1 \
    https://github.com/tensorflow/tensorflow.git "$TF_SRC"
fi

# --- 2. Cross-compile TFLite C API for ARMv7 ----------------------------------
if [ ! -f "$BUILD/libtensorflowlite_c.so" ]; then
  echo "==> cmake configure (arm-linux-gnueabihf)"
  cmake -S "$TF_SRC/tensorflow/lite/c" -B "$BUILD" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=armv7l \
    -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc \
    -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
    -DTFLITE_ENABLE_XNNPACK=OFF \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_NNAPI=OFF
  echo "==> cmake build"
  cmake --build "$BUILD" -j"$(nproc)"
fi

cp -f "$BUILD/libtensorflowlite_c.so" "$OUT/"
echo "==> built $OUT/libtensorflowlite_c.so"
file "$OUT/libtensorflowlite_c.so" || arm-linux-gnueabihf-readelf -h "$OUT/libtensorflowlite_c.so" | head -12

# --- 3. Cross-compile the Go spike for GOARCH=arm GOARM=7 ----------------------
echo "==> go build GOARCH=arm GOARM=7"
cd "$WORK"
CGO_ENABLED=1 GOOS=linux GOARCH=arm GOARM=7 \
CC=arm-linux-gnueabihf-gcc CXX=arm-linux-gnueabihf-g++ \
CGO_CFLAGS="-I$TF_SRC" \
CGO_LDFLAGS="-L$BUILD -ltensorflowlite_c -ldl -lrt" \
  go build -o "$OUT/m1-spike" ./cmd/m1

arm-linux-gnueabihf-readelf -h "$OUT/m1-spike" | head -12
echo "==> M0 RESULT: armv7 binary built OK: $OUT/m1-spike"
