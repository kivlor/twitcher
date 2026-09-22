# twitcher

> **twitcher** *n. Brit. informal — a birdwatcher who travels long distances to see rare birds.*

A minimal, headless bird-detection service for the **Raspberry Pi 3B** (arm64, 1 GB RAM). It
listens to a microphone, runs [BirdNET](https://birdnet.cornelllab.org/) v2.4 inference on
overlapping 3-second audio chunks, and writes detections to a local SQLite database that an
external service can query. Optional MQTT/Kafka event emission is planned.

It is a deliberately tiny, single-purpose program — no web UI, no HTTP API, no integrations.

## Relationship to BirdNET-Go

twitcher is **derived from** — and would not exist without —
[BirdNET-Go](https://github.com/tphakala/birdnet-go) by [Timo Hakala](https://github.com/tphakala)
and contributors. BirdNET-Go is a full-featured, self-hosted bird song analyzer with a web UI,
notifications, weather/eBird integrations and more — but it no longer supports the Raspberry Pi 3.

twitcher was built by extracting a minimal subset of BirdNET-Go's architecture and code and
rebuilding it for the Pi 3B's constraints:

- **Classifier:** BirdNET v2.4 TFLite path from `internal/classifier/`
  ([birdnet.go](https://github.com/tphakala/birdnet-go/blob/main/internal/classifier/birdnet.go),
  [analyze.go](https://github.com/tphakala/birdnet-go/blob/main/internal/classifier/analyze.go),
  [overlap.go](https://github.com/tphakala/birdnet-go/blob/main/internal/classifier/overlap.go))
  via [tphakala/go-tflite](https://github.com/tphakala/go-tflite) — trimmed to a single model,
  single backend.
- **Chunking pattern:** the buffer → overlapping-3-s-window → queue shape of
  [`internal/analysis/`](https://github.com/tphakala/birdnet-go/tree/main/internal/analysis),
  re-implemented in ~300 lines.
- **Audio capture:** the malgo/miniaudio ALSA init pattern of
  [`internal/audiocore/capture.go`](https://github.com/tphakala/birdnet-go/blob/main/internal/audiocore/capture.go),
  re-implemented without the router/encoder/level machinery.
- **Labels:** the BirdNET v2.4 species label list (~6,522 species) from
  [`internal/labels/`](https://github.com/tphakala/birdnet-go/tree/main/internal/labels).
- **DB schema (planned):** the `notes`/`results` GORM entities of
  [`internal/datastore/entities/note.go`](https://github.com/tphakala/birdnet-go/blob/main/internal/datastore/entities/note.go),
  kept schema-compatible so tooling written against BirdNET-Go can read twitcher's database.
- **MQTT emission (planned):** modeled on BirdNET-Go's
  [`internal/mqtt/`](https://github.com/tphakala/birdnet-go/tree/main/internal/mqtt).

The full extraction plan — what was taken, what was left behind, and why — lives in
[EXTRACTION_PLAN.md](EXTRACTION_PLAN.md).

Many thanks to the BirdNET-Go authors and, upstream of that, the
[BirdNET](https://github.com/kahst/BirdNET) project and the Cornell Lab of Ornithology.

## Measured performance on a real Pi 3B (Raspberry Pi OS Lite 64-bit, trixie)

| Metric | Value |
|---|---|
| Inference (FP32 model, 3 s chunk) | **0.47 s** (~6.3× real time) |
| Peak RSS during inference | **142 MiB** |
| Sustained-load CPU temperature | 57.5 °C, no throttling |

## Status

- ✅ **M1 — Offline classify** (`cmd/m1`): read a WAV, classify each 3 s chunk, print top-3.
- ✅ **M2 — Stream + detect** (`cmd/m2`): live ALSA capture → chunker with overlap → inference →
  log detections above a confidence threshold.
- ⬜ **M3 — Persist**: SQLite `notes`/`results`, BirdNET-Go-compatible schema.
- ⬜ **M4 — Harden**: queue-drop, device reconnect backoff, retention sweep, systemd unit,
  clip recording, MQTT/Kafka event emitter.
- ⬜ **M5 — Pi deployment**: 24 h soak test, swap/zram docs.

## Building for the Pi 3B

Cross-compile (CGO required — go-tflite + the prebuilt
[`tphakala/tflite_c`](https://github.com/tphakala/tflite_c) v2.17.1 linux-arm64 shared library):

```sh
./scripts/build_arm64.sh   # docker run ubuntu:24.04 + aarch64 toolchain
```

Model (`BirdNET_GLOBAL_6K_V2.4_M_Data1.1_FP32.tflite`), labels, and
`libtensorflowlite_c.so` are **not** committed; see [EXTRACTION_PLAN.md](EXTRACTION_PLAN.md)
and the [BirdNET-Go releases](https://github.com/tphakala/birdnet-go/releases) for sources.

## Running

```sh
LD_LIBRARY_PATH=build/arm64 ./build/arm64/m2 \
  -model model.tflite -labels labels.txt -device default \
  -threshold 0.80 -overlap 0.33
```

## License

BirdNET-Go's code and the BirdNET model are licensed **CC-BY-NC-SA 4.0**
(non-commercial). twitcher carries the same license and attribution — see
[BirdNET-Go's LICENSES.md](https://github.com/tphakala/birdnet-go/blob/main/LICENSES.md).
