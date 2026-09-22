# BirdNET-Go Extraction Guide & PRD — twitcher

**Purpose:** This document tells a build agent exactly what to extract from the
[BirdNET-Go](https://github.com/tphakala/birdnet-go) repository and what to leave behind) to build a
minimal, headless bird-detection service for a **Raspberry Pi 3B** (arm64/aarch64, 1 GB RAM), which is
no longer supported by upstream BirdNET-Go.

**Source location:** a clone of BirdNET-Go is expected at `../birdnet-go` (a sibling of this
directory). All `internal/...` and file paths in this document are relative to that clone's root;
repo-root files such as `soundscape.wav` and `LICENSES.md` also resolve there.

**Target result:** a single small Go binary that listens to an audio input, runs BirdNET v2.4
inference, and writes detections to a local SQLite database that an external service can query.

---

## 1. What we need (scope)

1. Capture audio from a sound card / USB mic (Linux/ALSA).
2. Run BirdNET v2.4 classification on 3-second overlapping chunks.
3. Persist detections (species, scientific name, confidence, timestamp, optionally the audio clip)
   to a local SQLite database.
4. Emit each detection as a JSON event to an external **MQTT broker** and/or **Kafka** topic on
   the network (optional, off by default). This is one-way push out; no inbound subscription.
5. The database file on disk is the primary query surface for an external service.

**Explicitly out of scope:** web UI, HTTP API, SSE, WebSocket, authentication,
alert-style notifications (Discord/Telegram/ntfy/BirdWeather), eBird/weather integrations, spectrogram streaming, sound-level
monitoring, multi-model orchestration (Perch, bat models, BirdNET v3, OpenVINO), telemetry, update
mechanism, backups, system dashboard, species translation/localization, range-filter metadata model
(optional — see §6.4).

---

## 2. Reality check: can a Pi 3B do this?

**Answer: YES — measured, not estimated (M0/M1 spikes, Sept 2025, on-device).**

- **CPU:** 4× Cortex-A53 @ 1.2 GHz, running **Raspberry Pi OS Lite 64-bit (trixie)**, native arm64
  (`GOARCH=arm64`). We originally targeted 32-bit ARMv7 for memory reasons; on-device M1 results
  made that moot (see below) and arm64 removes the entire hand-rolled TFLite cross-compile.
- **Measured on the real Pi 3B** (M1 spike, native arm64, prebuilt `tflite_c` v2.17.1 linux-arm64):
  - **Inference: 0.47 s per 3-s chunk** (FP32 model) — 6.3× faster than real time. No throttling,
    57.5 °C under sustained 2-minute load.
  - **Peak RSS: 142 MiB** — including the FP32 model. (For reference, the 32-bit build measured
    143 MiB at 1.68 s/chunk — memory was never going to be the differentiator.)
  - Output parity with qemu-emulated runs: identical species and confidences.
- **Design implications (keep, but now with generous margin):**
  - The queue-drop / drop-oldest design stays as insurance against pathological load, not as a
    necessity. At 0.47 s/chunk we can afford **1-s overlap** (chunk every 2 s, still 4× real time)
    if recall matters more than CPU.
  - Use the **TFLite backend only** (`tphakala/go-tflite` + prebuilt `tphakala/tflite_c`
    linux-arm64 `.so`, exactly v2.17.1). Do NOT link ONNX Runtime or other runtimes.
  - Keep GORM buffers small; use SQLite with `PRAGMA journal_mode=WAL` and a busy timeout so the
    external querying service can read while we write.
  - Swap/zram is now optional belt-and-braces rather than a requirement; still document it.

FP32 is comfortably viable — no INT8 fallback, no duty-cycling needed.

---

## 3. What to take from BirdNET-Go

Take the *core ideas and small self-contained packages*, not the giant wired-together ones. Where a
package is too entangled, re-implement a minimal version using the upstream code as a reference.

### 3.1 Take directly (copy, lightly adapted)

| Path | What it gives us | Adaptation notes |
|---|---|---|
| `internal/datastore/entities/note.go`, `results.go` | The `notes` + `results` GORM table schema — the exact query surface | Keep schema **identical** so external tooling written against BirdNET-Go can read our DB. Drop review/comment/lock entities. |
| `internal/datastore/sqlite.go` | SQLite datastore setup, WAL, pragmas, GORM bootstrap | Trim to only `SaveNote` + simple query methods. Drop MySQL entirely. |
| `internal/conf/consts.go`, `capture_window.go`, `defaults.go` | Capture window math (3 s / overlap), sample-rate constants, BirdNET constants | Strip to the essentials. |
| `internal/detection/` | The core `Detection`/`Note` domain types passed between pipeline stages | Small package, take mostly as-is. |
| `internal/audiocore/buffer/`, `internal/audiocore/resample/` | Ring buffer + audio resampling to 48 kHz mono | Needed for the capture→analyze handoff. |
| `internal/labels/` or classifier `label_files.go` + embedded labels | The BirdNET v2.4 label list (~6,522 species) | The labels file is essential; embed it or ship alongside the model. |
| `internal/conf/range_filter.go` (concept only) | Latitude/longitude/week-based species range filtering if desired | Optional; the MData model is heavy — see §6.4. |
| `internal/analysis/processor/`, `buffer_consumer.go`, `buffer_manager.go` | The chunking pattern: fill buffer → emit overlapping 3 s chunks → queue for inference | Reference the design; likely re-implement simpler (one goroutine capturing, one analyzing, one writing). |
| `internal/datastore/detection_repository.go` | How notes/results are persisted transactionally | Reference for correct save semantics (detection + top-3 results in one tx). |

### 3.2 Take the classifier — but only the v2.4 TFLite path

`internal/classifier/` is large. Extract only:

- `birdnet.go` — BirdNET v2.4 wrapper: loading the model, input preprocessing
  (spectrogram from 3 s × 48 kHz mono PCM), the 144-value input tensor construction.
- `analyze.go` — `Filter`/prediction filtering: threshold, "human", "dog", etc. exclusion, top-N
  selection. Take `pairLabelsAndConfidence` and the confidence gating logic.
- `overlap.go` — chunk overlap handling.
- **Backend:** `tflite_available.go` + the go-tflite calls in `model.go` that actually invoke
  the interpreter. Delete everything ONNX (`model_onnx.go`, `birdnet_v3_onnx.go`,
  `orchestrator_*` variants for perch/bat/v3, `model_openvino.go`, catalog/registry/manager
  machinery). The "model manager / orchestrator / multi-model registry" does not apply to us:
  we have exactly one model, loaded once at startup.

**Model + labels files:** BirdNET v2.4 TFLite model (`BirdNET_GLOBAL_6K_V2.4_M_Data1.1_FP32.tflite`)
and its labels — these are CC-BY-NC-SA licensed like the code; the new project must carry the same
license attribution (see `LICENSE`, `LICENSES.md` in `../birdnet-go`). They are NOT committed in
the repo (downloaded / embedded at build); check `../birdnet-go/internal/classifier/models_embedded.go`
and the release assets.

### 3.3 Audio capture — reference, probably re-implement

Upstream uses `gen2brain/malgo` (miniaudio) via `internal/audiocore/capture.go` with ALSA backend.
On a Pi 3B you have two options:

- **Keep malgo** (`malgo.BackendAlsa`, device `sysdefault`/`default`): least code, works with USB
  mics. Take `capture.go` + `device.go` + `source.go` as reference and strip to one device.
- **Simpler alternative:** open ALSA directly (`github.com/cocoonlife/talsa` or `malgo` alone with
  a 15–30 ms callback) — the upstream `audiocore` package is heavily engineered (router, liveness,
  stream manager, encoders for aac/mp3/opus/flac, equalizer, sound-level) and almost none of that
  is needed. Copying just the malgo init pattern into ~100 lines is likely easier than untangling
  `audiocore`. **Recommendation: re-implement, using `internal/audiocore/capture.go` as reference.**

Audio format needed by the model: 48 kHz, mono, float32 PCM, 3 s chunks (144,000 samples) with
configurable overlap (upstream default 0.0–0.99, commonly 0.5 → chunks every 1.5 s; on Pi 3 use
lower overlap like 0.33 to control CPU).

### 3.4 Do NOT take

- `frontend/`, `ui/`, `internal/api/` — all web.
- `internal/notification/`, `internal/birdweather/`, `internal/ebird/`,
  `internal/weather/`, `internal/alerting/` — integrations. **However**, take
  `internal/mqtt/` as a *reference* for F7 (MQTT/Kafka event emission) — it shows correct
  paho usage (broker address parsing, TLS, reconnect handling, publish suppression while
  disconnected). We only need a ~50-line subset. Upstream's MQTT is publish-only
  (detections out, e.g. Home Assistant discovery); ours is the same direction.
- `internal/mqtt/` full client — covered above; reference only.
- `internal/classifier/` everything for Perch, bat, v3, ONNX, OpenVINO, model manager/registry.
- `internal/observability/`, `internal/telemetry/`, `internal/health/`, `internal/monitor/`,
  `internal/diagnostics/`, `internal/support/`, `internal/update/`, `internal/backup/`,
  `internal/security/`, `internal/tls/`, `internal/secrets/`, `internal/serviceapi/`.
- `internal/spectrogram/`, `internal/imageprovider/` (unless you want clip spectrograms — no).
- `internal/diskmanager/` — but steal its *idea*: a simple retention loop that prunes old clips.
- Multi-user anything, sessions (`gorilla/sessions`, goth OIDC), Echo web framework, testcontainers.

---

## 4. Target architecture

```
┌────────────┐    ┌──────────────┐    ┌────────────┐    ┌─────────────┐
│ ALSA mic   │───▶│ chunker: 3 s │───▶│ BirdNET    │───▶│ SQLite      │
│ (malgo,    │    │ float32 mono │    │ v2.4 TFLite│    │ notes table │
│ 48 kHz)    │    │ + overlap    │    │ (1 worker) │    │ + clips/    │
└────────────┘    └──────────────┘    └────────────┘    └─────────────┘
                                            │                  ▲
                                     threshold + dog/cat/  ─────┬──────────┐
                                     human filter               │          ▼
                                                     optional: MQTT/Kafka  external service
                                                     event emitter (F7)   reads DB directly
```

Three goroutines and a small queue. If the queue is full, **drop the oldest chunk** (audio
continuity is less important than not falling behind — upstream learned this the hard way; see
`process_queue_drop_test.go`).

Config: a single YAML or TOML file — device name, latitude/longitude (for range filtering and
records), confidence threshold, overlap, clip recording on/off + retention days, DB path. No
hot-reload, no wizard. SIGHUP to restart capture is a nice-to-have.

---

## 5. PRD

### 5.1 Overview

**Product:** `twitcher` — a single static Go binary, headless, systemd-managed on
Raspberry Pi OS Lite (64-bit, trixie), that continuously detects birds and records them to a local SQLite DB.

**Non-goals:** any UI, any network listener, multi-model support, anything requiring >300 MB RSS
outside inference.

### 5.2 Functional requirements

- **F1 Audio capture.** Capture from a configured ALSA device at 48 kHz mono (resample if device
  doesn't support it). Recover automatically from device errors (reopen with backoff).
- **F2 Detection.** Classify each 3 s window with BirdNET v2.4; record a detection when confidence
  ≥ threshold (default 0.80, configurable). Exclude non-bird labels (Human, Dog, Cat, etc.) per
  upstream's exclusion list. Store the top-3 species+confidences per detection in `results`.
- **F3 Persistence.** Each detection = one row in `notes` (schema-compatible with BirdNET-Go's
  `NoteEntity`: Date, Time, BeginTime, EndTime, ScientificName, CommonName, Confidence, lat/lon,
  Threshold, Sensitivity, ProcessingTime) plus child rows in `results`. SQLite in WAL mode.
- **F4 Audio clips (optional, default off).** When enabled, save the 3 s window as FLAC/PCM to
  `clips/YYYY/MM/DD/` and set `ClipName`. Retention sweep deletes clips (and optionally notes)
  older than N days or when free space drops below a floor.
- **F5 Range filter (optional).** If lat/lon configured, optionally filter by species plausibility
  using the embedded label metadata — see §6.4 for the cost/benefit; MVP may skip this.
- **F6 Query surface.** No API. The database is the interface. Document the schema in the README.
  WAL mode + busy_timeout so external readers never block the writer.
- **F7 Event emission (optional, default off).** After a detection is committed to SQLite,
  publish a JSON payload to configured sinks. Sinks: **MQTT broker** (configurable topic, e.g.
  `birdnet/detections`, QoS 1, optional retain) and/or **Kafka** (configurable topic, key by
  species or node). Payload = flat JSON mirroring the note row (`begin_time, end_time,
  species_code, scientific_name, common_name, confidence, latitude, longitude, threshold,
  sensitivity, clip_name`) plus the top-3 `results` array. Requirements:
  - Emission must never block or crash the pipeline: one bounded internal queue + a single
    emitter goroutine per sink. If a broker is down, drop (default) or buffer-and-flush
    (configurable) — the DB is the source of truth, events are best-effort.
  - Reconnect with backoff; suppress publishes while disconnected (mirror upstream
    `internal/mqtt` behavior).
  - MQTT last-will message on `birdnet/status` ("offline") is a cheap nice-to-have.
  - No inbound subscription, no Home Assistant discovery — emit only.
  - Libraries: `eclipse/paho.mqtt.golang` for MQTT; `twmb/franz-go` for Kafka (pure Go, no
    CGO, low memory — preferable to segmentio/kafka-go on a Pi 3).
- **F8 Operability.** Structured logging to stdout/journald. systemd unit file shipped. Config via
  file + a couple of env overrides. Clean shutdown on SIGTERM (flush queue, close DB, free model).

### 5.3 Non-functional requirements

- **N1 Memory:** measured peak RSS is 142 MiB during inference (M1, FP32); budget < 400 MB
  worst case including SQLite and buffers; steady-state allocator trimmed
  (`GOGC`/`GOMEMLIMIT` set in code or unit file; preallocate the input/output buffers once and
  reuse).
- **N2 CPU:** measured 0.47 s per 3-s chunk (6.3× real-time headroom); the chunker must still
  drop backlog rather than grow the queue unboundedly (defensive design under pathological load).
- **N3 Storage:** with clips off, ~1–2 KB per detection; DB growth must be bounded by the
  retention sweep.
- **N4 Platform:** cross-compile with `GOOS=linux GOARCH=arm64`, CGO required
  (go-tflite + sqlite). Build uses the **prebuilt** `tphakala/tflite_c` v2.17.1 linux-arm64
  `libtensorflowlite_c.so` — no TFLite source compile. Container build (ubuntu:24.04,
  `gcc-aarch64-linux-gnu libc6-dev-arm64-cross golang-go`):
  `CGO_ENABLED=1 GOOS=linux GOARCH=arm64 CC=aarch64-linux-gnu-gcc CGO_CFLAGS="-I<tensorflow-src>" CGO_LDFLAGS="-L build/arm64 -ltensorflowlite_c -ldl -lrt" go build`
  (the `CGO_CFLAGS` include of the tensorflow source tree is only needed for go-tflite's headers).
- **N5 License:** code + model are CC-BY-NC-SA 4.0 (non-commercial). Carry attribution from
  BirdNET-Go and the BirdNET model authors. State this in the project README.

### 5.4 Milestones

1. **M1 — Offline classify:** ✅ **DONE.** `cmd/m1/main.go`: read a WAV, BirdNET v2.4 TFLite FP32,
   top-3 per 3-s chunk. Verified on the Pi 3B (native arm64): tawnyowl.wav → *Strix aluco* @
   1.000 on all windows; soundscape.wav → plausible species matching upstream expectations.
   0.47 s/chunk, 142 MiB peak RSS, no throttling. Artifacts in `build/arm64/`. (Earlier armv7
   32-bit spike in `build/armv7/` + `scripts/m0_cross_tflite.sh` is superseded — kept as
   historical reference only.)
2. **M2 — Stream + detect:** live ALSA capture → chunker → inference → log detections.
3. **M3 — Persist:** SQLite `notes`/`results`, BirdNET-Go-compatible schema.
4. **M4 — Harden:** queue-drop, device reconnect backoff, retention sweep, systemd unit,
   SIGHUP re-read config, clip recording, MQTT/Kafka event emitter (F7).
5. **M5 — Pi deployment:** swap/zram docs, boot-time startup, memory soak test for 24 h.

### 5.5 Verification

- M1: classify `../birdnet-go/soundscape.wav` / `tawnyowl.wav` from the repo; expect sensible species.
- M3: point the *upstream BirdNET-Go schema docs* at the DB file — external tooling must read it.
- M5: 24 h soak on real hardware: RSS stable, no queue growth, DB readable by external service,
  detection rate plausible for the location.

---

## 6. Notes, pitfalls, and open decisions for the build agent

1. **`internal/errors` and `internal/logger`** are used by nearly everything you copy. Either take
   them (they're small-ish) or strip their usage to plain `log`/`fmt` while extracting. Decide once,
   early.
2. **GORM vs raw SQL:** upstream uses GORM for notes. On a Pi 3, GORM is fine for ~1 insert/sec;
   keep it for schema compatibility and index parity (see the gorm tags on `NoteEntity` —
   replicate the indexes).
3. **go-tflite build:** use the **prebuilt** `tphakala/tflite_c` release `libtensorflowlite_c.so`
   (linux-arm64, v2.17.1 — matches the pinned TFLite version); do NOT rebuild TFLite from source.
   Link with `CGO_LDFLAGS=-L<dir> -ltensorflowlite_c -ldl -lrt`. On the target the `.so` can live
   in `/usr/local/lib` + `ldconfig`, or ship it beside the binary with `LD_LIBRARY_PATH`/rpath.
   (The prebuilt arm64 `.so` is significantly faster than a hand-rolled armv7 build with
   optimizations disabled: 0.47 vs 1.68 s/chunk on the same board.)
4. **Range filter cost:** upstream's range filter uses a second TFLite MData model. On 1 GB RAM,
   a second model in memory is risky. MVP: skip range filtering, or use it only to *annotate*
   (`Unlikely` flag exists in the schema for exactly this) without a second model by using the
   static label metadata where possible.
5. **Don't copy `internal/analysis` wholesale** — it wires in 15+ subsystems (monitor, SSE,
   telemetry, image cache, eBird, jobqueue). Copy the *shape* (buffer_manager → consumer →
   processor) into ~300 lines.
6. **Time-of-day / week:** the schema stores Date/Time strings and BeginTime; keep upstream's
   semantics so external dashboards behave.
7. **Check current upstream README for supported hardware claims** and the
   [hardware wiki](https://github.com/tphakala/birdnet-go/wiki/hardware) before assuming anything
   about model memory; benchmark early (M1 is the gate).
