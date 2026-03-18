# Changelog

## v2.14.0 (2026-03-17)

### Added
- **YOLO26 end-to-end output support**: Auto-detects ONNX output format at runtime — `[1, N, 6]` (YOLO26 e2e, pre-NMS'd) vs `[1, 84, 8400]` (YOLO11 raw candidates). New `postprocessE2E()` consumes YOLO26's built-in NMS output directly, skipping the manual per-class NMS pass.
- Backward compatible — both YOLO11 and YOLO26 `.onnx` models work without config changes, just swap the file.

### Changed
- **Upgraded default model to YOLO26m**: +1.6 mAP over YOLO11m on COCO (53.1 vs 51.5) with identical GPU inference time and VRAM footprint. Better small-object detection via STAL (Small-Target-Aware Label Assignment).

## v2.10.0 (2026-03-11)

### Added
- **Periodic snapshot processing**: Re-enabled ambient scene snapshots with moondream vision model (1.8B Q4, ~1.8GB VRAM) — small enough to coexist with YOLO on the RTX 3050 6GB
- **GpuCoordinator**: Shared coordination between EventManager and PeriodicSnapshotManager — events always get GPU priority, periodic moondream inference aborts within ~1s via curl progress callback
- **VisionClient abort support**: `analyze()` accepts an `abort_flag` (atomic bool) checked by libcurl progress callback — enables clean mid-inference cancellation
- **VisionClient::forceUnloadModel()**: Static method to explicitly evict a model from Ollama VRAM
- **`periodic_vision` config section**: Separate vision model config for periodic snapshots (model, endpoint, prompts, timeout) — decoupled from event LLaVA config

### Changed
- **Embedding client**: Added `keep_alive: 0` to Ollama embed requests so nomic-embed-text unloads from VRAM immediately after use
- **PeriodicSnapshotManager**: Now accepts GpuCoordinator — skips moondream if event is active, aborts mid-inference if event fires, always saves JPEG regardless
- **EventManager**: Signals GpuCoordinator on event start/finish — triggers periodic moondream abort on start, clears flag on finish

## v2.9.0 (2026-03-08)

### Changed
- **On-demand GPU lifecycle**: YOLO ONNX session now loads onto GPU only during motion events and unloads immediately after detection completes — GPU sits idle (~80MiB) instead of permanently holding the model (~1.2GB). Eliminates VRAM contention with Ollama/LLaVA.
- **LLaVA auto-unload**: Ollama requests now include `keep_alive=0`, so the LLaVA model is unloaded from GPU immediately after inference — no more stale model blocking CUDA EP.

### Fixed
- **LLaVA HTTP 500 errors**: Ollama returned HTTP 500 when trying to load LLaVA (~4.9GB) while YOLO held ~1.2GB on a 6GB GPU. Now YOLO unloads before LLaVA runs, giving it full GPU access.
- **False animal detections**: Removed continuous 24/7 background DetectionWorker loops that were never intended to run (main.cpp already skipped `startDetection()`). YOLO now only runs during motion events as designed.

### Added
- `DetectionEngine::load()` / `unload()` with mutex for thread-safe on-demand GPU session management
- 19 GPU lifecycle unit tests: concurrent load/unload, deadlock prevention, stress tests, edge cases

## v2.8.1 (2026-03-06)

### Fixed
- **RTSP reconnection deadlock**: once a stream went stale (30s without frames), every subsequent reconnect was immediately aborted by the interrupt callback — `last_activity_time_` still held the old stale timestamp during `avformat_open_input()`, so the callback saw `elapsed > 30s` and returned "Immediate exit requested". Fix: reset `last_activity_time_` before opening the new connection, not only after.

## v2.8.0 (2026-03-04)

### Fixed
- **MIME type for video files**: Drogon's `newFileResponse()` served `.mp4` files as `application/octet-stream` — browsers refused to play video without correct `video/mp4` content-type. Added `getMimeType()` helper for `/events/` and `/snapshots/` routes.
- **Recording bitrate cap**: added `maxrate=1000k` / `bufsize=2000k` to H.264 encoder so 30s recordings stay under ~4MB (HA ingress proxy size limit). Patio recordings were 5-6MB at CRF 28 with no cap.

### Added
- **Periodic snapshot manager**: ambient scene snapshots every 5 minutes per camera, saved to DB with optional LLaVA descriptions
- **Embedding client**: vector embedding support for AI context text

## v2.7.1 (2026-03-04)

### Fixed
- **Stale RTSP stream detection**: cameras that kept TCP alive but stopped sending video data caused `av_read_frame()` to block indefinitely — live view would freeze for hours with no reconnect
- Added 30-second stale-stream timeout to FFmpeg interrupt callback; forces reconnect when no frames arrive within the window

## v2.7.0 (2026-03-02)

### Changed
- **Confidence escalation**: system keeps re-checking YOLO on each frame until best confidence meets the gate — previously gave up after first below-gate detection and never re-checked
- **Snapshot from best frame**: snapshot saved from highest-confidence frame when gate is crossed, not the first low-confidence detection
- **Stop inference after notification**: YOLO inference stops once notification is sent — saves GPU cycles for the rest of the event (recording continues)
- **Post-roll gate fix**: post-roll phase now uses same escalation logic (was still using old fire-once-on-first-detection pattern)
- **Quieter logs**: no per-frame detection logs while below gate, no inference logs after notification sent

### Added
- 16 Catch2 unit tests for confidence gate (`[confidence_gate]` tag): escalation across frames, best-frame tracking, single-publish guarantee, multi-camera gates, recording deletion
- E2E test expanded to 5 scenarios: above gate, below gate, escalation (gate=0.78), narrow miss (gate=0.88), no class match

## v2.6.0 (2026-03-02)

### Changed
- **Result MQTT gated by confidence**: early `/result` only published when best detection meets `immediate_notification_confidence` gate (default 0.70) — no more HA notifications for low-confidence detections
- **Recordings deleted for low-confidence events**: events below the notification gate still save snapshots and log to DB, but recordings are removed to save disk space

### Added
- E2E test (`e2e_confidence_gate_test.py`): two-scenario test using live front_door camera with real car detections (~86%) — validates high-confidence publishes result+context and low-confidence suppresses both
- 5 Catch2 unit tests for confidence gate logic (`[confidence_gate]` tag)

## v2.5.0 (2026-03-02)

### Changed
- **Simplified MQTT**: reduced from 6 messages per event to 2 — one `result` (with `snapshot_url`) and one `context` (LLaVA description)
- **Snapshot before publish**: snapshot is now saved before the result message so `snapshot_url` is included from the start
- **Silent on 0 detections**: no MQTT, no snapshot, no DB write, recording file deleted — zero noise for HA

### Removed
- `/detection` started/completed status messages
- `/detected` ON/OFF binary sensor messages
- `phase` field from result payload (only one result now, no early/final distinction)
- Final result message (duplicate of early result with extra stats)
- 2-second `detected OFF` delay at end of event

## v2.4.0 (2026-03-02)

### Added
- **GPU inference**: config-driven CUDA Execution Provider (~25ms vs ~300ms CPU)
- **Graceful CPU fallback**: try/catch around CUDA EP registration; falls back to CPU if unavailable
- **`gpu` CMake preset**: bare-metal GPU builds with CUDA EP support
- **`gpu_enabled` config field**: new `detection.gpu_enabled` option (default: `false`)

## v2.3.0 (2026-02-28)

### Added
- **Early notification**: MQTT `detected ON` and result published immediately on first YOLO detection (~300ms after motion start, was ~46s)
- **Parallel LLaVA**: vision context runs in a background thread concurrent with recording, instead of blocking after finalization
- **Early snapshot**: snapshot saved at first detection for LLaVA input, not deferred to end of recording
- **Pipeline timing traces**: detailed `spdlog::info` for every phase — live start, each YOLO inference (first 3 + detections), early notification, post-roll duration, finalization, final MQTT publish, LLaVA join

### Changed
- MQTT result messages now include `"phase": "early"` or `"phase": "final"` to distinguish immediate vs complete results
- Final `detected ON` skipped if early notification already sent (avoids duplicate HA triggers)
- Snapshot only re-saved at finalization if no early snapshot exists

## v2.2.0 (2026-02-27)

### Changed
- **Build output relocated**: binary now builds to `build/` at project root instead of `yolo_detection_cpp/build/`
- **Install directory**: service runs from `/opt/hms-detection/` (binary, config, model, logs)
- **CMake presets**: removed vcpkg presets, `system-deps` is now the default (Release mode)
- **Systemd service**: `WorkingDirectory` and `ExecStart` point to `/opt/hms-detection/`

### Added
- `install.sh`: post-build script that copies binary, config, and ONNX model to `/opt/hms-detection/`

### Removed
- vcpkg build presets (using system-installed Debian packages only)
- Timeline-related build targets from CMakePresets (moved to hms-timeline repo)

## v2.1.0 (2026-02-26)

### Fixed
- **Deadlock on concurrent events**: VisionClient replaced Drogon HttpClient with libcurl for Ollama/LLaVA calls, fully decoupling from Drogon's event loop
- **DB pool exhaustion hang**: `DbPool::acquire()` now times out after 10s instead of blocking forever
- **Orphaned thread blocking**: `joinOrphanedThreads()` now detaches instead of blocking the MQTT callback thread
- **MQTT publish blocking**: Non-retained messages forced to QoS 0 (true fire-and-forget)

### Added
- HTTP endpoints for `/snapshots/{filename}` and `/events/{filename}` static file serving
- Absolute URLs in MQTT result/context messages (`http://192.168.2.5:8000/snapshots/...`)
- Path traversal protection on file serving endpoints
- try/catch around DB logging and LLaVA calls to prevent event thread hangs

### Changed
- Drogon IO threads increased from 2 to 4
- libcurl added as dependency for VisionClient

## v2.0.0

- Initial C++ rewrite of YOLO detection service
- RTSP capture with ring buffers and frame pool
- ONNX Runtime inference (yolo11s)
- FFmpeg-based event recording and snapshot writing
- MQTT-triggered motion events with preroll
- LLaVA vision context via Ollama
- PostgreSQL event logging
- Drogon HTTP API (health, detection, annotated snapshots)
