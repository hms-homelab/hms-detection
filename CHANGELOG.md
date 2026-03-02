# Changelog

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
