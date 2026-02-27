# Changelog

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
