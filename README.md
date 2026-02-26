# hms-detection v2.0.0

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-support-%23FFDD00.svg?logo=buy-me-a-coffee)](https://www.buymeacoffee.com/aamat09)
[![Build](https://github.com/hms-homelab/hms-detection/actions/workflows/build.yml/badge.svg)](https://github.com/hms-homelab/hms-detection/actions)

Real-time YOLO object detection, event recording, and AI vision context for security cameras. Built with C++20, FFmpeg, ONNX Runtime, and Drogon.

## Features

- **RTSP capture** via FFmpeg libav* with per-camera decode threads
- **Pre-allocated frame pool** with zero-copy recycling (no malloc churn)
- **Ring buffer** per camera with lock-free concurrent reads
- **YOLO inference** via ONNX Runtime with configurable classes and confidence thresholds
- **Event recording** with pre-roll/post-roll (FFmpeg muxer from ring buffer)
- **Snapshot capture** of highest-confidence detection frame with bounding boxes
- **MQTT integration** for Home Assistant motion events and detection results
- **LLaVA vision context** via Ollama for natural-language scene descriptions
- **PostgreSQL logging** of events, detections, and AI context
- **Health monitoring** via `/health` with per-camera stats
- **Live JPEG snapshots** via `/api/cameras/{id}/snapshot`
- Automatic reconnection with exponential backoff

## Architecture

```
Cameras (RTSP)
    |
    v
go2rtc (restream)  <-- single connection per camera
    |
    v
hms-detection      <-- RTSP decode --> ring buffer --> YOLO --> event pipeline
    |
    |-- MQTT motion/start  -->  detect + record + snapshot
    |-- MQTT result        <--  detections, recording URL, snapshot URL
    |-- MQTT context       <--  LLaVA natural-language description
    |-- PostgreSQL         <--  events, detections, ai_vision_context
    |
    |-- GET /health                      --> JSON status
    +-- GET /api/cameras/{id}/snapshot   --> JPEG image
```

### Full Event Pipeline

```
MQTT motion/start
    |
    v
Ring Buffer (preroll frames)
    |
    v
Event Thread (per camera)
    |-- Start FFmpeg recorder (preroll + live frames)
    |-- YOLO detection sampling (every 3rd frame)
    |-- Post-roll recording
    |-- Finalize MP4
    |-- Save best-frame snapshot with bounding boxes
    |-- Publish MQTT result (detections, URLs)
    |-- Log to PostgreSQL
    |-- LLaVA vision analysis (if confidence gate met)
    +-- Publish MQTT context
```

### Thread Model

```
Main thread --> Drogon HTTP server
|-- Drogon IO (2 threads)     --> /health + /snapshot handlers
|-- Capture: camera_1         --> av_read_frame --> decode --> ring buffer
|-- Capture: camera_2         --> same
|-- Capture: camera_N         --> same
|-- Event: camera_X           --> spawned on motion/start, joins on completion
+-- MQTT client               --> async publish/subscribe
```

## Quick Start

```bash
# 1. Copy and edit config
cp config.yaml.example config.yaml

# 2. Run with docker-compose (includes go2rtc)
docker compose up -d

# 3. Check health
curl http://localhost:8000/health | python3 -m json.tool

# 4. Get a snapshot
curl http://localhost:8000/api/cameras/patio/snapshot -o snapshot.jpg

# 5. Trigger detection via MQTT
mosquitto_pub -h localhost -t "camera/event/motion/start" \
  -m '{"camera_id": "patio", "post_roll_seconds": 5}'
```

## Container Image

```
docker pull ghcr.io/hms-homelab/hms-detection:latest
```

```bash
docker run -d \
  --network host \
  -v ./config.yaml:/app/config/config.yaml:ro \
  -v /mnt/ssd/events:/mnt/ssd/events \
  -v /mnt/ssd/snapshots:/mnt/ssd/snapshots \
  ghcr.io/hms-homelab/hms-detection:latest
```

## Building from Source

```bash
# Install dependencies (Debian/Ubuntu)
sudo apt install build-essential cmake ninja-build pkg-config \
  libdrogon-dev libtrantor-dev libpqxx-dev libpq-dev \
  libyaml-cpp-dev libjsoncpp-dev libspdlog-dev libfmt-dev \
  nlohmann-json3-dev uuid-dev libbrotli-dev libsqlite3-dev \
  libhiredis-dev default-libmysqlclient-dev libssl-dev libkrb5-dev \
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
  libonnxruntime-dev libpaho-mqttpp-dev

# Build
cmake -S yolo_detection_cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target hms_detection

# Run tests
cmake --build build --target detection_tests shared_tests
ctest --test-dir build --output-on-failure

# Run
./build/services/detection/hms_detection --config config.yaml
```

## Configuration

See [`config.yaml.example`](config.yaml.example) for all options.

| Section | Description |
|---------|-------------|
| `cameras` | RTSP URLs, enabled classes, confidence thresholds per camera |
| `buffer` | Ring buffer size (`preroll_seconds * fps`), max memory |
| `detection` | ONNX model path, confidence/IOU thresholds, class filter |
| `mqtt` | Broker host/port, credentials, topic prefix |
| `database` | PostgreSQL host/port, credentials, connection pool size |
| `llava` | Ollama endpoint, model, per-camera prompts, timeout |
| `timeline` | Timeline UI settings (events/snapshots dirs, CORS) |
| `api` | Listen host/port (default `0.0.0.0:8000`) |
| `logging` | Log level, file rotation |

## API Endpoints

### `GET /health`

Returns JSON with per-camera capture stats, MQTT status, and detection engine status.

```json
{
  "service": "hms-detection",
  "version": "2.0.0",
  "status": "healthy",
  "cameras": {
    "patio": {
      "is_connected": true,
      "is_healthy": true,
      "buffer_size": 75,
      "max_frames": 75,
      "frames_captured": 12345,
      "frame_width": 640,
      "frame_height": 480,
      "last_frame_ms_ago": 42,
      "reconnect_count": 0,
      "consecutive_failures": 0
    }
  }
}
```

### `GET /api/cameras/{camera_id}/snapshot`

Returns the latest captured frame as a JPEG image.

## MQTT Topics

| Topic | Direction | Description |
|-------|-----------|-------------|
| `camera/event/motion/start` | Subscribe | Trigger detection for a camera |
| `camera/event/motion/stop` | Subscribe | Signal end of motion event |
| `yolo_detection/{cam}/detection` | Publish | Event status (started/completed) |
| `yolo_detection/{cam}/result` | Publish | Detection results, recording/snapshot URLs |
| `yolo_detection/{cam}/detected` | Publish | Binary ON/OFF for Home Assistant |
| `yolo_detection/{cam}/context` | Publish | LLaVA natural-language scene description |
| `yolo_detection/status` | Publish | Service online/offline (retained) |

## Roadmap

- [x] **Phase 1**: Project structure, CMake, shared libraries
- [x] **Phase 2**: RTSP capture, ring buffers, health/snapshot API
- [x] **Phase 3**: YOLO inference (ONNX Runtime)
- [x] **Phase 4**: Event recording (FFmpeg mux), MQTT client, DB logging
- [x] **Phase 5**: LLaVA vision context (Ollama integration)
- [x] **Phase 6**: Unit tests, bug fixes, versioning, README
---

## ☕ Support

If this project is useful to you, consider buying me a coffee!

[![Buy Me A Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/aamat09)
