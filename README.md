# hms-detection

Real-time RTSP capture and YOLO object detection service for security cameras. Built with C++20, FFmpeg, and Drogon for high performance and low memory usage.

## Features

- RTSP capture via FFmpeg libav* with per-camera decode threads
- Pre-allocated frame pool with zero-copy recycling (no malloc churn)
- Ring buffer per camera with lock-free concurrent reads
- Live JPEG snapshots via `/api/cameras/{id}/snapshot`
- Health monitoring via `/health` with per-camera stats
- Automatic reconnection with exponential backoff
- ~6 threads total for 3 cameras (vs Python's multiprocessing overhead)

## Architecture

```
Cameras (RTSP)
    │
    ▼
go2rtc (restream)  ← single connection per camera
    │
    ▼
hms-detection      ← FFmpeg decode → ring buffer → HTTP API
    │
    ├── GET /health                          → JSON status
    ├── GET /api/cameras/{id}/snapshot       → JPEG image
    └── (future) MQTT motion events → Home Assistant
```

### Thread Model

```
Main thread → Drogon HTTP server
├── Drogon IO (2 threads)     → /health + /snapshot handlers
├── Capture: camera_1         → av_read_frame → decode → ring buffer
├── Capture: camera_2         → same
└── Capture: camera_N         → same
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
```

## Container Image

```
docker pull ghcr.io/hms-homelab/hms-detection:latest
```

```bash
docker run -d \
  --network host \
  -v ./config.yaml:/app/config/config.yaml:ro \
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
  libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

# Build
cmake -S yolo_detection_cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target hms_detection

# Run
./build/services/detection/hms_detection --config config.yaml
```

## Configuration

See [`config.yaml.example`](config.yaml.example) for all options. Key sections:

| Section | Description |
|---------|-------------|
| `cameras` | RTSP URLs, enabled classes, confidence thresholds |
| `buffer` | Ring buffer size (`preroll_seconds * fps`) |
| `api` | Listen host/port (default `0.0.0.0:8000`) |
| `logging` | Log level, file rotation |

## API Endpoints

### `GET /health`

Returns JSON with per-camera capture stats:

```json
{
  "service": "hms-detection",
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

## Roadmap

- [ ] **Phase 2** (current): RTSP capture + ring buffers + health/snapshot API
- [ ] **Phase 3**: YOLO inference (ONNX Runtime / TensorRT)
- [ ] **Phase 4**: Event recording (FFmpeg mux from ring buffer pre-roll)
- [ ] **Phase 5**: MQTT client (motion events to Home Assistant)
- [ ] **Phase 6**: Database logging (PostgreSQL event storage)
