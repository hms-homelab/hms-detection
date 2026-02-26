# Phase 4: MQTT + Event Recording + Snapshot Saving

## Overview

Phase 4 adds MQTT-triggered event recording with pre-roll, snapshot saving,
detection result publishing, and DB logging — making the C++ service a full
replacement for the Python `api_server.py` + `mqtt_client.py` + `event_recorder.py` stack.

## Components

### MqttClient (`shared/mqtt/`)
Paho MQTT C++ async_client wrapper:
- Auto-reconnect with exponential backoff (1s–64s)
- Fire-and-forget publish (no `token->wait()` — deadlock-safe from callbacks)
- Batch subscribe (single call for all topics)
- LWT: publishes "offline" on unexpected disconnect
- Topic wildcard matching (`+`, `#`)

### EventRecorder (`services/detection/`)
Direct H.264 MP4 encoding via libavcodec/libavformat (no FFmpeg subprocess):
- BGR24 → YUV420P → H.264 (ultrafast, CRF 28, faststart)
- Pre-roll: writes buffered frames as start of recording
- Post-roll: configurable seconds after motion stop
- Max duration: 30 seconds
- Output: `/mnt/ssd/events/{camera_id}_{YYYYMMDD_HHMMSS}.mp4`

### SnapshotWriter (`services/detection/`)
Saves annotated JPEG snapshots:
- Draws bounding boxes on best-confidence detection frame
- MJPEG encoding via FFmpeg
- Output: `/mnt/ssd/snapshots/{camera_id}_{YYYYMMDD_HHMMSS}.jpg`

### EventManager (`services/detection/`)
Full event orchestration triggered by MQTT:
1. Motion start → subscribe to `camera/event/motion/start`
2. Grab 75 preroll frames from ring buffer (deep-copy, release pool refs)
3. Start H.264 recording with preroll
4. Run YOLO detection (sample every 3rd frame)
5. Track best-confidence frame for snapshot
6. Motion stop → 5s post-roll
7. Finalize MP4 (faststart)
8. Save annotated snapshot
9. Publish detection results to MQTT
10. Log to PostgreSQL (detection_events + detections tables)

### EventLogger (`shared/db/`)
PostgreSQL event logging:
- `create_event()` — INSERT into detection_events
- `complete_event()` — UPDATE with duration, detection count
- `log_detections()` — INSERT individual detections

## MQTT Topics

### Subscribe
- `camera/event/motion/start` — `{"camera_id":"patio","post_roll_seconds":5}`
- `camera/event/motion/stop` — `{"camera_id":"patio"}`

### Publish
- `yolo_detection/{cam}/detection` — `{"status":"started"|"completed"}`
- `yolo_detection/{cam}/result` — full detection summary (matches Python schema)
- `yolo_detection/{cam}/detected` — `ON`/`OFF` binary sensor for HA
- `yolo_detection/status` — `online`/`offline` (retained, with LWT)

## Architecture

```
main()
  → load config
  → BufferService: start RTSP capture for all cameras (ring buffers)
  → BufferService: load ONNX model (no continuous workers)
  → MqttClient: connect (auto-reconnect, graceful degradation)
  → DbPool: create connection pool (graceful degradation)
  → EventManager: subscribe to motion topics
  → Drogon: app.run() (blocking HTTP server)
  → Shutdown: EventManager stop → MQTT offline → disconnect
```

MQTT and DB failures don't prevent HTTP from serving — `/health` reports "degraded".

## Thread Model

```
Main thread → drogon::app().run()
├── Drogon IO pool (2 threads)
├── RTSP capture: patio, side_window, front_door (3 threads)
├── MQTT client (1 Paho internal thread)
└── Event threads: 0-3 (spawned on motion, one per camera)
```

No continuous detection workers — YOLO inference only during events.

## Dependencies

Build: `libpaho-mqttpp-dev libpaho-mqtt-dev`
Runtime: `libpaho-mqttpp3-1 libpaho-mqtt1.3`

## Key Bug Fixes

### Frame Pool Exhaustion
- **Problem**: EventManager held `shared_ptr<FrameData>` from the pool for preroll frames, starving the capture thread
- **Fix**: Deep-copy preroll pixel data immediately, release pool references. Also increased pool headroom from 15 to 30 frames.

### Detection Mode
- **Problem**: Phase 3 continuous detection workers ran inference on all cameras 24/7
- **Fix**: Model loaded on startup but workers not started. Detection only runs during motion events inside EventManager.

## Verification

```bash
# Stop Python service
sudo systemctl stop yolo-detection

# Run C++ service
cd /opt/yolo_detection/hms-detection
./yolo_detection_cpp/build/services/detection/hms_detection --config config.yaml

# Monitor MQTT
mosquitto_sub -h 192.168.2.15 -u aamat -P exploracion -t "yolo_detection/#" -v

# Trigger motion
mosquitto_pub -h 192.168.2.15 -u aamat -P exploracion \
  -t "camera/event/motion/start" \
  -m '{"camera_id":"front_door","post_roll_seconds":5}'

# Wait, then stop
mosquitto_pub -h 192.168.2.15 -u aamat -P exploracion \
  -t "camera/event/motion/stop" \
  -m '{"camera_id":"front_door"}'

# Check output
ls -lt /mnt/ssd/events/front_door_*
ls -lt /mnt/ssd/snapshots/front_door_*

# Restore Python service
sudo systemctl start yolo-detection
```
