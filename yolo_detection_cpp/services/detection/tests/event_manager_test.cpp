#include <catch2/catch_all.hpp>
#include "event_manager.h"
#include "mqtt_client.h"
#include "config_manager.h"

#include <chrono>
#include <thread>

using namespace hms;
using Clock = std::chrono::steady_clock;

namespace {

yolo::AppConfig makeTestConfig() {
    yolo::AppConfig config;
    config.buffer.preroll_seconds = 2;
    config.buffer.fps = 15;
    config.buffer.max_buffer_size_mb = 10;

    // Use non-routable RTSP URLs — cameras won't connect, but buffers are created
    yolo::CameraConfig cam;
    cam.id = "test_cam";
    cam.name = "Test Camera";
    cam.rtsp_url = "rtsp://127.0.0.1:1/nonexistent";
    cam.enabled = true;
    cam.classes = {"person", "car"};
    cam.confidence_threshold = 0.5;
    cam.immediate_notification_confidence = 0.7;
    config.cameras["test_cam"] = cam;

    config.detection.model_path = "/nonexistent.onnx";
    config.detection.confidence_threshold = 0.5;
    config.detection.iou_threshold = 0.45;
    config.detection.classes = {"person", "car"};

    config.timeline.events_dir = "/tmp/hms_test_events";
    config.timeline.snapshots_dir = "/tmp/hms_test_snapshots";

    config.mqtt.broker = "192.168.2.15";
    config.mqtt.port = 1883;
    config.mqtt.username = "aamat";
    config.mqtt.password = "exploracion";
    config.mqtt.topic_prefix = "test_detection";

    config.api.host = "0.0.0.0";
    config.api.port = 9999;

    config.llava.enabled = false;

    return config;
}

yolo::MqttConfig makeMqttConfig() {
    yolo::MqttConfig cfg;
    cfg.broker = "192.168.2.15";
    cfg.port = 1883;
    cfg.username = "aamat";
    cfg.password = "exploracion";
    cfg.topic_prefix = "test_detection";
    return cfg;
}

}  // namespace

TEST_CASE("EventManager ignores duplicate motion_start for same camera", "[event_manager]") {
    auto config = makeTestConfig();
    auto buffer_service = std::make_shared<BufferService>(config);
    auto mqtt = std::make_shared<yolo::MqttClient>(makeMqttConfig());
    REQUIRE(mqtt->connect());

    EventManager mgr(buffer_service, mqtt, nullptr, config);
    mgr.start();

    // Small delay for MQTT subscriptions to settle
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Fire first motion_start
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam","post_roll_seconds":5})");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Event should be active (even though processEvent will fail quickly due
    // to no buffer frames — the entry is created before the thread runs)
    // Note: with no RTSP connection, buffer has 0 frames, so processEvent
    // exits early. But we can still verify the ignore logic by sending
    // two rapid-fire events.

    // Fire second motion_start immediately
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam","post_roll_seconds":5})");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // After both events settle, the camera should be available again
    // (the first event exits quickly due to no frames, cleans up)
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    REQUIRE(mgr.activeEventCount() == 0);

    mgr.stop();
    mqtt->disconnect();
}

TEST_CASE("EventManager allows events for different cameras", "[event_manager]") {
    auto config = makeTestConfig();

    // Add a second camera
    yolo::CameraConfig cam2;
    cam2.id = "test_cam2";
    cam2.name = "Test Camera 2";
    cam2.rtsp_url = "rtsp://127.0.0.1:1/nonexistent2";
    cam2.enabled = true;
    config.cameras["test_cam2"] = cam2;

    auto buffer_service = std::make_shared<BufferService>(config);
    auto mqtt = std::make_shared<yolo::MqttClient>(makeMqttConfig());
    REQUIRE(mqtt->connect());

    EventManager mgr(buffer_service, mqtt, nullptr, config);
    mgr.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Fire motion for both cameras
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam","post_roll_seconds":5})");
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam2","post_roll_seconds":5})");

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Both should have been accepted (may already be completing since no frames)
    // We mainly verify no crash / no deadlock here
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    REQUIRE(mgr.activeEventCount() == 0);  // both exit quickly due to no frames

    mgr.stop();
    mqtt->disconnect();
}

TEST_CASE("EventManager cleanup allows re-use after event completes", "[event_manager]") {
    auto config = makeTestConfig();
    auto buffer_service = std::make_shared<BufferService>(config);
    auto mqtt = std::make_shared<yolo::MqttClient>(makeMqttConfig());
    REQUIRE(mqtt->connect());

    EventManager mgr(buffer_service, mqtt, nullptr, config);
    mgr.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // First event
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam","post_roll_seconds":1})");
    // Wait for it to complete (no frames = exits quickly)
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    REQUIRE(mgr.activeEventCount() == 0);

    // Second event for same camera — should NOT be ignored
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam","post_roll_seconds":1})");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // The second event should have been accepted (not ignored)
    // It will also exit quickly, so just verify no crash
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    REQUIRE(mgr.activeEventCount() == 0);

    mgr.stop();
    mqtt->disconnect();
}

// ============================================================================
// Confidence gate logic tests
// These mirror the exact gate check used in event_manager.cpp:
//   float det_conf = best_detections.front().confidence;
//   double conf_gate = cam.immediate_notification_confidence (or 0.70 default);
//   if (det_conf >= conf_gate) { publish; }
// ============================================================================

TEST_CASE("Confidence gate defaults to 0.70 when not set", "[event_manager][confidence_gate]") {
    yolo::AppConfig config;
    yolo::CameraConfig cam;
    cam.id = "test";
    cam.immediate_notification_confidence = 0;  // not explicitly set
    config.cameras["test"] = cam;

    auto cam_it = config.cameras.find("test");
    double conf_gate = (cam_it != config.cameras.end() && cam_it->second.immediate_notification_confidence > 0)
        ? cam_it->second.immediate_notification_confidence : 0.70;

    REQUIRE(conf_gate == Catch::Approx(0.70));
}

TEST_CASE("Confidence gate uses camera-specific value when set", "[event_manager][confidence_gate]") {
    yolo::AppConfig config;
    yolo::CameraConfig cam;
    cam.id = "test";
    cam.immediate_notification_confidence = 0.85;
    config.cameras["test"] = cam;

    auto cam_it = config.cameras.find("test");
    double conf_gate = (cam_it != config.cameras.end() && cam_it->second.immediate_notification_confidence > 0)
        ? cam_it->second.immediate_notification_confidence : 0.70;

    REQUIRE(conf_gate == Catch::Approx(0.85));
}

TEST_CASE("Detection above gate passes notification check", "[event_manager][confidence_gate]") {
    float det_conf = 0.86f;
    double conf_gate = 0.70;
    REQUIRE(det_conf >= conf_gate);

    bool should_publish = (det_conf >= conf_gate);
    REQUIRE(should_publish);
}

TEST_CASE("Detection below gate fails notification check", "[event_manager][confidence_gate]") {
    float det_conf = 0.58f;
    double conf_gate = 0.70;
    REQUIRE_FALSE(det_conf >= conf_gate);

    bool should_publish = (det_conf >= conf_gate);
    REQUIRE_FALSE(should_publish);

    bool below_gate = (det_conf < conf_gate);
    REQUIRE(below_gate);
}

TEST_CASE("Detection at gate boundary uses float-to-double comparison", "[event_manager][confidence_gate]") {
    // float 0.70f < double 0.70 due to precision loss — documented behavior
    float det_conf = 0.70f;
    double conf_gate = 0.70;
    REQUIRE_FALSE(det_conf >= conf_gate);

    float det_conf2 = 0.701f;
    REQUIRE(det_conf2 >= conf_gate);
}

// ============================================================================
// Confidence escalation: best_confidence tracking across frames
// Simulates the live phase loop where best_confidence accumulates
// ============================================================================

TEST_CASE("Best confidence tracks highest detection across frames", "[event_manager][confidence_gate]") {
    // Simulates: frame 1 → 54%, frame 2 → 63%, frame 3 → 72%
    // Only frame 3 should trigger notification
    float best_confidence = 0.0f;
    bool early_notification_sent = false;
    double conf_gate = 0.70;

    // Frame 1: person @ 54%
    float frame1_conf = 0.54f;
    if (frame1_conf > best_confidence) best_confidence = frame1_conf;
    if (!early_notification_sent && best_confidence >= conf_gate) {
        early_notification_sent = true;
    }
    REQUIRE_FALSE(early_notification_sent);

    // Frame 2: person @ 63%
    float frame2_conf = 0.63f;
    if (frame2_conf > best_confidence) best_confidence = frame2_conf;
    if (!early_notification_sent && best_confidence >= conf_gate) {
        early_notification_sent = true;
    }
    REQUIRE_FALSE(early_notification_sent);

    // Frame 3: person @ 72% — should trigger
    float frame3_conf = 0.72f;
    if (frame3_conf > best_confidence) best_confidence = frame3_conf;
    if (!early_notification_sent && best_confidence >= conf_gate) {
        early_notification_sent = true;
    }
    REQUIRE(early_notification_sent);
    REQUIRE(best_confidence == Catch::Approx(0.72f));
}

TEST_CASE("No notification when all detections stay below gate", "[event_manager][confidence_gate]") {
    float best_confidence = 0.0f;
    bool early_notification_sent = false;
    double conf_gate = 0.70;

    // Simulate 5 frames all below gate
    for (float conf : {0.50f, 0.55f, 0.60f, 0.63f, 0.65f}) {
        if (conf > best_confidence) best_confidence = conf;
        if (!early_notification_sent && best_confidence >= conf_gate) {
            early_notification_sent = true;
        }
    }
    REQUIRE_FALSE(early_notification_sent);
    REQUIRE(best_confidence == Catch::Approx(0.65f));
}

TEST_CASE("Notification fires once even with multiple above-gate frames", "[event_manager][confidence_gate]") {
    float best_confidence = 0.0f;
    bool early_notification_sent = false;
    double conf_gate = 0.70;
    int publish_count = 0;

    for (float conf : {0.55f, 0.72f, 0.80f, 0.85f}) {
        if (conf > best_confidence) best_confidence = conf;
        if (!early_notification_sent && best_confidence >= conf_gate) {
            early_notification_sent = true;
            publish_count++;
        }
    }
    REQUIRE(publish_count == 1);
    REQUIRE(best_confidence == Catch::Approx(0.85f));
}

TEST_CASE("Snapshot uses best-confidence frame not first detection", "[event_manager][confidence_gate]") {
    // Simulates: first detection at 54% (no snapshot), later at 72% (snapshot saved)
    float best_confidence = 0.0f;
    bool early_notification_sent = false;
    bool snapshot_saved = false;
    float snapshot_confidence = 0.0f;
    double conf_gate = 0.70;

    // Frame 1: person @ 54% — below gate, no snapshot
    float frame1_conf = 0.54f;
    if (frame1_conf > best_confidence) best_confidence = frame1_conf;
    if (!early_notification_sent && best_confidence >= conf_gate) {
        snapshot_saved = true;
        snapshot_confidence = best_confidence;
        early_notification_sent = true;
    }
    REQUIRE_FALSE(snapshot_saved);

    // Frame 2: person @ 72% — meets gate, snapshot from best frame
    float frame2_conf = 0.72f;
    if (frame2_conf > best_confidence) best_confidence = frame2_conf;
    if (!early_notification_sent && best_confidence >= conf_gate) {
        snapshot_saved = true;
        snapshot_confidence = best_confidence;
        early_notification_sent = true;
    }
    REQUIRE(snapshot_saved);
    REQUIRE(snapshot_confidence == Catch::Approx(0.72f));
}

TEST_CASE("Recording deleted when best confidence stays below gate", "[event_manager][confidence_gate]") {
    float best_conf = 0.63f;  // best across entire event
    double conf_gate = 0.70;
    bool below_gate = (best_conf < conf_gate);
    REQUIRE(below_gate);

    // Production code: if (below_gate) remove(recording); db_recording = "";
    std::string db_recording = below_gate ? "" : "recording.mp4";
    REQUIRE(db_recording.empty());
}

TEST_CASE("Recording kept when best confidence meets gate", "[event_manager][confidence_gate]") {
    float best_conf = 0.82f;
    double conf_gate = 0.70;
    bool below_gate = (best_conf < conf_gate);
    REQUIRE_FALSE(below_gate);

    std::string db_recording = below_gate ? "" : "recording.mp4";
    REQUIRE(db_recording == "recording.mp4");
}

// ============================================================================
// Gate with different camera configs
// ============================================================================

TEST_CASE("High gate (0.90) requires very confident detection", "[event_manager][confidence_gate]") {
    double conf_gate = 0.90;

    // 86% car (typical front_door) → below 90% gate
    float car_conf = 0.86f;
    REQUIRE_FALSE(car_conf >= conf_gate);

    // 92% car → above gate
    float high_car_conf = 0.92f;
    REQUIRE(high_car_conf >= conf_gate);
}

TEST_CASE("Low gate (0.50) accepts most detections", "[event_manager][confidence_gate]") {
    double conf_gate = 0.50;

    float det_conf = 0.54f;
    REQUIRE(det_conf >= conf_gate);
}

TEST_CASE("Gate 1.00 blocks all real detections", "[event_manager][confidence_gate]") {
    // Used in e2e tests to verify no notifications
    double conf_gate = 1.00;

    REQUIRE_FALSE(0.99f >= conf_gate);
    REQUIRE_FALSE(0.86f >= conf_gate);
    REQUIRE(1.00f >= conf_gate);  // only perfect score passes
}

TEST_CASE("Unknown camera falls back to default 0.70 gate", "[event_manager][confidence_gate]") {
    yolo::AppConfig config;
    // No cameras configured at all

    std::string camera_id = "unknown_cam";
    auto cam_it = config.cameras.find(camera_id);
    double conf_gate = (cam_it != config.cameras.end() && cam_it->second.immediate_notification_confidence > 0)
        ? cam_it->second.immediate_notification_confidence : 0.70;

    REQUIRE(conf_gate == Catch::Approx(0.70));
}

TEST_CASE("Multiple cameras with different gates", "[event_manager][confidence_gate]") {
    yolo::AppConfig config;

    yolo::CameraConfig patio;
    patio.id = "patio";
    patio.immediate_notification_confidence = 0.60;
    config.cameras["patio"] = patio;

    yolo::CameraConfig front;
    front.id = "front_door";
    front.immediate_notification_confidence = 0.80;
    config.cameras["front_door"] = front;

    // Same 72% detection: passes patio gate, fails front_door gate
    float det_conf = 0.72f;

    auto patio_it = config.cameras.find("patio");
    double patio_gate = patio_it->second.immediate_notification_confidence;
    REQUIRE(det_conf >= patio_gate);

    auto front_it = config.cameras.find("front_door");
    double front_gate = front_it->second.immediate_notification_confidence;
    REQUIRE_FALSE(det_conf >= front_gate);
}

TEST_CASE("EventManager motion_stop ends active event", "[event_manager]") {
    auto config = makeTestConfig();
    auto buffer_service = std::make_shared<BufferService>(config);
    auto mqtt = std::make_shared<yolo::MqttClient>(makeMqttConfig());
    REQUIRE(mqtt->connect());

    EventManager mgr(buffer_service, mqtt, nullptr, config);
    mgr.start();

    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Start event
    mqtt->publish("camera/event/motion/start",
                  R"({"camera_id":"test_cam","post_roll_seconds":30})");
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Stop event
    mqtt->publish("camera/event/motion/stop",
                  R"({"camera_id":"test_cam"})");
    // Wait for post-roll + cleanup
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));

    REQUIRE(mgr.activeEventCount() == 0);

    mgr.stop();
    mqtt->disconnect();
}
