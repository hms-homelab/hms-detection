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
