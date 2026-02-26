#include <catch2/catch_all.hpp>
#include "buffer_service.h"

using namespace hms;

namespace {
yolo::AppConfig makeTestConfig(bool enable_all = true) {
    yolo::AppConfig config;
    config.buffer.preroll_seconds = 5;
    config.buffer.fps = 15;

    yolo::CameraConfig cam1;
    cam1.id = "test_cam1";
    cam1.name = "Test Camera 1";
    cam1.rtsp_url = "rtsp://localhost:8554/nonexistent1";
    cam1.enabled = enable_all;
    config.cameras["test_cam1"] = cam1;

    yolo::CameraConfig cam2;
    cam2.id = "test_cam2";
    cam2.name = "Test Camera 2";
    cam2.rtsp_url = "rtsp://localhost:8554/nonexistent2";
    cam2.enabled = true;
    config.cameras["test_cam2"] = cam2;

    yolo::CameraConfig cam3;
    cam3.id = "test_cam3";
    cam3.name = "Test Camera 3 (disabled)";
    cam3.rtsp_url = "rtsp://localhost:8554/nonexistent3";
    cam3.enabled = false;
    config.cameras["test_cam3"] = cam3;

    return config;
}
}  // namespace

TEST_CASE("BufferService constructs from config", "[buffer_service]") {
    auto config = makeTestConfig();
    BufferService svc(config);

    // cam3 is disabled, so only 2 cameras
    auto ids = svc.cameraIds();
    REQUIRE(ids.size() == 2);
}

TEST_CASE("BufferService disabled cameras excluded", "[buffer_service]") {
    auto config = makeTestConfig();
    BufferService svc(config);

    auto ids = svc.cameraIds();
    // cam3 should not be present
    bool found_cam3 = false;
    for (const auto& id : ids) {
        if (id == "test_cam3") found_cam3 = true;
    }
    REQUIRE_FALSE(found_cam3);
}

TEST_CASE("BufferService getLatestFrame returns nullptr for unknown camera", "[buffer_service]") {
    auto config = makeTestConfig();
    BufferService svc(config);

    REQUIRE(svc.getLatestFrame("nonexistent") == nullptr);
}

TEST_CASE("BufferService getCameraBuffer returns nullptr for unknown camera", "[buffer_service]") {
    auto config = makeTestConfig();
    BufferService svc(config);

    REQUIRE(svc.getCameraBuffer("nonexistent") == nullptr);
}

TEST_CASE("BufferService getAllStats returns correct shape", "[buffer_service]") {
    auto config = makeTestConfig();
    BufferService svc(config);

    auto stats = svc.getAllStats();
    REQUIRE(stats.size() == 2);  // 2 enabled cameras

    for (const auto& s : stats) {
        REQUIRE_FALSE(s.camera_id.empty());
        REQUIRE_FALSE(s.camera_name.empty());
        REQUIRE(s.max_frames == 75);  // 5 * 15
        REQUIRE(s.buffer_size == 0);  // no frames yet
        REQUIRE(s.frames_captured == 0);
        REQUIRE(s.is_connected == false);
        REQUIRE(s.is_healthy == false);
    }
}

TEST_CASE("BufferService isHealthy false when no frames", "[buffer_service]") {
    auto config = makeTestConfig();
    BufferService svc(config);

    REQUIRE_FALSE(svc.isHealthy());
}

TEST_CASE("BufferService all cameras disabled yields empty", "[buffer_service]") {
    auto config = makeTestConfig(false);  // cam1 disabled
    // cam2 still enabled, cam3 always disabled
    BufferService svc(config);

    auto ids = svc.cameraIds();
    // cam1 disabled + cam3 disabled = only cam2
    REQUIRE(ids.size() == 1);
}
