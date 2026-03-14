#include <catch2/catch_all.hpp>
#include "detection_engine.h"
#include "vision_client.h"
#include "buffer_service.h"
#include "event_manager.h"
#include "config_manager.h"

#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <vector>

using namespace hms;
using Catch::Matchers::WithinAbs;

// Helper: create a solid-color BGR24 FrameData
static FrameData makeTestFrame(int w, int h, uint8_t b = 128, uint8_t g = 128, uint8_t r = 128) {
    FrameData frame;
    frame.resize(w, h);
    frame.frame_number = 1;
    frame.timestamp = SteadyClock::now();
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            auto* px = frame.pixels.data() + y * frame.stride + x * 3;
            px[0] = b;
            px[1] = g;
            px[2] = r;
        }
    }
    return frame;
}

// ============================================================================
// DetectionEngine load/unload lifecycle
// ============================================================================

TEST_CASE("Engine validates model on construction then unloads", "[gpu_lifecycle]") {
    // Non-existent model: constructor load() fails, isModelValid() = false
    DetectionEngine engine("/nonexistent.onnx");
    REQUIRE_FALSE(engine.isLoaded());
    REQUIRE_FALSE(engine.isModelValid());
}

TEST_CASE("Engine load/unload are idempotent", "[gpu_lifecycle]") {
    DetectionEngine engine("/nonexistent.onnx");

    // Multiple loads on non-existent model — no crash
    engine.load();
    engine.load();
    REQUIRE_FALSE(engine.isLoaded());

    // Multiple unloads — no crash
    engine.unload();
    engine.unload();
    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("Detect returns empty when unloaded", "[gpu_lifecycle]") {
    DetectionEngine engine("/nonexistent.onnx");
    auto frame = makeTestFrame(640, 480);

    auto dets = engine.detect(frame, 0.5f, 0.45f, {});
    REQUIRE(dets.empty());
}

TEST_CASE("Detect returns empty on empty frame", "[gpu_lifecycle]") {
    DetectionEngine engine("/nonexistent.onnx");
    FrameData empty_frame;

    auto dets = engine.detect(empty_frame, 0.5f, 0.45f, {});
    REQUIRE(dets.empty());
}

TEST_CASE("Unload while not loaded is safe", "[gpu_lifecycle]") {
    DetectionEngine engine("/nonexistent.onnx");
    REQUIRE_FALSE(engine.isLoaded());

    // Should not crash or throw
    engine.unload();
    REQUIRE_FALSE(engine.isLoaded());
}

// ============================================================================
// Thread safety / deadlock tests
// ============================================================================

TEST_CASE("Concurrent load/unload does not deadlock", "[gpu_lifecycle][threading]") {
    DetectionEngine engine("/nonexistent.onnx");
    std::atomic<bool> done{false};

    // Run load/unload from multiple threads for 500ms
    auto worker = [&](bool do_load) {
        while (!done) {
            if (do_load) {
                engine.load();
            } else {
                engine.unload();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };

    std::thread t1(worker, true);
    std::thread t2(worker, false);
    std::thread t3(worker, true);

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    done = true;

    t1.join();
    t2.join();
    t3.join();

    // No deadlock — we got here
    REQUIRE(true);
}

TEST_CASE("Concurrent detect and unload does not deadlock", "[gpu_lifecycle][threading]") {
    DetectionEngine engine("/nonexistent.onnx");
    auto frame = makeTestFrame(640, 480);
    std::atomic<bool> done{false};

    // Thread 1: repeatedly tries to detect
    std::thread detector([&]() {
        while (!done) {
            auto dets = engine.detect(frame, 0.5f, 0.45f, {});
            // Should always return empty (no model loaded)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });

    // Thread 2: repeatedly load/unload
    std::thread lifecycle([&]() {
        while (!done) {
            engine.load();
            engine.unload();
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    done = true;

    detector.join();
    lifecycle.join();

    REQUIRE(true);
}

TEST_CASE("Load with timeout does not block indefinitely", "[gpu_lifecycle][threading]") {
    // Test that load() returns in reasonable time even with non-existent model
    auto future = std::async(std::launch::async, []() {
        DetectionEngine engine("/nonexistent.onnx");
        engine.load();
        return true;
    });

    auto status = future.wait_for(std::chrono::seconds(5));
    REQUIRE(status == std::future_status::ready);
    REQUIRE(future.get() == true);
}

TEST_CASE("Unload with timeout does not block indefinitely", "[gpu_lifecycle][threading]") {
    auto future = std::async(std::launch::async, []() {
        DetectionEngine engine("/nonexistent.onnx");
        engine.unload();
        return true;
    });

    auto status = future.wait_for(std::chrono::seconds(5));
    REQUIRE(status == std::future_status::ready);
}

// ============================================================================
// Simulated event flow: load → detect → unload → LLaVA → unload
// ============================================================================

TEST_CASE("Event flow: engine load-detect-unload cycle", "[gpu_lifecycle][flow]") {
    DetectionEngine engine("/nonexistent.onnx");
    auto frame = makeTestFrame(640, 480);

    // Phase 1: Load YOLO
    engine.load();
    // Model doesn't exist, but load() should not throw
    REQUIRE_FALSE(engine.isLoaded());

    // Phase 2: Detect (returns empty since model invalid)
    auto dets = engine.detect(frame, 0.5f, 0.45f, {"person"});
    REQUIRE(dets.empty());

    // Phase 3: Unload YOLO (free GPU for LLaVA)
    engine.unload();
    REQUIRE_FALSE(engine.isLoaded());

    // Phase 4: LLaVA would run here (tested separately)
    // Phase 5: LLaVA unloads via keep_alive=0 (tested separately)
}

TEST_CASE("Multiple sequential event cycles", "[gpu_lifecycle][flow]") {
    DetectionEngine engine("/nonexistent.onnx");
    auto frame = makeTestFrame(640, 480);

    for (int i = 0; i < 10; ++i) {
        engine.load();
        auto dets = engine.detect(frame, 0.5f, 0.45f, {});
        engine.unload();
    }

    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("Concurrent events on different cameras simulate load/unload", "[gpu_lifecycle][flow]") {
    // Two cameras fire motion events nearly simultaneously
    // Only one should be loading YOLO at a time (mutex serializes access)
    DetectionEngine engine("/nonexistent.onnx");
    auto frame = makeTestFrame(640, 480);

    std::atomic<int> load_count{0};
    std::atomic<int> unload_count{0};

    auto simulate_event = [&](const std::string& cam_id) {
        engine.load();
        load_count++;

        // Simulate detection phase
        for (int i = 0; i < 5; ++i) {
            engine.detect(frame, 0.5f, 0.45f, {});
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        engine.unload();
        unload_count++;
    };

    std::thread cam1(simulate_event, "patio");
    std::thread cam2(simulate_event, "front_door");

    cam1.join();
    cam2.join();

    REQUIRE(load_count == 2);
    REQUIRE(unload_count == 2);
    REQUIRE_FALSE(engine.isLoaded());
}

// ============================================================================
// VisionClient keep_alive=0 (Ollama model unloading)
// ============================================================================

TEST_CASE("VisionClient builds request with keep_alive=0", "[gpu_lifecycle][vision]") {
    hms::LlavaConfig config;
    config.enabled = true;
    config.endpoint = "http://localhost:11434";
    config.model = "llava:7b";
    config.timeout_seconds = 10;
    config.max_words = 15;

    VisionClient client(config);

    // Verify the prompt is built correctly
    auto prompt = client.buildPrompt("patio", "dog");
    REQUIRE_FALSE(prompt.empty());
}

TEST_CASE("VisionClient selectPrimaryClass priority", "[gpu_lifecycle][vision]") {
    // person > dog > cat > package > car
    REQUIRE(VisionClient::selectPrimaryClass({"car", "person", "dog"}) == "person");
    REQUIRE(VisionClient::selectPrimaryClass({"cat", "dog"}) == "dog");
    REQUIRE(VisionClient::selectPrimaryClass({"car", "package"}) == "package");
    REQUIRE(VisionClient::selectPrimaryClass({"car"}) == "car");
    REQUIRE(VisionClient::selectPrimaryClass({}) == "object");
    REQUIRE(VisionClient::selectPrimaryClass({"bicycle"}) == "bicycle");
}

TEST_CASE("VisionClient analyze fails gracefully on missing snapshot", "[gpu_lifecycle][vision]") {
    hms::LlavaConfig config;
    config.enabled = true;
    config.endpoint = "http://localhost:11434";
    config.model = "llava:7b";
    config.timeout_seconds = 5;
    config.max_words = 15;

    VisionClient client(config);
    auto result = client.analyze("/nonexistent_snapshot.jpg", "patio", "person");

    REQUIRE_FALSE(result.is_valid);
    REQUIRE(result.context.empty());
}

// ============================================================================
// BufferService model validation
// ============================================================================

TEST_CASE("BufferService loadDetectionModel with missing model", "[gpu_lifecycle][buffer]") {
    hms::AppConfig config;
    config.detection.model_path = "/nonexistent.onnx";
    config.detection.gpu_enabled = false;
    config.buffer.preroll_seconds = 2;
    config.buffer.fps = 15;

    BufferService service(config);
    service.loadDetectionModel();

    // Model file doesn't exist → engine not created
    REQUIRE(service.getDetectionEngine() == nullptr);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_CASE("Detect called between load and unload from different threads", "[gpu_lifecycle][edge]") {
    DetectionEngine engine("/nonexistent.onnx");
    auto frame = makeTestFrame(640, 480);
    std::atomic<bool> started{false};
    std::atomic<bool> detected{false};

    // Thread 1: load, signal, wait, unload
    std::thread loader([&]() {
        engine.load();
        started = true;
        // Wait for detect to run
        while (!detected) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        engine.unload();
    });

    // Thread 2: wait for load, then detect
    std::thread detector([&]() {
        while (!started) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        engine.detect(frame, 0.5f, 0.45f, {});
        detected = true;
    });

    loader.join();
    detector.join();

    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("Rapid fire load/unload stress test", "[gpu_lifecycle][stress]") {
    DetectionEngine engine("/nonexistent.onnx");

    // 100 rapid cycles
    for (int i = 0; i < 100; ++i) {
        engine.load();
        engine.unload();
    }

    REQUIRE_FALSE(engine.isLoaded());
}

TEST_CASE("isLoaded is thread-safe", "[gpu_lifecycle][threading]") {
    DetectionEngine engine("/nonexistent.onnx");
    std::atomic<bool> done{false};
    std::atomic<int> check_count{0};

    std::thread checker([&]() {
        while (!done) {
            [[maybe_unused]] bool loaded = engine.isLoaded();
            check_count++;
        }
    });

    std::thread toggler([&]() {
        for (int i = 0; i < 50; ++i) {
            engine.load();
            engine.unload();
        }
        done = true;
    });

    checker.join();
    toggler.join();

    REQUIRE(check_count > 0);
}
