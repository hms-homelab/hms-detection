#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "detection_engine.h"

using namespace hms;
using Catch::Matchers::WithinAbs;

// Helper: create a solid-color BGR24 FrameData
static FrameData makeFrame(int w, int h, uint8_t b = 0, uint8_t g = 0, uint8_t r = 0) {
    FrameData frame;
    frame.resize(w, h);
    frame.frame_number = 1;
    frame.timestamp = Clock::now();
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

// ============================================================
// Preprocessing tests
// ============================================================

TEST_CASE("Preprocess letterbox scaling - 640x480 input", "[detection][preprocess]") {
    // Create engine without a model (won't load, but preprocess is still testable)
    DetectionEngine engine("/nonexistent.onnx");
    REQUIRE_FALSE(engine.isLoaded());

    auto frame = makeFrame(640, 480);
    float scale, pad_x, pad_y;
    auto tensor = engine.preprocess(frame, scale, pad_x, pad_y);

    // 640x480 → fit into 640x640
    // scale = min(640/640, 640/480) = min(1.0, 1.333) = 1.0
    // new_w = 640, new_h = 480
    // pad_x = 0, pad_y = (640-480)/2 = 80
    REQUIRE_THAT(scale, WithinAbs(1.0f, 0.01f));
    REQUIRE_THAT(pad_x, WithinAbs(0.0f, 1.0f));
    REQUIRE_THAT(pad_y, WithinAbs(80.0f, 1.0f));

    // Tensor should be [3, 640, 640] = 1,228,800 floats
    REQUIRE(tensor.size() == 3 * 640 * 640);
}

TEST_CASE("Preprocess letterbox scaling - 1920x1080 input", "[detection][preprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    auto frame = makeFrame(1920, 1080);
    float scale, pad_x, pad_y;
    auto tensor = engine.preprocess(frame, scale, pad_x, pad_y);

    // 1920x1080 → fit into 640x640
    // scale = min(640/1920, 640/1080) = min(0.333, 0.593) = 0.333
    // new_w = round(1920*0.333) = 640, new_h = round(1080*0.333) = 360
    // pad_x = 0, pad_y = (640-360)/2 = 140
    REQUIRE_THAT(scale, WithinAbs(0.333f, 0.01f));
    REQUIRE_THAT(pad_x, WithinAbs(0.0f, 1.0f));
    REQUIRE_THAT(pad_y, WithinAbs(140.0f, 1.0f));

    REQUIRE(tensor.size() == 3 * 640 * 640);
}

TEST_CASE("Preprocess letterbox scaling - square 640x640 input", "[detection][preprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    auto frame = makeFrame(640, 640);
    float scale, pad_x, pad_y;
    auto tensor = engine.preprocess(frame, scale, pad_x, pad_y);

    // Perfect fit, no padding
    REQUIRE_THAT(scale, WithinAbs(1.0f, 0.01f));
    REQUIRE_THAT(pad_x, WithinAbs(0.0f, 1.0f));
    REQUIRE_THAT(pad_y, WithinAbs(0.0f, 1.0f));
}

TEST_CASE("Preprocess pixel values are normalized to [0,1]", "[detection][preprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    // White frame (B=255, G=255, R=255)
    auto frame = makeFrame(640, 640, 255, 255, 255);
    float scale, pad_x, pad_y;
    auto tensor = engine.preprocess(frame, scale, pad_x, pad_y);

    // Check a pixel in the R channel — should be 1.0 (255/255)
    // NCHW layout: R channel = tensor[0 * 640*640 + 0] for (0,0)
    REQUIRE_THAT(tensor[0], WithinAbs(1.0f, 0.001f));
    // G channel
    REQUIRE_THAT(tensor[640 * 640], WithinAbs(1.0f, 0.001f));
    // B channel
    REQUIRE_THAT(tensor[2 * 640 * 640], WithinAbs(1.0f, 0.001f));
}

TEST_CASE("Preprocess gray padding value", "[detection][preprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    // 640x480 → 80px padding on top and bottom
    auto frame = makeFrame(640, 480, 0, 0, 0);  // black frame
    float scale, pad_x, pad_y;
    auto tensor = engine.preprocess(frame, scale, pad_x, pad_y);

    // Top-left pixel (0,0) should be in padding zone → gray = 114/255 ≈ 0.447
    float expected_gray = 114.0f / 255.0f;
    REQUIRE_THAT(tensor[0], WithinAbs(expected_gray, 0.01f));
}

// ============================================================
// IoU tests
// ============================================================

TEST_CASE("IoU - identical boxes", "[detection][iou]") {
    Detection a{.x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100};
    Detection b{.x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100};
    REQUIRE_THAT(DetectionEngine::iou(a, b), WithinAbs(1.0f, 0.001f));
}

TEST_CASE("IoU - disjoint boxes", "[detection][iou]") {
    Detection a{.x1 = 0, .y1 = 0, .x2 = 50, .y2 = 50};
    Detection b{.x1 = 100, .y1 = 100, .x2 = 200, .y2 = 200};
    REQUIRE_THAT(DetectionEngine::iou(a, b), WithinAbs(0.0f, 0.001f));
}

TEST_CASE("IoU - partial overlap", "[detection][iou]") {
    // a = [0,0,100,100] area=10000
    // b = [50,0,150,100] area=10000
    // intersection = [50,0,100,100] area=5000
    // union = 10000 + 10000 - 5000 = 15000
    // iou = 5000/15000 = 0.333
    Detection a{.x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100};
    Detection b{.x1 = 50, .y1 = 0, .x2 = 150, .y2 = 100};
    REQUIRE_THAT(DetectionEngine::iou(a, b), WithinAbs(0.333f, 0.01f));
}

TEST_CASE("IoU - contained box", "[detection][iou]") {
    // Small box fully inside large box
    // a = [0,0,100,100] area=10000
    // b = [25,25,75,75] area=2500
    // intersection = 2500
    // union = 10000 + 2500 - 2500 = 10000
    // iou = 2500/10000 = 0.25
    Detection a{.x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100};
    Detection b{.x1 = 25, .y1 = 25, .x2 = 75, .y2 = 75};
    REQUIRE_THAT(DetectionEngine::iou(a, b), WithinAbs(0.25f, 0.01f));
}

// ============================================================
// NMS tests
// ============================================================

TEST_CASE("NMS - empty input", "[detection][nms]") {
    auto keep = DetectionEngine::nms({}, 0.45f);
    REQUIRE(keep.empty());
}

TEST_CASE("NMS - single detection kept", "[detection][nms]") {
    std::vector<Detection> dets = {
        {.class_id = 0, .confidence = 0.9f, .x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100},
    };
    auto keep = DetectionEngine::nms(dets, 0.45f);
    REQUIRE(keep.size() == 1);
    REQUIRE(keep[0] == 0);
}

TEST_CASE("NMS - overlapping same class suppresses lower confidence", "[detection][nms]") {
    std::vector<Detection> dets = {
        {.class_id = 0, .confidence = 0.9f, .x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100},
        {.class_id = 0, .confidence = 0.7f, .x1 = 10, .y1 = 10, .x2 = 110, .y2 = 110},
    };
    auto keep = DetectionEngine::nms(dets, 0.45f);
    REQUIRE(keep.size() == 1);
    // Should keep the higher confidence one
    REQUIRE(dets[keep[0]].confidence > 0.8f);
}

TEST_CASE("NMS - non-overlapping same class keeps both", "[detection][nms]") {
    std::vector<Detection> dets = {
        {.class_id = 0, .confidence = 0.9f, .x1 = 0, .y1 = 0, .x2 = 50, .y2 = 50},
        {.class_id = 0, .confidence = 0.8f, .x1 = 200, .y1 = 200, .x2 = 300, .y2 = 300},
    };
    auto keep = DetectionEngine::nms(dets, 0.45f);
    REQUIRE(keep.size() == 2);
}

TEST_CASE("NMS - different classes not suppressed", "[detection][nms]") {
    // Same box, different classes → NMS is per-class, both kept
    std::vector<Detection> dets = {
        {.class_id = 0, .confidence = 0.9f, .x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100},
        {.class_id = 1, .confidence = 0.85f, .x1 = 0, .y1 = 0, .x2 = 100, .y2 = 100},
    };
    auto keep = DetectionEngine::nms(dets, 0.45f);
    REQUIRE(keep.size() == 2);
}

// ============================================================
// Postprocess tests (synthetic output tensor)
// ============================================================

TEST_CASE("Postprocess - single high-confidence detection", "[detection][postprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    // Simulate output tensor [1, 84, 2] — 2 candidates, 4 + 80 classes
    const int num_candidates = 2;
    const int num_values = 84;
    std::vector<float> output(num_values * num_candidates, 0.0f);

    // Candidate 0: cx=320, cy=320, w=100, h=200, class 0 (person) = 0.9
    output[0 * num_candidates + 0] = 320.0f;  // cx
    output[1 * num_candidates + 0] = 320.0f;  // cy
    output[2 * num_candidates + 0] = 100.0f;  // w
    output[3 * num_candidates + 0] = 200.0f;  // h
    output[4 * num_candidates + 0] = 0.9f;    // class 0 = person

    // Candidate 1: low confidence (all class scores 0)
    output[0 * num_candidates + 1] = 100.0f;
    output[1 * num_candidates + 1] = 100.0f;
    output[2 * num_candidates + 1] = 50.0f;
    output[3 * num_candidates + 1] = 50.0f;
    // All class scores remain 0 → should be filtered out

    // No padding, scale=1 (640x640 input)
    auto dets = engine.postprocess(output.data(), num_candidates,
                                   0.5f, 0.45f,
                                   1.0f, 0.0f, 0.0f,
                                   640, 640, {});

    REQUIRE(dets.size() == 1);
    REQUIRE(dets[0].class_name == "person");
    REQUIRE(dets[0].class_id == 0);
    REQUIRE_THAT(dets[0].confidence, WithinAbs(0.9f, 0.01f));

    // BBox: cx=320, cy=320, w=100, h=200 → x1=270, y1=220, x2=370, y2=420
    REQUIRE_THAT(dets[0].x1, WithinAbs(270.0f, 1.0f));
    REQUIRE_THAT(dets[0].y1, WithinAbs(220.0f, 1.0f));
    REQUIRE_THAT(dets[0].x2, WithinAbs(370.0f, 1.0f));
    REQUIRE_THAT(dets[0].y2, WithinAbs(420.0f, 1.0f));
}

TEST_CASE("Postprocess - class filter", "[detection][postprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    const int num_candidates = 2;
    const int num_values = 84;
    std::vector<float> output(num_values * num_candidates, 0.0f);

    // Candidate 0: person (class 0) = 0.9
    output[0 * num_candidates + 0] = 320.0f;
    output[1 * num_candidates + 0] = 320.0f;
    output[2 * num_candidates + 0] = 100.0f;
    output[3 * num_candidates + 0] = 200.0f;
    output[4 * num_candidates + 0] = 0.9f;

    // Candidate 1: car (class 2) = 0.85
    output[0 * num_candidates + 1] = 100.0f;
    output[1 * num_candidates + 1] = 100.0f;
    output[2 * num_candidates + 1] = 80.0f;
    output[3 * num_candidates + 1] = 60.0f;
    output[6 * num_candidates + 1] = 0.85f;  // class 2 = car

    // Filter: only "person"
    auto dets = engine.postprocess(output.data(), num_candidates,
                                   0.5f, 0.45f,
                                   1.0f, 0.0f, 0.0f,
                                   640, 640, {"person"});

    REQUIRE(dets.size() == 1);
    REQUIRE(dets[0].class_name == "person");
}

TEST_CASE("Postprocess - confidence threshold filters low scores", "[detection][postprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    const int num_candidates = 1;
    const int num_values = 84;
    std::vector<float> output(num_values * num_candidates, 0.0f);

    // Candidate with confidence 0.3 (below 0.5 threshold)
    output[0] = 320.0f;
    output[1] = 320.0f;
    output[2] = 100.0f;
    output[3] = 200.0f;
    output[4] = 0.3f;  // class 0 = person at 0.3

    auto dets = engine.postprocess(output.data(), num_candidates,
                                   0.5f, 0.45f,
                                   1.0f, 0.0f, 0.0f,
                                   640, 640, {});

    REQUIRE(dets.empty());
}

TEST_CASE("Postprocess - reverse letterbox transforms coordinates", "[detection][postprocess]") {
    DetectionEngine engine("/nonexistent.onnx");

    const int num_candidates = 1;
    const int num_values = 84;
    std::vector<float> output(num_values * num_candidates, 0.0f);

    // Detection at center of 640x640 letterboxed image
    output[0] = 320.0f;  // cx
    output[1] = 320.0f;  // cy
    output[2] = 100.0f;  // w
    output[3] = 100.0f;  // h
    output[4] = 0.9f;

    // Simulating 1920x1080 original: scale=0.333, pad_x=0, pad_y=140
    float scale = 640.0f / 1920.0f;  // 0.3333
    float pad_x = 0.0f;
    float pad_y = (640.0f - 1080.0f * scale) / 2.0f;  // (640-360)/2 = 140

    auto dets = engine.postprocess(output.data(), num_candidates,
                                   0.5f, 0.45f,
                                   scale, pad_x, pad_y,
                                   1920, 1080, {});

    REQUIRE(dets.size() == 1);
    // cx=320, cy=320 in 640x640 space
    // x1 = (320-50-0)/0.333 = 270/0.333 = 810
    // y1 = (320-50-140)/0.333 = 130/0.333 = 390
    // x2 = (320+50-0)/0.333 = 370/0.333 = 1110
    // y2 = (320+50-140)/0.333 = 230/0.333 = 690
    REQUIRE_THAT(dets[0].x1, WithinAbs(810.0f, 5.0f));
    REQUIRE_THAT(dets[0].y1, WithinAbs(390.0f, 5.0f));
    REQUIRE_THAT(dets[0].x2, WithinAbs(1110.0f, 5.0f));
    REQUIRE_THAT(dets[0].y2, WithinAbs(690.0f, 5.0f));
}

// ============================================================
// Class names
// ============================================================

TEST_CASE("Class names - 80 COCO classes loaded", "[detection][classes]") {
    DetectionEngine engine("/nonexistent.onnx");
    auto& names = engine.classNames();
    REQUIRE(names.size() == 80);
    REQUIRE(names[0] == "person");
    REQUIRE(names[1] == "bicycle");
    REQUIRE(names[2] == "car");
    REQUIRE(names[79] == "toothbrush");
}
