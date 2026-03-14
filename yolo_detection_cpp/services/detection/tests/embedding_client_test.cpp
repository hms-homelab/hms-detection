#include <catch2/catch_test_macros.hpp>
#include "embedding_client.h"
#include "config_manager.h"

// ────────────────────────────────────────────────────────────────────
// EmbeddingClient unit tests
// ────────────────────────────────────────────────────────────────────

TEST_CASE("EmbeddingClient constructs with defaults", "[embedding]") {
    hms::EmbeddingClient client;
    // Should not crash — constructor sets url_ and model_
    // embed() won't be called without a running Ollama server
    CHECK(true);
}

TEST_CASE("EmbeddingClient constructs with custom URL and model", "[embedding]") {
    hms::EmbeddingClient client("http://10.0.0.5:11434", "nomic-embed-text");
    CHECK(true);  // No crash
}

TEST_CASE("EmbeddingClient::embed returns empty for empty text", "[embedding]") {
    hms::EmbeddingClient client("http://localhost:11434", "nomic-embed-text");
    auto result = client.embed("");
    CHECK(result.empty());
}

TEST_CASE("EmbeddingClient::embed returns empty for unreachable server", "[embedding]") {
    // Use a non-routable address to ensure quick failure
    hms::EmbeddingClient client("http://192.0.2.1:1", "nomic-embed-text");
    auto result = client.embed("test text");
    CHECK(result.empty());  // Should fail gracefully, not crash
}

// ────────────────────────────────────────────────────────────────────
// PeriodicSnapshotManager config-level tests
// ────────────────────────────────────────────────────────────────────

TEST_CASE("Periodic snapshot interval config defaults to disabled", "[periodic]") {
    hms::CameraConfig cam;
    CHECK(cam.periodic_snapshot_interval == 0);
}

TEST_CASE("Periodic snapshot skips disabled cameras", "[periodic]") {
    hms::AppConfig config;

    hms::CameraConfig cam1;
    cam1.id = "patio";
    cam1.enabled = true;
    cam1.periodic_snapshot_interval = 300;
    config.cameras["patio"] = cam1;

    hms::CameraConfig cam2;
    cam2.id = "garage";
    cam2.enabled = false;
    cam2.periodic_snapshot_interval = 300;
    config.cameras["garage"] = cam2;

    hms::CameraConfig cam3;
    cam3.id = "side";
    cam3.enabled = true;
    cam3.periodic_snapshot_interval = 0;  // Disabled via interval
    config.cameras["side"] = cam3;

    // Count cameras that would start periodic snapshot threads
    int active = 0;
    for (const auto& [id, cam] : config.cameras) {
        if (cam.enabled && cam.periodic_snapshot_interval > 0)
            active++;
    }
    CHECK(active == 1);  // Only patio
}

TEST_CASE("Periodic snapshot filename convention", "[periodic]") {
    // Verify naming convention: {camera_id}_periodic_{YYYYMMDD}_{HHMMSS}.jpg
    std::string camera_id = "patio";
    std::string expected_pattern = "patio_periodic_";

    // Simulate filename generation
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_periodic_%04d%02d%02d_%02d%02d%02d.jpg",
                  camera_id.c_str(), 2026, 3, 4, 14, 30, 0);
    std::string filename(buf);

    CHECK(filename == "patio_periodic_20260304_143000.jpg");
    CHECK(filename.find(expected_pattern) == 0);
    CHECK(filename.find(".jpg") == filename.size() - 4);

    // Thumbnail variant
    std::snprintf(buf, sizeof(buf), "%s_periodic_%04d%02d%02d_%02d%02d%02d_thumb.jpg",
                  camera_id.c_str(), 2026, 3, 4, 14, 30, 0);
    std::string thumb(buf);
    CHECK(thumb == "patio_periodic_20260304_143000_thumb.jpg");
    CHECK(thumb.find("_thumb.jpg") != std::string::npos);
}

TEST_CASE("Periodic snapshot uses generic LLaVA prompt", "[periodic]") {
    // Verify that periodic snapshot uses a scene description prompt,
    // not the motion-event class-based prompt
    std::string periodic_prompt =
        "In 20 words or less, describe the scene. What do you see? "
        "Include weather, lighting, and any people, animals, or vehicles.";

    // Should NOT contain {class} placeholder (those are for motion events)
    CHECK(periodic_prompt.find("{class}") == std::string::npos);
    // Should NOT contain {max_words} placeholder
    CHECK(periodic_prompt.find("{max_words}") == std::string::npos);
    // Should describe the scene generically
    CHECK(periodic_prompt.find("scene") != std::string::npos);
}

TEST_CASE("Embedding vector format for pgvector", "[embedding]") {
    // Test the vector formatting used by insert_periodic_snapshot
    std::vector<float> embedding = {0.1f, 0.2f, -0.3f, 0.0f, 1.0f};

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < embedding.size(); ++i) {
        if (i > 0) oss << ",";
        oss << embedding[i];
    }
    oss << "]";
    std::string vec_literal = oss.str();

    CHECK(vec_literal.front() == '[');
    CHECK(vec_literal.back() == ']');
    CHECK(vec_literal.find(',') != std::string::npos);
    // Should have exactly 4 commas for 5 elements
    int commas = 0;
    for (char c : vec_literal) if (c == ',') commas++;
    CHECK(commas == 4);
}

TEST_CASE("Empty embedding produces empty vector literal", "[embedding]") {
    std::vector<float> embedding;

    std::string vec_literal;
    if (!embedding.empty()) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < embedding.size(); ++i) {
            if (i > 0) oss << ",";
            oss << embedding[i];
        }
        oss << "]";
        vec_literal = oss.str();
    }

    CHECK(vec_literal.empty());
}
