#include <catch2/catch_test_macros.hpp>
#include "vision_client.h"

#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

yolo::LlavaConfig makeConfig() {
    yolo::LlavaConfig cfg;
    cfg.enabled = true;
    cfg.endpoint = "http://localhost:11434";
    cfg.model = "llava:7b";
    cfg.max_words = 15;
    cfg.timeout_seconds = 30;
    cfg.default_prompt = "In {max_words} words or less, describe the {class}.";
    return cfg;
}

} // anonymous namespace

TEST_CASE("VisionClient::buildPrompt uses camera-specific template", "[vision]") {
    auto cfg = makeConfig();
    cfg.prompts["patio"] = "Look at the {class} on the patio in {max_words} words.";

    hms::VisionClient client(cfg);
    auto prompt = client.buildPrompt("patio", "person");

    CHECK(prompt == "Look at the person on the patio in 15 words.");
}

TEST_CASE("VisionClient::buildPrompt falls back to default key", "[vision]") {
    auto cfg = makeConfig();
    cfg.prompts["default"] = "Default: describe the {class}.";

    hms::VisionClient client(cfg);
    auto prompt = client.buildPrompt("unknown_camera", "dog");

    CHECK(prompt == "Default: describe the dog.");
}

TEST_CASE("VisionClient::buildPrompt falls back to default_prompt", "[vision]") {
    auto cfg = makeConfig();
    // No prompts map entries at all

    hms::VisionClient client(cfg);
    auto prompt = client.buildPrompt("any_camera", "cat");

    CHECK(prompt == "In 15 words or less, describe the cat.");
}

TEST_CASE("VisionClient::buildPrompt replaces {class} and {max_words}", "[vision]") {
    auto cfg = makeConfig();
    cfg.max_words = 25;
    cfg.default_prompt = "Describe the {class} in exactly {max_words} words. The {class} is important.";

    hms::VisionClient client(cfg);
    auto prompt = client.buildPrompt("cam1", "car");

    CHECK(prompt == "Describe the car in exactly 25 words. The car is important.");
}

TEST_CASE("VisionClient::selectPrimaryClass returns person first", "[vision]") {
    std::vector<std::string> classes = {"car", "dog", "person", "cat"};
    CHECK(hms::VisionClient::selectPrimaryClass(classes) == "person");
}

TEST_CASE("VisionClient::selectPrimaryClass returns car when no priority match", "[vision]") {
    std::vector<std::string> classes = {"bicycle", "car", "truck"};
    CHECK(hms::VisionClient::selectPrimaryClass(classes) == "car");
}

TEST_CASE("VisionClient::selectPrimaryClass returns first for unknown classes", "[vision]") {
    std::vector<std::string> classes = {"bicycle", "truck"};
    CHECK(hms::VisionClient::selectPrimaryClass(classes) == "bicycle");
}

TEST_CASE("VisionClient::selectPrimaryClass returns object for empty list", "[vision]") {
    std::vector<std::string> classes;
    CHECK(hms::VisionClient::selectPrimaryClass(classes) == "object");
}

TEST_CASE("VisionClient::base64Encode roundtrip", "[vision]") {
    // Known test vector: "Hello" -> "SGVsbG8="
    std::vector<unsigned char> data = {'H', 'e', 'l', 'l', 'o'};
    auto encoded = hms::VisionClient::base64Encode(data);
    CHECK(encoded == "SGVsbG8=");
}

TEST_CASE("VisionClient::base64Encode handles empty input", "[vision]") {
    std::vector<unsigned char> data;
    auto encoded = hms::VisionClient::base64Encode(data);
    CHECK(encoded.empty());
}

TEST_CASE("VisionClient::base64Encode handles single byte", "[vision]") {
    std::vector<unsigned char> data = {'A'};  // 'A' = 0x41 -> "QQ=="
    auto encoded = hms::VisionClient::base64Encode(data);
    CHECK(encoded == "QQ==");
}

TEST_CASE("VisionClient::base64Encode handles two bytes", "[vision]") {
    std::vector<unsigned char> data = {'A', 'B'};  // -> "QUI="
    auto encoded = hms::VisionClient::base64Encode(data);
    CHECK(encoded == "QUI=");
}

TEST_CASE("VisionClient::base64Encode handles three bytes (no padding)", "[vision]") {
    std::vector<unsigned char> data = {'A', 'B', 'C'};  // -> "QUJD"
    auto encoded = hms::VisionClient::base64Encode(data);
    CHECK(encoded == "QUJD");
}

TEST_CASE("VisionClient::analyze returns invalid for nonexistent file", "[vision]") {
    auto cfg = makeConfig();
    hms::VisionClient client(cfg);

    // This won't make a network call — it fails at the file read stage
    auto result = client.analyze("/nonexistent/snapshot.jpg", "patio", "person");

    CHECK_FALSE(result.is_valid);
    CHECK(result.context.empty());
}
