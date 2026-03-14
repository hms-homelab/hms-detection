#include "vision_client.h"

#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>

namespace hms {

VisionClient::VisionClient(const hms::LlavaConfig& config)
    : config_(config)
{
}

LLMConfig VisionClient::makeLLMConfig(const LlavaConfig& lc) {
    LLMConfig c;
    c.enabled = lc.enabled;
    c.provider = LLMClient::parseProvider(lc.provider);
    c.endpoint = lc.endpoint;
    c.model = lc.model;
    c.api_key = lc.api_key;
    c.temperature = lc.temperature;
    c.max_tokens = lc.max_tokens;
    c.timeout_seconds = lc.timeout_seconds;
    c.connect_timeout_seconds = 10;
    c.keep_alive_seconds = 0;  // always unload after vision call
    return c;
}

VisionClient::Result VisionClient::analyze(const std::string& snapshot_path,
                                           const std::string& camera_id,
                                           const std::string& detected_class,
                                           const std::atomic<bool>* abort_flag) {
    Result result;

    // Check abort before doing any work
    if (abort_flag && abort_flag->load(std::memory_order_acquire)) {
        result.was_aborted = true;
        spdlog::info("VisionClient: aborted before start for {}", camera_id);
        return result;
    }

    // 1. Read snapshot file
    std::ifstream file(snapshot_path, std::ios::binary);
    if (!file) {
        spdlog::error("VisionClient: cannot open snapshot: {}", snapshot_path);
        return result;
    }
    std::vector<unsigned char> image_data(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    file.close();

    if (image_data.empty()) {
        spdlog::error("VisionClient: empty snapshot file: {}", snapshot_path);
        return result;
    }

    // Check abort after file read
    if (abort_flag && abort_flag->load(std::memory_order_acquire)) {
        result.was_aborted = true;
        spdlog::info("VisionClient: aborted after file read for {}", camera_id);
        return result;
    }

    // 2. Build prompt
    last_prompt_ = buildPrompt(camera_id, detected_class);

    // 3. Delegate to LLMClient for the actual API call
    LLMClient client(makeLLMConfig(config_));
    LLMImage img{LLMClient::base64Encode(image_data), "image/jpeg"};
    auto response = client.generateVision(last_prompt_, {img}, abort_flag);

    result.response_time_seconds = response.elapsed_seconds;
    result.was_aborted = response.was_aborted;

    if (response.was_aborted) {
        spdlog::info("VisionClient: request aborted for {} after {:.1f}s",
                      camera_id, result.response_time_seconds);
        return result;
    }

    if (!response.text) {
        spdlog::error("VisionClient: no response from {} for {} ({:.1f}s)",
                      config_.model, camera_id, result.response_time_seconds);
        return result;
    }

    // 4. Validate response
    result.context = response.text.value();

    // Trim whitespace
    auto ltrim = result.context.find_first_not_of(" \t\n\r");
    if (ltrim != std::string::npos) {
        result.context = result.context.substr(ltrim);
    }
    auto rtrim = result.context.find_last_not_of(" \t\n\r");
    if (rtrim != std::string::npos) {
        result.context = result.context.substr(0, rtrim + 1);
    }

    // Validate: >= 15 chars and contains at least one space
    result.is_valid = result.context.size() >= 15 &&
                      result.context.find(' ') != std::string::npos;

    if (!result.is_valid) {
        spdlog::warn("VisionClient: invalid response (len={}, text='{}')",
                     result.context.size(), result.context);
    }

    spdlog::info("VisionClient: {} ({}) analysis for {} in {:.1f}s valid={} text='{}'",
                 config_.model, config_.provider, camera_id,
                 result.response_time_seconds, result.is_valid, result.context);

    return result;
}

std::string VisionClient::buildPrompt(const std::string& camera_id,
                                      const std::string& detected_class) {
    // Look up camera-specific prompt, fall back to "default" key, then to default_prompt
    std::string tmpl;
    auto it = config_.prompts.find(camera_id);
    if (it != config_.prompts.end()) {
        tmpl = it->second;
    } else {
        auto def_it = config_.prompts.find("default");
        if (def_it != config_.prompts.end()) {
            tmpl = def_it->second;
        } else {
            tmpl = config_.default_prompt;
        }
    }

    return LLMClient::substituteTemplate(tmpl, {
        {"max_words", std::to_string(config_.max_words)},
        {"class", detected_class}
    });
}

std::string VisionClient::selectPrimaryClass(
        const std::vector<std::string>& classes) {
    static const std::vector<std::string> priority = {
        "person", "dog", "cat", "package", "car"
    };

    for (const auto& p : priority) {
        for (const auto& c : classes) {
            if (c == p) return p;
        }
    }

    return classes.empty() ? "object" : classes.front();
}

}  // namespace hms
