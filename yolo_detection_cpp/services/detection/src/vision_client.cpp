#include "vision_client.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include <chrono>
#include <fstream>

using json = nlohmann::json;

namespace hms {

VisionClient::VisionClient(const yolo::LlavaConfig& config)
    : config_(config)
{
}

// libcurl write callback
static size_t curlWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* response = static_cast<std::string*>(userdata);
    response->append(ptr, size * nmemb);
    return size * nmemb;
}

VisionClient::Result VisionClient::analyze(const std::string& snapshot_path,
                                           const std::string& camera_id,
                                           const std::string& detected_class) {
    Result result;
    auto t0 = std::chrono::steady_clock::now();

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

    // 2. Build prompt
    last_prompt_ = buildPrompt(camera_id, detected_class);

    // 3. Build Ollama request body
    json body = {
        {"model", config_.model},
        {"prompt", last_prompt_},
        {"images", {base64Encode(image_data)}},
        {"stream", false}
    };
    std::string body_str = body.dump();

    // 4. Send HTTP POST to Ollama using libcurl (decoupled from Drogon event loop)
    std::string url = config_.endpoint + "/api/generate";
    std::string response_body;

    CURL* curl = curl_easy_init();
    if (!curl) {
        spdlog::error("VisionClient: curl_easy_init failed");
        return result;
    }

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, static_cast<long>(body_str.size()));
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curlWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, static_cast<long>(config_.timeout_seconds));
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);  // Thread-safe

    CURLcode res = curl_easy_perform(curl);

    auto elapsed = std::chrono::steady_clock::now() - t0;
    result.response_time_seconds = std::chrono::duration<double>(elapsed).count();

    if (res != CURLE_OK) {
        spdlog::error("VisionClient: curl error for {}: {} ({:.1f}s)",
                      camera_id, curl_easy_strerror(res), result.response_time_seconds);
        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);
        return result;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (http_code != 200) {
        spdlog::error("VisionClient: Ollama returned HTTP {}", http_code);
        return result;
    }

    // 5. Parse response
    try {
        auto j = json::parse(response_body);
        result.context = j.value("response", "");

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
    } catch (const json::exception& e) {
        spdlog::error("VisionClient: failed to parse Ollama response: {}",
                      e.what());
    }

    spdlog::info("VisionClient: {} analysis for {} in {:.1f}s valid={} text='{}'",
                 config_.model, camera_id, result.response_time_seconds,
                 result.is_valid, result.context);

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

    // Replace {max_words} and {class} placeholders
    std::string prompt = tmpl;
    std::string max_words_str = std::to_string(config_.max_words);

    for (std::string::size_type pos = 0;
         (pos = prompt.find("{max_words}", pos)) != std::string::npos; ) {
        prompt.replace(pos, 11, max_words_str);
        pos += max_words_str.size();
    }
    for (std::string::size_type pos = 0;
         (pos = prompt.find("{class}", pos)) != std::string::npos; ) {
        prompt.replace(pos, 7, detected_class);
        pos += detected_class.size();
    }

    return prompt;
}

std::string VisionClient::selectPrimaryClass(
        const std::vector<std::string>& classes) {
    // Priority: person > dog > cat > package > car
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

std::string VisionClient::base64Encode(const std::vector<unsigned char>& data) {
    static constexpr char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string encoded;
    encoded.reserve(((data.size() + 2) / 3) * 4);

    size_t i = 0;
    for (; i + 2 < data.size(); i += 3) {
        uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                     (static_cast<uint32_t>(data[i + 1]) << 8) |
                      static_cast<uint32_t>(data[i + 2]);
        encoded += table[(n >> 18) & 0x3F];
        encoded += table[(n >> 12) & 0x3F];
        encoded += table[(n >> 6)  & 0x3F];
        encoded += table[n & 0x3F];
    }

    if (i + 1 == data.size()) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        encoded += table[(n >> 18) & 0x3F];
        encoded += table[(n >> 12) & 0x3F];
        encoded += '=';
        encoded += '=';
    } else if (i + 2 == data.size()) {
        uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                     (static_cast<uint32_t>(data[i + 1]) << 8);
        encoded += table[(n >> 18) & 0x3F];
        encoded += table[(n >> 12) & 0x3F];
        encoded += table[(n >> 6)  & 0x3F];
        encoded += '=';
    }

    return encoded;
}

}  // namespace hms
