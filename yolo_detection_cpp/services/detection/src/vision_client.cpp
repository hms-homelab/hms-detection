#include "vision_client.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <drogon/HttpClient.h>

#include <chrono>
#include <fstream>
#include <future>

using json = nlohmann::json;

namespace hms {

VisionClient::VisionClient(const yolo::LlavaConfig& config)
    : config_(config)
{
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

    // 4. Send HTTP POST to Ollama /api/generate using Drogon HttpClient
    //    Use std::promise/future to get synchronous behavior from async client.
    auto promise = std::make_shared<std::promise<Result>>();
    auto future = promise->get_future();

    auto client = drogon::HttpClient::newHttpClient(config_.endpoint);
    client->setUserAgent("hms-detection/1.0");

    auto req = drogon::HttpRequest::newHttpRequest();
    req->setPath("/api/generate");
    req->setMethod(drogon::Post);
    req->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    req->setBody(body_str);

    double timeout_sec = static_cast<double>(config_.timeout_seconds);

    client->sendRequest(req,
        [promise, t0](drogon::ReqResult rr,
                      const drogon::HttpResponsePtr& resp) {
            Result res;
            auto elapsed = std::chrono::steady_clock::now() - t0;
            res.response_time_seconds =
                std::chrono::duration<double>(elapsed).count();

            if (rr != drogon::ReqResult::Ok || !resp) {
                spdlog::error("VisionClient: HTTP request failed (result={})",
                              static_cast<int>(rr));
                promise->set_value(res);
                return;
            }

            auto status = resp->statusCode();
            if (status != drogon::k200OK) {
                spdlog::error("VisionClient: Ollama returned HTTP {}",
                              static_cast<int>(status));
                promise->set_value(res);
                return;
            }

            // Parse response
            try {
                auto j = json::parse(resp->body());
                res.context = j.value("response", "");

                // Trim whitespace
                auto ltrim = res.context.find_first_not_of(" \t\n\r");
                if (ltrim != std::string::npos) {
                    res.context = res.context.substr(ltrim);
                }
                auto rtrim = res.context.find_last_not_of(" \t\n\r");
                if (rtrim != std::string::npos) {
                    res.context = res.context.substr(0, rtrim + 1);
                }

                // Validate: >= 15 chars and contains at least one space
                res.is_valid = res.context.size() >= 15 &&
                               res.context.find(' ') != std::string::npos;

                if (!res.is_valid) {
                    spdlog::warn("VisionClient: invalid response (len={}, text='{}')",
                                 res.context.size(), res.context);
                }
            } catch (const json::exception& e) {
                spdlog::error("VisionClient: failed to parse Ollama response: {}",
                              e.what());
            }

            promise->set_value(res);
        },
        timeout_sec);

    // Block until response arrives (or timeout)
    try {
        auto status = future.wait_for(
            std::chrono::seconds(config_.timeout_seconds + 5));
        if (status == std::future_status::timeout) {
            spdlog::error("VisionClient: future timed out for {}", camera_id);
            return result;
        }
        result = future.get();
    } catch (const std::exception& e) {
        spdlog::error("VisionClient: exception waiting for response: {}", e.what());
    }

    auto total = std::chrono::steady_clock::now() - t0;
    result.response_time_seconds = std::chrono::duration<double>(total).count();

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
