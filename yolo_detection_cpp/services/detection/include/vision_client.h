#pragma once

#include "config_manager.h"

#include <string>

namespace hms {

/// Synchronous Ollama LLaVA client for vision context generation.
/// Uses Drogon's HttpClient internally with std::promise/future for blocking.
class VisionClient {
public:
    struct Result {
        std::string context;
        double response_time_seconds = 0;
        bool is_valid = false;
    };

    explicit VisionClient(const yolo::LlavaConfig& config);

    /// Synchronous call — blocks until Ollama responds (up to timeout).
    /// Safe to call from event threads (outside Drogon event loop).
    Result analyze(const std::string& snapshot_path,
                   const std::string& camera_id,
                   const std::string& detected_class);

    /// The prompt used in the last analyze() call
    const std::string& lastPrompt() const { return last_prompt_; }

    /// Select the highest-priority class from a set of detected classes.
    /// Priority: person > dog > cat > package > car > first in set.
    static std::string selectPrimaryClass(
        const std::vector<std::string>& classes);

private:
    std::string buildPrompt(const std::string& camera_id,
                            const std::string& detected_class);

    static std::string base64Encode(const std::vector<unsigned char>& data);

    yolo::LlavaConfig config_;
    std::string last_prompt_;
};

}  // namespace hms
