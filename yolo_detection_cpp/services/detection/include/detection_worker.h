#pragma once

#include "camera_buffer.h"
#include "detection_engine.h"
#include "config_manager.h"

#include <atomic>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace hms {

struct DetectionResult {
    std::vector<Detection> detections;
    Clock::time_point timestamp;
    uint64_t frame_number = 0;
};

class DetectionWorker {
public:
    DetectionWorker(const std::string& camera_id,
                    std::shared_ptr<CameraBuffer> buffer,
                    std::shared_ptr<DetectionEngine> engine,
                    const yolo::CameraConfig& camera_config,
                    const yolo::DetectionConfig& detection_config);

    ~DetectionWorker();

    DetectionWorker(const DetectionWorker&) = delete;
    DetectionWorker& operator=(const DetectionWorker&) = delete;

    void start();
    void stop();

    /// Latest detection result (thread-safe read)
    std::optional<DetectionResult> getLatestResult() const;

    struct Stats {
        uint64_t frames_processed = 0;
        uint64_t detections_found = 0;
        double avg_inference_ms = 0;
        bool is_running = false;
    };

    Stats stats() const;

private:
    void detectionLoop();

    std::string camera_id_;
    std::shared_ptr<CameraBuffer> buffer_;
    std::shared_ptr<DetectionEngine> engine_;
    std::vector<std::string> filter_classes_;
    float confidence_threshold_;
    float iou_threshold_;
    int sample_interval_ms_;

    mutable std::shared_mutex result_mutex_;
    std::optional<DetectionResult> latest_result_;

    std::atomic<bool> running_{false};
    std::thread thread_;

    // Stats
    std::atomic<uint64_t> frames_processed_{0};
    std::atomic<uint64_t> detections_found_{0};
    std::atomic<double> avg_inference_ms_{0};
};

}  // namespace hms
