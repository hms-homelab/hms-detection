#include "detection_worker.h"

#include <spdlog/spdlog.h>

#include <chrono>

namespace hms {

DetectionWorker::DetectionWorker(const std::string& camera_id,
                                 std::shared_ptr<CameraBuffer> buffer,
                                 std::shared_ptr<DetectionEngine> engine,
                                 const yolo::CameraConfig& camera_config,
                                 const yolo::DetectionConfig& detection_config)
    : camera_id_(camera_id)
    , buffer_(std::move(buffer))
    , engine_(std::move(engine))
    , confidence_threshold_(static_cast<float>(
          camera_config.confidence_threshold > 0
              ? camera_config.confidence_threshold
              : detection_config.confidence_threshold))
    , iou_threshold_(static_cast<float>(detection_config.iou_threshold))
    , sample_interval_ms_(333)  // ~3 fps sampling
{
    // Use camera-specific classes if set, otherwise global detection classes
    if (!camera_config.classes.empty()) {
        filter_classes_ = camera_config.classes;
    } else {
        filter_classes_ = detection_config.classes;
    }
}

DetectionWorker::~DetectionWorker() {
    stop();
}

void DetectionWorker::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&DetectionWorker::detectionLoop, this);
    spdlog::info("[{}] Detection worker started (conf={:.2f}, iou={:.2f}, interval={}ms)",
                 camera_id_, confidence_threshold_, iou_threshold_, sample_interval_ms_);
}

void DetectionWorker::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    spdlog::info("[{}] Detection worker stopped", camera_id_);
}

std::optional<DetectionResult> DetectionWorker::getLatestResult() const {
    std::shared_lock lock(result_mutex_);
    return latest_result_;
}

DetectionWorker::Stats DetectionWorker::stats() const {
    return Stats{
        .frames_processed = frames_processed_.load(),
        .detections_found = detections_found_.load(),
        .avg_inference_ms = avg_inference_ms_.load(),
        .is_running = running_.load(),
    };
}

void DetectionWorker::detectionLoop() {
    uint64_t last_frame_number = 0;
    double total_inference_ms = 0;

    while (running_) {
        auto frame = buffer_->getLatestFrame();
        if (!frame || frame->frame_number == last_frame_number) {
            std::this_thread::sleep_for(std::chrono::milliseconds(sample_interval_ms_));
            continue;
        }

        last_frame_number = frame->frame_number;
        auto start = std::chrono::steady_clock::now();

        auto detections = engine_->detect(*frame, confidence_threshold_,
                                          iou_threshold_, filter_classes_);

        auto elapsed = std::chrono::steady_clock::now() - start;
        double ms = std::chrono::duration<double, std::milli>(elapsed).count();

        // Update result
        {
            std::unique_lock lock(result_mutex_);
            latest_result_ = DetectionResult{
                .detections = std::move(detections),
                .timestamp = frame->timestamp,
                .frame_number = frame->frame_number,
            };
        }

        // Update stats
        uint64_t count = frames_processed_.fetch_add(1) + 1;
        detections_found_.fetch_add(latest_result_->detections.size());

        total_inference_ms += ms;
        avg_inference_ms_.store(total_inference_ms / count);

        // Sleep for remaining interval time
        auto process_time = std::chrono::steady_clock::now() - start;
        auto remaining = std::chrono::milliseconds(sample_interval_ms_) - process_time;
        if (remaining > std::chrono::milliseconds(0)) {
            std::this_thread::sleep_for(remaining);
        }
    }
}

}  // namespace hms
