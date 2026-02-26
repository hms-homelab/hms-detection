#pragma once

#include "camera_buffer.h"
#include "detection_engine.h"
#include "detection_worker.h"
#include "frame_data.h"
#include "rtsp_capture.h"
#include "config_manager.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace hms {

/// Orchestrates RTSP capture, ring buffers, and detection workers for all cameras.
class BufferService {
public:
    struct CameraStats {
        std::string camera_id;
        std::string camera_name;
        size_t buffer_size = 0;
        size_t max_frames = 0;
        uint64_t frames_captured = 0;
        uint64_t reconnect_count = 0;
        uint64_t consecutive_failures = 0;
        bool is_connected = false;
        bool is_healthy = false;
        int frame_width = 0;
        int frame_height = 0;
        Clock::time_point last_frame_time;
    };

    explicit BufferService(const yolo::AppConfig& config);
    ~BufferService();

    BufferService(const BufferService&) = delete;
    BufferService& operator=(const BufferService&) = delete;

    void startAll();
    void stopAll();

    // --- Detection ---

    /// Load the ONNX model without starting continuous workers
    void loadDetectionModel();

    /// Start continuous detection workers for all cameras
    void startDetection();

    /// Stop all detection workers
    void stopDetection();

    /// Get the shared detection engine (may be null if model not loaded)
    std::shared_ptr<DetectionEngine> getDetectionEngine() const;

    /// Get latest detection result for a camera
    std::optional<DetectionResult> getDetectionResult(const std::string& camera_id) const;

    /// Get detection stats for all cameras
    std::unordered_map<std::string, DetectionWorker::Stats> getDetectionStats() const;

    // --- Frame access ---

    /// Get the most recent frame for a camera, or nullptr.
    std::shared_ptr<FrameData> getLatestFrame(const std::string& camera_id) const;

    /// Get the ring buffer for a camera (for future pre-roll access).
    std::shared_ptr<CameraBuffer> getCameraBuffer(const std::string& camera_id) const;

    /// Get stats for all cameras.
    std::vector<CameraStats> getAllStats() const;

    /// True if at least one camera is connected and has frames.
    bool isHealthy() const;

    /// Get list of camera IDs.
    std::vector<std::string> cameraIds() const;

private:
    struct CameraState {
        std::string name;
        std::shared_ptr<FramePool> pool;
        std::shared_ptr<CameraBuffer> buffer;
        std::unique_ptr<RtspCapture> capture;
    };

    yolo::AppConfig config_;
    std::unordered_map<std::string, CameraState> cameras_;

    // Detection
    std::shared_ptr<DetectionEngine> detection_engine_;
    std::unordered_map<std::string, std::unique_ptr<DetectionWorker>> detection_workers_;
};

}  // namespace hms
