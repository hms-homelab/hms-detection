#include "buffer_service.h"

#include <spdlog/spdlog.h>

namespace hms {

BufferService::BufferService(const yolo::AppConfig& config) {
    for (const auto& [id, cam_cfg] : config.cameras) {
        if (!cam_cfg.enabled) {
            spdlog::info("[{}] Camera disabled, skipping", id);
            continue;
        }

        size_t buffer_capacity = static_cast<size_t>(
            config.buffer.preroll_seconds * config.buffer.fps);
        if (buffer_capacity == 0) buffer_capacity = 75;  // fallback

        // Pool size: buffer capacity + 15 headroom for in-flight frames
        size_t pool_size = buffer_capacity + 15;

        auto pool = std::make_shared<FramePool>(pool_size);
        auto buffer = std::make_shared<CameraBuffer>(buffer_capacity);

        auto capture = std::make_unique<RtspCapture>(
            id, cam_cfg.rtsp_url, pool,
            [buf = buffer](std::shared_ptr<FrameData> frame) {
                buf->push(std::move(frame));
            });

        cameras_[id] = CameraState{
            .name = cam_cfg.name,
            .pool = std::move(pool),
            .buffer = std::move(buffer),
            .capture = std::move(capture),
        };

        spdlog::info("[{}] Configured: pool={}, buffer={}", id, pool_size, buffer_capacity);
    }
}

BufferService::~BufferService() {
    stopAll();
}

void BufferService::startAll() {
    spdlog::info("Starting capture for {} camera(s)", cameras_.size());
    for (auto& [id, state] : cameras_) {
        state.capture->start();
    }
}

void BufferService::stopAll() {
    spdlog::info("Stopping all captures");
    for (auto& [id, state] : cameras_) {
        state.capture->stop();
    }
}

std::shared_ptr<FrameData> BufferService::getLatestFrame(const std::string& camera_id) const {
    auto it = cameras_.find(camera_id);
    if (it == cameras_.end()) return nullptr;
    return it->second.buffer->getLatestFrame();
}

std::shared_ptr<CameraBuffer> BufferService::getCameraBuffer(const std::string& camera_id) const {
    auto it = cameras_.find(camera_id);
    if (it == cameras_.end()) return nullptr;
    return it->second.buffer;
}

std::vector<BufferService::CameraStats> BufferService::getAllStats() const {
    std::vector<CameraStats> result;
    result.reserve(cameras_.size());

    for (const auto& [id, state] : cameras_) {
        auto capture_stats = state.capture->stats();
        auto buf_size = state.buffer->size();

        result.push_back(CameraStats{
            .camera_id = id,
            .camera_name = state.name,
            .buffer_size = buf_size,
            .max_frames = state.buffer->capacity(),
            .frames_captured = capture_stats.frames_captured,
            .reconnect_count = capture_stats.reconnect_count,
            .consecutive_failures = capture_stats.consecutive_failures,
            .is_connected = capture_stats.is_connected,
            .is_healthy = capture_stats.is_connected && buf_size > 0,
            .frame_width = capture_stats.frame_width,
            .frame_height = capture_stats.frame_height,
            .last_frame_time = capture_stats.last_frame_time,
        });
    }

    return result;
}

bool BufferService::isHealthy() const {
    for (const auto& [id, state] : cameras_) {
        auto s = state.capture->stats();
        if (s.is_connected && state.buffer->size() > 0) return true;
    }
    return false;
}

std::vector<std::string> BufferService::cameraIds() const {
    std::vector<std::string> ids;
    ids.reserve(cameras_.size());
    for (const auto& [id, _] : cameras_) {
        ids.push_back(id);
    }
    return ids;
}

}  // namespace hms
