#include "controllers/health_controller.h"
#include "buffer_service.h"
#include "time_utils.h"

#include <drogon/HttpResponse.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace hms {

void HealthController::setBufferService(std::shared_ptr<BufferService> svc) {
    buffer_service_ = std::move(svc);
}

void HealthController::getHealth(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    using json = nlohmann::json;

    auto stats = buffer_service_->getAllStats();
    bool healthy = buffer_service_->isHealthy();

    json cameras_json = json::object();
    for (const auto& s : stats) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - s.last_frame_time).count();

        cameras_json[s.camera_id] = {
            {"name", s.camera_name},
            {"buffer_size", s.buffer_size},
            {"max_frames", s.max_frames},
            {"frames_captured", s.frames_captured},
            {"reconnect_count", s.reconnect_count},
            {"consecutive_failures", s.consecutive_failures},
            {"is_connected", s.is_connected},
            {"is_healthy", s.is_healthy},
            {"frame_width", s.frame_width},
            {"frame_height", s.frame_height},
            {"last_frame_ms_ago", s.frames_captured > 0 ? elapsed_ms : -1},
        };
    }

    // Detection stats
    json detection_json = json::object();
    auto engine = buffer_service_->getDetectionEngine();
    detection_json["model_loaded"] = (engine && engine->isLoaded());
    if (engine && engine->isLoaded()) {
        detection_json["input_size"] = std::to_string(engine->inputWidth()) + "x"
                                       + std::to_string(engine->inputHeight());
    }

    auto det_stats = buffer_service_->getDetectionStats();
    for (const auto& [cam_id, ds] : det_stats) {
        json cam_det = {
            {"frames_processed", ds.frames_processed},
            {"detections_found", ds.detections_found},
            {"avg_inference_ms", std::round(ds.avg_inference_ms * 10) / 10},
            {"is_running", ds.is_running},
        };

        // Include last detection class names
        auto result = buffer_service_->getDetectionResult(cam_id);
        if (result && !result->detections.empty()) {
            json last_classes = json::array();
            for (const auto& d : result->detections) {
                last_classes.push_back(d.class_name);
            }
            cam_det["last_detections"] = last_classes;
        }

        detection_json[cam_id] = cam_det;
    }

    json result = {
        {"service", "hms-detection"},
        {"status", healthy ? "healthy" : "degraded"},
        {"timestamp", yolo::time_utils::now_iso8601()},
        {"cameras", cameras_json},
        {"detection", detection_json},
    };

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(healthy ? drogon::k200OK : drogon::k503ServiceUnavailable);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(result.dump());
    callback(resp);
}

}  // namespace hms
