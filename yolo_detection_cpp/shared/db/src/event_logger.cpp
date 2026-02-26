#include "event_logger.h"

#include <spdlog/spdlog.h>
#include <pqxx/pqxx>
#include <chrono>

namespace yolo {

void EventLogger::create_event(DbPool& db,
                                const std::string& event_id,
                                const std::string& camera_id,
                                const std::string& recording_filename,
                                const std::string& snapshot_filename) {
    try {
        auto conn = db.acquire();
        pqxx::work txn(*conn);

        txn.exec(R"(
            INSERT INTO detection_events
                (event_id, camera_id, camera_name, started_at, status,
                 recording_url, snapshot_url)
            VALUES ($1, $2, $3, NOW(), 'recording', $4, $5)
        )", pqxx::params{
            event_id, camera_id, camera_id,
            recording_filename, snapshot_filename
        });

        txn.commit();
        spdlog::debug("EventLogger: created event {} for {}", event_id, camera_id);

    } catch (const std::exception& e) {
        spdlog::error("EventLogger: failed to create event: {}", e.what());
    }
}

void EventLogger::complete_event(DbPool& db,
                                  const std::string& event_id,
                                  double duration_seconds,
                                  int frames_processed,
                                  int detections_count) {
    try {
        auto conn = db.acquire();
        pqxx::work txn(*conn);

        txn.exec(R"(
            UPDATE detection_events
            SET ended_at = NOW(),
                duration_seconds = $2,
                total_detections = $3,
                status = 'completed'
            WHERE event_id = $1
        )", pqxx::params{event_id, duration_seconds, detections_count});

        txn.commit();
        spdlog::debug("EventLogger: completed event {} ({:.1f}s, {} detections)",
                      event_id, duration_seconds, detections_count);

    } catch (const std::exception& e) {
        spdlog::error("EventLogger: failed to complete event: {}", e.what());
    }
}

void EventLogger::log_detections(DbPool& db,
                                  const std::string& event_id,
                                  const std::vector<DetectionRecord>& detections) {
    if (detections.empty()) return;

    try {
        auto conn = db.acquire();
        pqxx::work txn(*conn);

        for (const auto& det : detections) {
            txn.exec(R"(
                INSERT INTO detections
                    (event_id, class_name, confidence, x1, y1, x2, y2)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            )", pqxx::params{
                event_id, det.class_name,
                static_cast<double>(det.confidence),
                static_cast<double>(det.x1), static_cast<double>(det.y1),
                static_cast<double>(det.x2), static_cast<double>(det.y2)
            });
        }

        txn.commit();
        spdlog::debug("EventLogger: logged {} detections for event {}",
                      detections.size(), event_id);

    } catch (const std::exception& e) {
        spdlog::error("EventLogger: failed to log detections: {}", e.what());
    }
}

}  // namespace yolo
