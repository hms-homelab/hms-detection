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
                    (event_id, class_name, confidence,
                     bbox_x1, bbox_y1, bbox_x2, bbox_y2, detected_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
            )", pqxx::params{
                event_id, det.class_name,
                static_cast<double>(det.confidence),
                static_cast<int>(det.x1), static_cast<int>(det.y1),
                static_cast<int>(det.x2), static_cast<int>(det.y2)
            });
        }

        txn.commit();
        spdlog::debug("EventLogger: logged {} detections for event {}",
                      detections.size(), event_id);

    } catch (const std::exception& e) {
        spdlog::error("EventLogger: failed to log detections: {}", e.what());
    }
}

void EventLogger::log_ai_context(DbPool& db,
                                  const std::string& event_id,
                                  const std::string& camera_id,
                                  const AiVisionRecord& record) {
    try {
        auto conn = db.acquire();
        pqxx::work txn(*conn);

        // Format detected_classes as PostgreSQL text array literal: {person,car}
        std::string pg_array = "{";
        for (size_t i = 0; i < record.detected_classes.size(); ++i) {
            if (i > 0) pg_array += ",";
            pg_array += record.detected_classes[i];
        }
        pg_array += "}";

        txn.exec(R"(
            INSERT INTO ai_vision_context
                (event_id, camera_id, context_text, source_model, prompt_used,
                 detected_classes, response_time_seconds, is_valid, analyzed_at)
            VALUES ($1, $2, $3, $4, $5, $6::text[], $7, $8, CURRENT_TIMESTAMP)
        )", pqxx::params{
            event_id, camera_id,
            record.context_text,
            record.source_model,
            record.prompt_used,
            pg_array,
            record.response_time_seconds,
            record.is_valid
        });

        txn.commit();
        spdlog::debug("EventLogger: logged AI context for event {} (valid={})",
                      event_id, record.is_valid);

    } catch (const std::exception& e) {
        spdlog::error("EventLogger: failed to log AI context: {}", e.what());
    }
}

}  // namespace yolo
