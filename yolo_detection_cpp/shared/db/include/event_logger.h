#pragma once

#include "db_pool.h"

#include <string>
#include <vector>

namespace yolo {

/// Database event logging for detection events.
/// Uses DbPool for thread-safe connections.
struct EventLogger {
    /// Insert a new detection event (started)
    static void create_event(DbPool& db,
                             const std::string& event_id,
                             const std::string& camera_id,
                             const std::string& recording_filename,
                             const std::string& snapshot_filename);

    /// Mark event as completed with stats
    static void complete_event(DbPool& db,
                               const std::string& event_id,
                               double duration_seconds,
                               int frames_processed,
                               int detections_count);

    /// Log individual detections for an event
    struct DetectionRecord {
        std::string class_name;
        float confidence;
        float x1, y1, x2, y2;
    };

    static void log_detections(DbPool& db,
                               const std::string& event_id,
                               const std::vector<DetectionRecord>& detections);

    /// AI vision context record for LLaVA analysis results
    struct AiVisionRecord {
        std::string context_text;
        std::vector<std::string> detected_classes;
        std::string source_model = "llava:7b";
        std::string prompt_used;
        double response_time_seconds = 0;
        bool is_valid = true;
    };

    /// Log AI vision context analysis to ai_vision_context table
    static void log_ai_context(DbPool& db,
                               const std::string& event_id,
                               const std::string& camera_id,
                               const AiVisionRecord& record);
};

}  // namespace yolo
