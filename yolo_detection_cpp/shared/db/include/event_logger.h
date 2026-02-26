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
};

}  // namespace yolo
