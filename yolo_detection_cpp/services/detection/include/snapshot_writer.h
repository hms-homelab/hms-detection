#pragma once

#include "frame_data.h"
#include "detection_engine.h"

#include <string>
#include <vector>

namespace hms {

/// Save annotated JPEG snapshots to disk.
/// Reuses drawBoundingBoxes + encodeJpeg patterns from detection_controller.
struct SnapshotWriter {
    /// Draw bounding boxes on BGR24 pixel data (modifies in place)
    static void drawBoundingBoxes(std::vector<uint8_t>& pixels,
                                   int width, int height, int stride,
                                   const std::vector<Detection>& detections);

    /// Encode BGR24 frame to JPEG bytes using FFmpeg MJPEG encoder
    static std::string encodeJpeg(const uint8_t* pixels,
                                   int width, int height, int stride);

    /// Save annotated snapshot to disk. Returns the full file path, or empty on error.
    /// output_dir: e.g. /mnt/ssd/snapshots
    static std::string save(const FrameData& frame,
                            const std::vector<Detection>& detections,
                            const std::string& camera_id,
                            const std::string& output_dir = "/mnt/ssd/snapshots");
};

}  // namespace hms
