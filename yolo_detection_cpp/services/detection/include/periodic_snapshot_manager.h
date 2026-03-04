#pragma once

#include "buffer_service.h"
#include "config_manager.h"
#include "db_pool.h"

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

namespace hms {

/// Takes periodic ambient snapshots from each camera's buffer,
/// runs LLaVA description + embedding, and stores to periodic_snapshots table.
class PeriodicSnapshotManager {
public:
    PeriodicSnapshotManager(std::shared_ptr<BufferService> buffer_service,
                            std::shared_ptr<yolo::DbPool> db,
                            const yolo::AppConfig& config);
    ~PeriodicSnapshotManager();

    PeriodicSnapshotManager(const PeriodicSnapshotManager&) = delete;
    PeriodicSnapshotManager& operator=(const PeriodicSnapshotManager&) = delete;

    void start();
    void stop();

private:
    void cameraLoop(const std::string& camera_id, int interval_seconds);

    /// Save a JPEG from raw frame data, returns filename or empty on error
    std::string saveSnapshot(const FrameData& frame, const std::string& camera_id,
                             const std::string& snapshots_dir);

    /// Generate a smaller thumbnail JPEG, returns filename or empty on error
    std::string saveThumbnail(const FrameData& frame, const std::string& camera_id,
                              const std::string& snapshots_dir);

    std::shared_ptr<BufferService> buffer_service_;
    std::shared_ptr<yolo::DbPool> db_;
    yolo::AppConfig config_;
    std::atomic<bool> running_{false};
    std::vector<std::thread> threads_;
};

}  // namespace hms
