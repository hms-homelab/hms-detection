#include "periodic_snapshot_manager.h"
#include "snapshot_writer.h"
#include "vision_client.h"
#include "embedding_client.h"
#include "api_queries.h"
#include "time_utils.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

namespace hms {

PeriodicSnapshotManager::PeriodicSnapshotManager(
    std::shared_ptr<BufferService> buffer_service,
    std::shared_ptr<hms::DbPool> db,
    std::shared_ptr<GpuCoordinator> gpu_coord,
    const hms::AppConfig& config)
    : buffer_service_(std::move(buffer_service))
    , db_(std::move(db))
    , gpu_coord_(std::move(gpu_coord))
    , config_(config)
{
}

PeriodicSnapshotManager::~PeriodicSnapshotManager() {
    stop();
}

void PeriodicSnapshotManager::start() {
    if (running_) return;
    running_ = true;

    for (const auto& [cam_id, cam_config] : config_.cameras) {
        if (!cam_config.enabled) continue;
        int interval = cam_config.periodic_snapshot_interval;
        if (interval <= 0) continue;

        spdlog::info("PeriodicSnapshotManager: starting {} every {}s", cam_id, interval);
        threads_.emplace_back(&PeriodicSnapshotManager::cameraLoop, this, cam_id, interval);
    }

    if (threads_.empty()) {
        spdlog::info("PeriodicSnapshotManager: no cameras configured for periodic snapshots");
    }
}

void PeriodicSnapshotManager::stop() {
    running_ = false;
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
}

void PeriodicSnapshotManager::cameraLoop(const std::string& camera_id,
                                          int interval_seconds) {
    spdlog::info("PeriodicSnapshotManager: thread started for {}", camera_id);

    // Wait a bit on startup to let RTSP streams connect
    for (int i = 0; i < 30 && running_; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    while (running_) {
        try {
            // 1. Grab latest frame (CPU only — always safe)
            auto frame = buffer_service_->getLatestFrame(camera_id);
            if (!frame || frame->pixels.empty()) {
                spdlog::warn("PeriodicSnapshotManager: no frame for {}", camera_id);
                for (int i = 0; i < interval_seconds && running_; ++i) {
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                }
                continue;
            }

            // Deep-copy frame to avoid holding buffer references
            FrameData frame_copy;
            frame_copy.pixels = frame->pixels;
            frame_copy.width = frame->width;
            frame_copy.height = frame->height;
            frame_copy.stride = frame->stride;
            frame_copy.timestamp = frame->timestamp;
            frame_copy.frame_number = frame->frame_number;

            auto snapshots_dir = config_.timeline.snapshots_dir;

            // 2. Save full JPEG (no bounding boxes — ambient snapshot, CPU only)
            auto snapshot_filename = saveSnapshot(frame_copy, camera_id, snapshots_dir);
            if (snapshot_filename.empty()) {
                spdlog::error("PeriodicSnapshotManager: failed to save snapshot for {}", camera_id);
                for (int i = 0; i < interval_seconds && running_; ++i)
                    std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }

            // 3. Save thumbnail (CPU only)
            auto thumbnail_filename = saveThumbnail(frame_copy, camera_id, snapshots_dir);

            // 4. Run moondream vision analysis (GPU — check coordinator first)
            std::string context_text;
            bool is_valid = false;
            bool was_aborted = false;

            if (config_.periodic_vision.enabled) {
                // Skip if event is already active — don't compete for Ollama queue
                if (gpu_coord_ && gpu_coord_->isEventActive()) {
                    spdlog::info("PeriodicSnapshotManager: [{}] skipping moondream — event active",
                                 camera_id);
                } else {
                    VisionClient vision(config_.periodic_vision);
                    auto snapshot_path = snapshots_dir + "/" + snapshot_filename;

                    // Pass abort flag so curl cancels if an event fires mid-inference.
                    // The coordinator's abort flag gets set by EventManager::processEvent().
                    auto result = vision.analyze(snapshot_path, camera_id, "scene",
                        gpu_coord_ ? &gpu_coord_->abortPeriodicFlag() : nullptr);

                    context_text = result.context;
                    is_valid = result.is_valid;
                    was_aborted = result.was_aborted;

                    if (was_aborted) {
                        spdlog::info("PeriodicSnapshotManager: [{}] moondream aborted — event took priority",
                                     camera_id);
                    } else {
                        spdlog::info("PeriodicSnapshotManager: [{}] moondream: valid={} text='{}'",
                                     camera_id, is_valid, context_text);
                    }
                }
            }

            // 5. Generate embedding (only if we got valid context)
            std::vector<float> embedding;
            if (!context_text.empty() && is_valid && !was_aborted) {
                // Skip embedding if event just started
                if (!gpu_coord_ || !gpu_coord_->isEventActive()) {
                    EmbeddingClient emb_client(config_.periodic_vision.endpoint, "nomic-embed-text");
                    embedding = emb_client.embed(context_text);
                    if (embedding.empty()) {
                        spdlog::warn("PeriodicSnapshotManager: embedding failed for {}", camera_id);
                    }
                }
            }

            // 6. Insert into DB (always — even without context, the snapshot is valuable)
            if (db_) {
                std::string model_used = was_aborted ? "" : config_.periodic_vision.model;
                hms::api_queries::insert_periodic_snapshot(
                    *db_, camera_id, snapshot_filename, thumbnail_filename,
                    context_text, embedding, model_used, is_valid);
            }

            spdlog::info("PeriodicSnapshotManager: completed snapshot for {} -> {}{}",
                        camera_id, snapshot_filename,
                        was_aborted ? " (no context — aborted)" :
                        context_text.empty() ? " (no context)" : "");

        } catch (const std::exception& e) {
            spdlog::error("PeriodicSnapshotManager: error for {}: {}", camera_id, e.what());
        }

        // Sleep for interval (check running_ every second for responsive shutdown)
        for (int i = 0; i < interval_seconds && running_; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }

    spdlog::info("PeriodicSnapshotManager: thread stopped for {}", camera_id);
}

std::string PeriodicSnapshotManager::saveSnapshot(const FrameData& frame,
                                                    const std::string& camera_id,
                                                    const std::string& snapshots_dir) {
    // Generate filename: {camera_id}_periodic_{YYYYMMDD}_{HHMMSS}.jpg
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time_t_now, &tm);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_periodic_%04d%02d%02d_%02d%02d%02d.jpg",
                  camera_id.c_str(),
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::string filename(buf);

    // Encode to JPEG
    auto jpeg = SnapshotWriter::encodeJpeg(frame.pixels.data(),
                                            frame.width, frame.height, frame.stride);
    if (jpeg.empty()) return {};

    // Write file
    fs::create_directories(snapshots_dir);
    auto path = fs::path(snapshots_dir) / filename;
    std::ofstream f(path, std::ios::binary);
    if (!f) return {};
    f.write(jpeg.data(), static_cast<std::streamsize>(jpeg.size()));
    f.close();

    return filename;
}

std::string PeriodicSnapshotManager::saveThumbnail(const FrameData& frame,
                                                    const std::string& camera_id,
                                                    const std::string& snapshots_dir) {
    // Generate thumbnail filename
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&time_t_now, &tm);

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s_periodic_%04d%02d%02d_%02d%02d%02d_thumb.jpg",
                  camera_id.c_str(),
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    std::string filename(buf);

    // For thumbnail, we scale down. Since SnapshotWriter::encodeJpeg works with
    // the full frame, and sub-streams are already 640x480, the "thumbnail" is
    // just a higher-compression JPEG of the same frame.
    // (Sub-stream is already small enough — 640x480 → ~15-25KB JPEG)
    auto jpeg = SnapshotWriter::encodeJpeg(frame.pixels.data(),
                                            frame.width, frame.height, frame.stride);
    if (jpeg.empty()) return {};

    auto path = fs::path(snapshots_dir) / filename;
    std::ofstream f(path, std::ios::binary);
    if (!f) return {};
    f.write(jpeg.data(), static_cast<std::streamsize>(jpeg.size()));
    f.close();

    return filename;
}

}  // namespace hms
