#include "event_manager.h"
#include "event_logger.h"
#include "vision_client.h"
#include "time_utils.h"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <random>

using json = nlohmann::json;

namespace hms {

EventManager::EventManager(std::shared_ptr<BufferService> buffer_service,
                           std::shared_ptr<yolo::MqttClient> mqtt,
                           std::shared_ptr<yolo::DbPool> db,
                           const yolo::AppConfig& config)
    : buffer_service_(std::move(buffer_service))
    , mqtt_(std::move(mqtt))
    , db_(std::move(db))
    , config_(config)
{
}

EventManager::~EventManager() {
    stop();
}

void EventManager::start() {
    running_ = true;

    if (!mqtt_ || !mqtt_->isConnected()) {
        spdlog::warn("EventManager: MQTT not connected, skipping subscriptions");
        return;
    }

    // Subscribe to motion start/stop topics
    std::vector<std::string> topics = {
        "camera/event/motion/start",
        "camera/event/motion/stop",
    };

    mqtt_->subscribe(topics, [this](const std::string& topic, const std::string& payload) {
        if (!running_) return;

        try {
            auto msg = json::parse(payload);
            std::string camera_id = msg.value("camera_id", "");
            if (camera_id.empty()) {
                spdlog::warn("EventManager: received message with no camera_id on {}", topic);
                return;
            }

            if (topic == "camera/event/motion/start") {
                int post_roll = msg.value("post_roll_seconds", 5);
                onMotionStart(camera_id, post_roll);
            } else if (topic == "camera/event/motion/stop") {
                onMotionStop(camera_id);
            }
        } catch (const json::exception& e) {
            spdlog::error("EventManager: failed to parse MQTT payload on {}: {}", topic, e.what());
        }
    }, 1);

    spdlog::info("EventManager: started, listening for motion events");
}

void EventManager::stop() {
    running_ = false;

    std::lock_guard lock(events_mutex_);
    for (auto& [cam_id, event] : active_events_) {
        event->stop_requested = true;
    }
    for (auto& [cam_id, event] : active_events_) {
        if (event->thread.joinable()) {
            event->thread.join();
        }
    }
    active_events_.clear();
    spdlog::info("EventManager: stopped");
}

size_t EventManager::activeEventCount() const {
    std::lock_guard lock(events_mutex_);
    return active_events_.size();
}

void EventManager::onMotionStart(const std::string& camera_id, int post_roll_seconds) {
    std::lock_guard lock(events_mutex_);

    // Cancel previous event for this camera if still running
    auto it = active_events_.find(camera_id);
    if (it != active_events_.end()) {
        spdlog::info("EventManager: cancelling previous event for {}", camera_id);
        it->second->stop_requested = true;
        if (it->second->thread.joinable()) {
            it->second->thread.detach();  // Let it finish in background
        }
        active_events_.erase(it);
    }

    // Spawn new event thread
    auto event = std::make_unique<ActiveEvent>();
    auto* event_ptr = event.get();
    event->thread = std::thread([this, camera_id, post_roll_seconds, event_ptr]() {
        processEvent(camera_id, post_roll_seconds);
        event_ptr->running = false;
    });

    active_events_[camera_id] = std::move(event);
    spdlog::info("EventManager: motion start for {}", camera_id);
}

void EventManager::onMotionStop(const std::string& camera_id) {
    std::lock_guard lock(events_mutex_);

    auto it = active_events_.find(camera_id);
    if (it != active_events_.end()) {
        it->second->stop_requested = true;
        spdlog::info("EventManager: motion stop for {}", camera_id);
    }
}

void EventManager::processEvent(const std::string& camera_id, int post_roll_seconds) {
    auto prefix = mqtt_ ? mqtt_->topicPrefix() : "yolo_detection";
    auto event_id = generateEventId();

    spdlog::info("EventManager: processing event {} for {}", event_id, camera_id);

    // Find this event's stop flag
    ActiveEvent* my_event = nullptr;
    {
        std::lock_guard lock(events_mutex_);
        auto it = active_events_.find(camera_id);
        if (it != active_events_.end()) {
            my_event = it->second.get();
        }
    }
    if (!my_event) return;

    // 1. Publish detection started
    if (mqtt_) {
        json status_msg = {
            {"status", "started"},
            {"timestamp", yolo::time_utils::now_iso8601()},
            {"camera_id", camera_id}
        };
        mqtt_->publish(prefix + "/" + camera_id + "/detection", status_msg.dump());
    }

    // 2. Get camera buffer and detection engine
    auto buffer = buffer_service_->getCameraBuffer(camera_id);
    auto engine = buffer_service_->getDetectionEngine();
    if (!buffer) {
        spdlog::error("EventManager: no buffer for camera {}", camera_id);
        return;
    }

    // 3. Get preroll frames — deep-copy pixels and release pool references immediately
    std::vector<std::shared_ptr<FrameData>> preroll_frames;
    {
        auto pool_frames = buffer->getBuffer();
        preroll_frames.reserve(pool_frames.size());
        for (const auto& pf : pool_frames) {
            if (!pf) continue;
            auto copy = std::make_shared<FrameData>();
            copy->pixels = pf->pixels;  // deep copy
            copy->width = pf->width;
            copy->height = pf->height;
            copy->stride = pf->stride;
            copy->timestamp = pf->timestamp;
            copy->frame_number = pf->frame_number;
            preroll_frames.push_back(std::move(copy));
        }
        // pool_frames destroyed here — shared_ptrs returned to pool
    }
    spdlog::info("EventManager: {} preroll frames for {}", preroll_frames.size(), camera_id);

    // 4. Determine frame dimensions from preroll or latest frame
    int width = 0, height = 0;
    for (const auto& f : preroll_frames) {
        if (f && f->width > 0) { width = f->width; height = f->height; break; }
    }
    if (width == 0) {
        auto latest = buffer->getLatestFrame();
        if (latest) { width = latest->width; height = latest->height; }
    }
    if (width == 0) {
        spdlog::error("EventManager: no frames available for {}", camera_id);
        return;
    }

    // 5. Start recorder with preroll
    EventRecorder recorder;
    int fps = config_.buffer.fps > 0 ? config_.buffer.fps : 10;
    std::string events_dir = config_.timeline.events_dir;
    if (!recorder.start(camera_id, preroll_frames, width, height, fps, events_dir)) {
        spdlog::error("EventManager: failed to start recorder for {}", camera_id);
        return;
    }

    // Preroll written to recorder — release copies to free memory
    preroll_frames.clear();

    // 6. Run detection on preroll is already done via recorder.
    //    Now detect during live phase only.
    std::vector<Detection> all_detections;
    std::unique_ptr<FrameData> best_frame;  // owned copy, not pool ref
    float best_confidence = 0.0f;
    std::vector<Detection> best_detections;

    // Get camera-specific config
    float conf_threshold = static_cast<float>(config_.detection.confidence_threshold);
    float iou_threshold = static_cast<float>(config_.detection.iou_threshold);
    std::vector<std::string> filter_classes = config_.detection.classes;

    auto cam_it = config_.cameras.find(camera_id);
    if (cam_it != config_.cameras.end()) {
        if (cam_it->second.confidence_threshold > 0) {
            conf_threshold = static_cast<float>(cam_it->second.confidence_threshold);
        }
        if (!cam_it->second.classes.empty()) {
            filter_classes = cam_it->second.classes;
        }
    }

    // Helper: deep-copy a frame (avoids pinning pool frames)
    auto copyFrame = [](const FrameData& src) {
        auto copy = std::make_unique<FrameData>();
        copy->pixels = src.pixels;
        copy->width = src.width;
        copy->height = src.height;
        copy->stride = src.stride;
        copy->timestamp = src.timestamp;
        copy->frame_number = src.frame_number;
        return copy;
    };

    // 7. Live phase: pull frames, write to recorder, sample detections
    auto start_time = Clock::now();
    int frames_since_detection = 0;
    constexpr int DETECTION_SAMPLE_INTERVAL = 3;  // detect every 3rd frame

    while (!my_event->stop_requested && !recorder.isMaxDurationReached()) {
        auto frame = buffer->getLatestFrame();
        if (!frame || frame->width != width) {
            std::this_thread::sleep_for(std::chrono::milliseconds(30));
            continue;
        }

        recorder.writeFrame(*frame);
        frames_since_detection++;

        // Sample detection
        if (engine && engine->isLoaded() && frames_since_detection >= DETECTION_SAMPLE_INTERVAL) {
            frames_since_detection = 0;
            auto dets = engine->detect(*frame, conf_threshold, iou_threshold, filter_classes);
            for (const auto& d : dets) {
                all_detections.push_back(d);
                if (d.confidence > best_confidence) {
                    best_confidence = d.confidence;
                    best_frame = copyFrame(*frame);
                    best_detections = dets;
                }
            }
        }

        // Release pool ref before sleeping
        frame.reset();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
    }

    // 8. Post-roll: continue recording for post_roll_seconds
    recorder.requestStop(post_roll_seconds);
    while (!recorder.isPostRollComplete() && !recorder.isMaxDurationReached()) {
        auto frame = buffer->getLatestFrame();
        if (frame && frame->width == width) {
            recorder.writeFrame(*frame);

            // Continue detection sampling during post-roll
            frames_since_detection++;
            if (engine && engine->isLoaded() && frames_since_detection >= DETECTION_SAMPLE_INTERVAL) {
                frames_since_detection = 0;
                auto dets = engine->detect(*frame, conf_threshold, iou_threshold, filter_classes);
                for (const auto& d : dets) {
                    all_detections.push_back(d);
                    if (d.confidence > best_confidence) {
                        best_confidence = d.confidence;
                        best_frame = copyFrame(*frame);
                        best_detections = dets;
                    }
                }
            }
        }
        frame.reset();  // release pool ref before sleeping
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / fps));
    }

    // 9. Finalize recording
    recorder.finalize();

    // 10. Save snapshot (best detection frame)
    std::string snapshot_path;
    std::string snapshots_dir = config_.timeline.snapshots_dir;
    if (best_frame && !best_detections.empty()) {
        snapshot_path = SnapshotWriter::save(*best_frame, best_detections,
                                             camera_id, snapshots_dir);
    }

    // 11. Compute duration
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - start_time);
    double duration_seconds = elapsed.count() / 1000.0;

    // 12. Deduplicate detections for MQTT payload (one per class, highest confidence)
    std::unordered_map<std::string, Detection> unique_dets;
    for (const auto& d : all_detections) {
        auto it2 = unique_dets.find(d.class_name);
        if (it2 == unique_dets.end() || d.confidence > it2->second.confidence) {
            unique_dets[d.class_name] = d;
        }
    }

    // Build class counts
    std::unordered_map<std::string, int> class_counts;
    for (const auto& d : all_detections) {
        class_counts[d.class_name]++;
    }

    std::vector<std::string> unique_classes;
    for (const auto& [cls, _] : class_counts) {
        unique_classes.push_back(cls);
    }

    // Build detection message
    std::string detection_message;
    if (unique_classes.empty()) {
        detection_message = "No objects detected";
    } else {
        std::vector<std::string> items;
        for (size_t i = 0; i < std::min(unique_classes.size(), size_t(5)); ++i) {
            const auto& cls = unique_classes[i];
            std::string article = (cls[0] == 'a' || cls[0] == 'e' || cls[0] == 'i'
                                   || cls[0] == 'o' || cls[0] == 'u') ? "an" : "a";
            items.push_back(article + " " + cls);
        }
        detection_message = "Detected ";
        for (size_t i = 0; i < items.size(); ++i) {
            if (i > 0 && i == items.size() - 1) detection_message += " and ";
            else if (i > 0) detection_message += ", ";
            detection_message += items[i];
        }
    }

    // Build deduplicated detection array for MQTT
    json dets_json = json::array();
    for (const auto& [cls, d] : unique_dets) {
        dets_json.push_back({
            {"class", d.class_name},
            {"class_id", d.class_id},
            {"confidence", std::round(d.confidence * 1000) / 1000},
            {"bbox", {{"x1", static_cast<int>(d.x1)},
                      {"y1", static_cast<int>(d.y1)},
                      {"x2", static_cast<int>(d.x2)},
                      {"y2", static_cast<int>(d.y2)}}},
        });
    }

    // Snapshot filename (just the filename for URLs)
    std::string snapshot_filename;
    if (!snapshot_path.empty()) {
        snapshot_filename = std::filesystem::path(snapshot_path).filename().string();
    }

    // 13. Publish result to MQTT
    if (mqtt_) {
        json result_msg = {
            {"camera_id", camera_id},
            {"timestamp", yolo::time_utils::now_iso8601()},
            {"detections", dets_json},
            {"detection_count", static_cast<int>(all_detections.size())},
            {"unique_classes", unique_classes},
            {"class_counts", class_counts},
            {"detected_objects", detection_message},
            {"detection_message", detection_message},
            {"frames_processed", recorder.framesWritten()},
            {"processing_time_seconds", std::round(duration_seconds * 100) / 100},
            {"snapshot_url", snapshot_filename.empty() ? json(nullptr) : json("/snapshots/" + snapshot_filename)},
            {"recording_url", recorder.fileName().empty() ? json(nullptr) : json("/events/" + recorder.fileName())},
            {"recording_filename", recorder.fileName()},
        };
        mqtt_->publish(prefix + "/" + camera_id + "/result", result_msg.dump());

        // Binary sensor for HA
        mqtt_->publish(prefix + "/" + camera_id + "/detected",
                       all_detections.empty() ? "OFF" : "ON");

        // Detection completed status
        json complete_msg = {
            {"status", "completed"},
            {"timestamp", yolo::time_utils::now_iso8601()},
            {"camera_id", camera_id}
        };
        mqtt_->publish(prefix + "/" + camera_id + "/detection", complete_msg.dump());
    }

    // 14. Reset binary sensor after a short delay
    if (mqtt_ && !all_detections.empty()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        mqtt_->publish(prefix + "/" + camera_id + "/detected", "OFF");
    }

    // 15. Log to database
    if (db_) {
        yolo::EventLogger::create_event(*db_, event_id, camera_id,
                                        recorder.fileName(), snapshot_filename);

        // Log deduplicated detections
        std::vector<yolo::EventLogger::DetectionRecord> det_records;
        for (const auto& [cls, d] : unique_dets) {
            det_records.push_back({d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2});
        }
        yolo::EventLogger::log_detections(*db_, event_id, det_records);

        yolo::EventLogger::complete_event(*db_, event_id, duration_seconds,
                                          recorder.framesWritten(),
                                          static_cast<int>(all_detections.size()));
    }

    // 16. LLaVA vision context (optional, runs after result already published)
    if (config_.llava.enabled && !snapshot_path.empty() && !best_detections.empty()) {
        float best_conf = best_detections.front().confidence;
        auto cam_conf_it = config_.cameras.find(camera_id);
        double conf_gate = (cam_conf_it != config_.cameras.end())
            ? cam_conf_it->second.immediate_notification_confidence : 0.70;

        if (best_conf >= conf_gate) {
            std::string primary_class = VisionClient::selectPrimaryClass(unique_classes);

            VisionClient vision(config_.llava);
            auto vr = vision.analyze(snapshot_path, camera_id, primary_class);

            if (vr.is_valid && mqtt_) {
                json context_msg = {
                    {"camera_id", camera_id},
                    {"timestamp", yolo::time_utils::now_iso8601()},
                    {"context", vr.context},
                    {"recording_url", recorder.fileName().empty() ? json(nullptr)
                        : json("/events/" + recorder.fileName())},
                    {"recording_filename", recorder.fileName()},
                    {"snapshot_url", snapshot_filename.empty() ? json(nullptr)
                        : json("/snapshots/" + snapshot_filename)},
                    {"source", "llava"}
                };
                mqtt_->publish(prefix + "/" + camera_id + "/context",
                               context_msg.dump());
                spdlog::info("EventManager: published LLaVA context for {}: {}",
                             camera_id, vr.context);
            }

            if (db_) {
                yolo::EventLogger::log_ai_context(*db_, event_id, camera_id, {
                    .context_text = vr.context,
                    .detected_classes = unique_classes,
                    .source_model = config_.llava.model,
                    .prompt_used = vision.lastPrompt(),
                    .response_time_seconds = vr.response_time_seconds,
                    .is_valid = vr.is_valid,
                });
            }
        }
    }

    // 17. Cleanup: remove from active events
    {
        std::lock_guard lock(events_mutex_);
        auto it = active_events_.find(camera_id);
        if (it != active_events_.end() && !it->second->running) {
            it->second->thread.detach();
            active_events_.erase(it);
        }
    }

    spdlog::info("EventManager: event {} completed for {} ({:.1f}s, {} frames, {} detections)",
                 event_id, camera_id, duration_seconds,
                 recorder.framesWritten(), all_detections.size());
}

std::string EventManager::generateEventId() {
    // Simple UUID-like ID: timestamp + random suffix
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    static std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist;
    uint32_t r = dist(gen);

    char buf[32];
    snprintf(buf, sizeof(buf), "%lx-%08x",
             static_cast<unsigned long>(ms), r);
    return buf;
}

}  // namespace hms
