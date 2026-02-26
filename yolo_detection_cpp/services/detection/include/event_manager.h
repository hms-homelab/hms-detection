#pragma once

#include "buffer_service.h"
#include "event_recorder.h"
#include "snapshot_writer.h"
#include "mqtt_client.h"
#include "db_pool.h"
#include "config_manager.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace hms {

/// Full event orchestration: MQTT trigger → detect → record → snapshot → publish.
/// Subscribes to MQTT motion events, manages one event thread per camera.
class EventManager {
public:
    EventManager(std::shared_ptr<BufferService> buffer_service,
                 std::shared_ptr<yolo::MqttClient> mqtt,
                 std::shared_ptr<yolo::DbPool> db,
                 const yolo::AppConfig& config);
    ~EventManager();

    EventManager(const EventManager&) = delete;
    EventManager& operator=(const EventManager&) = delete;

    /// Subscribe to MQTT motion topics
    void start();

    /// Stop all active events and cleanup
    void stop();

    /// Number of currently active events
    size_t activeEventCount() const;

private:
    /// Called when motion start MQTT message arrives
    void onMotionStart(const std::string& camera_id, int post_roll_seconds);

    /// Called when motion stop MQTT message arrives
    void onMotionStop(const std::string& camera_id);

    /// Full event processing thread (one per camera, per event)
    void processEvent(const std::string& camera_id, int post_roll_seconds);

    /// Generate UUID-like event ID
    static std::string generateEventId();

    struct ActiveEvent {
        std::thread thread;
        std::atomic<bool> stop_requested{false};
        std::atomic<bool> running{true};
    };

    std::shared_ptr<BufferService> buffer_service_;
    std::shared_ptr<yolo::MqttClient> mqtt_;
    std::shared_ptr<yolo::DbPool> db_;
    yolo::AppConfig config_;

    mutable std::mutex events_mutex_;
    std::unordered_map<std::string, std::unique_ptr<ActiveEvent>> active_events_;

    std::atomic<bool> running_{false};
};

}  // namespace hms
