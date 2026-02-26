#pragma once

#include "config_manager.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mqtt/async_client.h>

namespace yolo {

/// Thread-safe Paho MQTT async_client wrapper with auto-reconnect.
/// Pattern: fire-and-forget publish (no token->wait()), batch subscribe,
/// callback-based message dispatch with wildcard matching.
class MqttClient : public mqtt::callback {
public:
    using MessageCallback = std::function<void(const std::string& topic,
                                               const std::string& payload)>;

    explicit MqttClient(const MqttConfig& config);
    ~MqttClient() override;

    MqttClient(const MqttClient&) = delete;
    MqttClient& operator=(const MqttClient&) = delete;

    /// Connect to broker (blocking, with timeout). Returns true on success.
    /// Graceful degradation: failure does not throw, just logs and returns false.
    bool connect();

    /// Disconnect gracefully
    void disconnect();

    /// Fire-and-forget publish (safe to call from any thread, including Paho callbacks)
    void publish(const std::string& topic, const std::string& payload,
                 int qos = 1, bool retain = false);

    /// Subscribe to topics with a callback. Performs a single batch subscribe call.
    void subscribe(const std::vector<std::string>& topics,
                   MessageCallback callback, int qos = 1);

    /// Thread-safe connection status check
    bool isConnected() const;

    /// Topic prefix from config
    const std::string& topicPrefix() const { return config_.topic_prefix; }

private:
    // mqtt::callback overrides (called on Paho's internal thread)
    void connected(const std::string& cause) override;
    void connection_lost(const std::string& cause) override;
    void message_arrived(mqtt::const_message_ptr msg) override;

    /// Check if a topic matches a subscription pattern (supports + and # wildcards)
    static bool topicMatches(const std::string& pattern, const std::string& topic);

    MqttConfig config_;
    std::unique_ptr<mqtt::async_client> client_;
    mqtt::connect_options conn_opts_;

    mutable std::recursive_mutex mutex_;
    std::map<std::string, MessageCallback> subscriptions_;  // pattern → callback

    // For re-subscribing after reconnect
    struct PendingSub {
        std::vector<std::string> topics;
        int qos;
    };
    std::vector<PendingSub> pending_subs_;
};

}  // namespace yolo
