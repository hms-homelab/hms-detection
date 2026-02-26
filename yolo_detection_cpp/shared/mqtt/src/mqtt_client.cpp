#include "mqtt_client.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <sstream>

namespace yolo {

MqttClient::MqttClient(const MqttConfig& config)
    : config_(config)
{
    std::string broker_uri = "tcp://" + config_.broker + ":" + std::to_string(config_.port);
    std::string client_id = "hms_detection_" + std::to_string(::getpid());

    client_ = std::make_unique<mqtt::async_client>(broker_uri, client_id);
    client_->set_callback(*this);

    // LWT: publish "offline" on unexpected disconnect
    std::string will_topic = config_.topic_prefix + "/status";
    mqtt::will_options will(will_topic, std::string("offline"), 1, true);

    conn_opts_ = mqtt::connect_options_builder()
        .automatic_reconnect(std::chrono::seconds(1), std::chrono::seconds(64))
        .clean_session(true)
        .keep_alive_interval(std::chrono::seconds(60))
        .connect_timeout(std::chrono::seconds(10))
        .will(std::move(will))
        .finalize();

    if (!config_.username.empty()) {
        conn_opts_.set_user_name(config_.username);
        conn_opts_.set_password(config_.password);
    }
}

MqttClient::~MqttClient() {
    disconnect();
}

bool MqttClient::connect() {
    try {
        spdlog::info("MQTT: connecting to {}:{}...", config_.broker, config_.port);
        auto tok = client_->connect(conn_opts_);
        tok->wait_for(std::chrono::seconds(10));

        if (client_->is_connected()) {
            spdlog::info("MQTT: connected to {}:{}", config_.broker, config_.port);
            return true;
        }

        spdlog::warn("MQTT: connection pending (will auto-reconnect in background)");
        return false;

    } catch (const mqtt::exception& e) {
        spdlog::warn("MQTT: connection failed: {} (will auto-reconnect)", e.what());
        return false;
    }
}

void MqttClient::disconnect() {
    if (!client_) return;

    try {
        if (client_->is_connected()) {
            // Publish offline status before disconnecting
            publish(config_.topic_prefix + "/status", "offline", 1, true);
            auto tok = client_->disconnect();
            tok->wait_for(std::chrono::seconds(2));
        }
    } catch (const mqtt::exception& e) {
        spdlog::debug("MQTT: disconnect error: {}", e.what());
    }
}

void MqttClient::publish(const std::string& topic, const std::string& payload,
                          int qos, bool retain) {
    if (!client_ || !client_->is_connected()) return;

    try {
        // Fire-and-forget: no token->wait() — safe from Paho callbacks
        client_->publish(topic, payload.data(), payload.size(), qos, retain);
    } catch (const mqtt::exception& e) {
        spdlog::debug("MQTT: publish failed on {}: {}", topic, e.what());
    }
}

void MqttClient::subscribe(const std::vector<std::string>& topics,
                            MessageCallback callback, int qos) {
    std::lock_guard lock(mutex_);

    // Register callback for each topic pattern
    for (const auto& topic : topics) {
        subscriptions_[topic] = callback;
    }

    // Save for re-subscribe on reconnect
    pending_subs_.push_back({topics, qos});

    if (!client_ || !client_->is_connected()) return;

    try {
        // Batch subscribe: single call with all topics
        auto topic_coll = mqtt::string_collection::create(topics);
        std::vector<int> qos_levels(topics.size(), qos);
        client_->subscribe(topic_coll, qos_levels)->wait_for(std::chrono::seconds(5));

        for (const auto& t : topics) {
            spdlog::info("MQTT: subscribed to {}", t);
        }
    } catch (const mqtt::exception& e) {
        spdlog::warn("MQTT: subscribe failed: {}", e.what());
    }
}

bool MqttClient::isConnected() const {
    return client_ && client_->is_connected();
}

// --- Paho callbacks (called on Paho's internal thread) ---

void MqttClient::connected(const std::string& cause) {
    spdlog::info("MQTT: connected ({})", cause.empty() ? "initial" : cause);

    // Re-subscribe to all topics after reconnect
    std::lock_guard lock(mutex_);
    for (const auto& sub : pending_subs_) {
        try {
            auto topic_coll = mqtt::string_collection::create(sub.topics);
            std::vector<int> qos_levels(sub.topics.size(), sub.qos);
            // Fire-and-forget from callback — no wait
            client_->subscribe(topic_coll, qos_levels);
        } catch (const mqtt::exception& e) {
            spdlog::warn("MQTT: re-subscribe failed: {}", e.what());
        }
    }
}

void MqttClient::connection_lost(const std::string& cause) {
    spdlog::warn("MQTT: connection lost: {} (auto-reconnect enabled)",
                 cause.empty() ? "unknown" : cause);
}

void MqttClient::message_arrived(mqtt::const_message_ptr msg) {
    const auto& topic = msg->get_topic();
    auto payload = msg->get_payload_str();

    std::lock_guard lock(mutex_);
    for (const auto& [pattern, callback] : subscriptions_) {
        if (topicMatches(pattern, topic)) {
            try {
                callback(topic, payload);
            } catch (const std::exception& e) {
                spdlog::error("MQTT: callback error for {}: {}", topic, e.what());
            }
            return;  // first match wins
        }
    }
}

bool MqttClient::topicMatches(const std::string& pattern, const std::string& topic) {
    // Split both into segments
    auto split = [](const std::string& s) {
        std::vector<std::string> parts;
        std::istringstream iss(s);
        std::string part;
        while (std::getline(iss, part, '/')) {
            parts.push_back(part);
        }
        return parts;
    };

    auto pat_parts = split(pattern);
    auto top_parts = split(topic);

    size_t pi = 0;
    for (size_t ti = 0; ti < top_parts.size(); ++ti) {
        if (pi >= pat_parts.size()) return false;

        if (pat_parts[pi] == "#") {
            return true;  // # matches everything remaining
        }
        if (pat_parts[pi] == "+") {
            ++pi;
            continue;  // + matches exactly one level
        }
        if (pat_parts[pi] != top_parts[ti]) {
            return false;
        }
        ++pi;
    }

    return pi == pat_parts.size();
}

}  // namespace yolo
