#include <drogon/drogon.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <filesystem>
#include <csignal>
#include <atomic>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/log.h>
}

#include "config_manager.h"
#include "buffer_service.h"
#include "mqtt_client.h"
#include "db_pool.h"
#include "event_manager.h"
#include "controllers/health_controller.h"
#include "controllers/detection_controller.h"

namespace fs = std::filesystem;

namespace {

std::atomic<bool> g_shutdown{false};
std::shared_ptr<hms::BufferService> g_buffer_service;
std::shared_ptr<hms::EventManager> g_event_manager;
std::shared_ptr<yolo::MqttClient> g_mqtt;

void signal_handler(int sig) {
    spdlog::info("Received signal {}, shutting down...", sig);
    g_shutdown = true;
    if (g_event_manager) {
        g_event_manager->stop();
    }
    if (g_buffer_service) {
        g_buffer_service->stopDetection();
        g_buffer_service->stopAll();
    }
    drogon::app().quit();
}

void setup_logging(const yolo::LoggingConfig& log_config) {
    std::vector<spdlog::sink_ptr> sinks;
    sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

    if (!log_config.file.empty()) {
        auto dir = fs::path(log_config.file).parent_path();
        if (!dir.empty()) fs::create_directories(dir);
        sinks.push_back(std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_config.file, log_config.max_bytes, log_config.backup_count));
    }

    auto logger = std::make_shared<spdlog::logger>("hms-detection", sinks.begin(), sinks.end());

    spdlog::level::level_enum level = spdlog::level::info;
    if (log_config.level == "DEBUG" || log_config.level == "debug") level = spdlog::level::debug;
    else if (log_config.level == "WARNING" || log_config.level == "warning") level = spdlog::level::warn;
    else if (log_config.level == "ERROR" || log_config.level == "error") level = spdlog::level::err;

    logger->set_level(level);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] %v");
    spdlog::set_default_logger(logger);
    spdlog::flush_every(std::chrono::seconds(3));
}

std::string find_config_path(int argc, char* argv[]) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--config") return argv[i + 1];
    }
    if (fs::exists("config.yaml")) return "config.yaml";
    if (fs::exists("/app/config/config.yaml")) return "/app/config/config.yaml";
    if (fs::exists("/opt/yolo_detection/config.yaml")) return "/opt/yolo_detection/config.yaml";
    return "config.yaml";
}

/// Forward FFmpeg log messages to spdlog
void ffmpeg_log_callback(void* /*ptr*/, int level, const char* fmt, va_list vl) {
    if (level > AV_LOG_WARNING) return;  // Only warnings and errors

    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, vl);

    // Strip trailing newline
    auto len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';

    if (level <= AV_LOG_ERROR) {
        spdlog::error("[ffmpeg] {}", buf);
    } else {
        spdlog::warn("[ffmpeg] {}", buf);
    }
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    try {
        auto config_path = find_config_path(argc, argv);
        auto config = yolo::ConfigManager::load(config_path);

        setup_logging(config.logging);
        spdlog::info("Starting hms-detection service v3.0.0");
        spdlog::info("Config: {}", config_path);

        // Initialize FFmpeg
        avformat_network_init();
        av_log_set_callback(ffmpeg_log_callback);

        // Create buffer service
        g_buffer_service = std::make_shared<hms::BufferService>(config);

        // Wire controller dependencies
        hms::HealthController::setBufferService(g_buffer_service);
        hms::DetectionController::setBufferService(g_buffer_service);

        // Signal handlers
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Start capturing
        g_buffer_service->startAll();

        // Load detection model (but don't start continuous workers —
        // detection only runs on-demand during motion events)
        g_buffer_service->loadDetectionModel();

        // --- MQTT (initialized BEFORE app.run(), independent subsystem) ---
        g_mqtt = std::make_shared<yolo::MqttClient>(config.mqtt);
        try {
            if (g_mqtt->connect()) {
                // Publish online status (retained)
                g_mqtt->publish(config.mqtt.topic_prefix + "/status", "online", 1, true);
            }
        } catch (const std::exception& e) {
            spdlog::warn("MQTT unavailable: {} (HTTP will continue serving)", e.what());
        }

        // Inject MQTT into health controller for status reporting
        hms::HealthController::setMqttClient(g_mqtt);

        // --- Database pool (for event logging) ---
        std::shared_ptr<yolo::DbPool> db;
        try {
            yolo::DbPool::Config db_cfg;
            db_cfg.host = config.database.host;
            db_cfg.port = config.database.port;
            db_cfg.user = config.database.user;
            db_cfg.password = config.database.password;
            db_cfg.database = config.database.database;
            db_cfg.pool_size = config.database.pool_size;
            db = std::make_shared<yolo::DbPool>(db_cfg);
        } catch (const std::exception& e) {
            spdlog::warn("Database unavailable: {} (event logging disabled)", e.what());
        }

        // --- EventManager (MQTT trigger → detect → record → publish) ---
        g_event_manager = std::make_shared<hms::EventManager>(
            g_buffer_service, g_mqtt, db, config);
        g_event_manager->start();

        // Configure Drogon
        auto& app = drogon::app();
        app.setLogLevel(trantor::Logger::kWarn);
        app.addListener(config.api.host, config.api.port);
        app.setThreadNum(2);
        app.setMaxConnectionNum(100);

        // Global CORS headers
        app.registerPostHandlingAdvice(
            [](const drogon::HttpRequestPtr& req, const drogon::HttpResponsePtr& resp) {
                auto origin = std::string(req->getHeader("Origin"));
                std::string allow_origin = origin.empty() ? "*" : origin;
                resp->addHeader("Access-Control-Allow-Origin", allow_origin);
                resp->addHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
                resp->addHeader("Access-Control-Allow-Headers",
                                "Content-Type, Authorization, Accept");
            }
        );

        spdlog::info("Listening on {}:{}", config.api.host, config.api.port);
        spdlog::info("Cameras: {}", g_buffer_service->cameraIds().size());

        app.run();  // Blocks until quit

        // Cleanup
        spdlog::info("Shutting down...");
        if (g_event_manager) g_event_manager->stop();
        g_buffer_service->stopDetection();
        g_buffer_service->stopAll();

        // MQTT offline + disconnect
        if (g_mqtt) {
            g_mqtt->publish(config.mqtt.topic_prefix + "/status", "offline", 1, true);
            g_mqtt->disconnect();
        }

        g_event_manager.reset();
        g_mqtt.reset();
        g_buffer_service.reset();
        avformat_network_deinit();
        spdlog::info("Shutdown complete");

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
