#pragma once

#include <drogon/HttpController.h>
#include <memory>

namespace hms { class MqttClient; }

namespace hms {

class BufferService;
class EventManager;

class HealthController : public drogon::HttpController<HealthController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(HealthController::getHealth, "/health", drogon::Get);
    METHOD_LIST_END

    void getHealth(const drogon::HttpRequestPtr& req,
                   std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    static void setBufferService(std::shared_ptr<BufferService> svc);
    static void setMqttClient(std::shared_ptr<hms::MqttClient> mqtt);
    static void setEventManager(std::shared_ptr<EventManager> em);

private:
    static inline std::shared_ptr<BufferService> buffer_service_;
    static inline std::shared_ptr<hms::MqttClient> mqtt_client_;
    static inline std::shared_ptr<EventManager> event_manager_;
};

}  // namespace hms
