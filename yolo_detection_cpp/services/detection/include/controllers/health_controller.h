#pragma once

#include <drogon/HttpController.h>
#include <memory>

namespace yolo { class MqttClient; }

namespace hms {

class BufferService;

class HealthController : public drogon::HttpController<HealthController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(HealthController::getHealth, "/health", drogon::Get);
    METHOD_LIST_END

    void getHealth(const drogon::HttpRequestPtr& req,
                   std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    static void setBufferService(std::shared_ptr<BufferService> svc);
    static void setMqttClient(std::shared_ptr<yolo::MqttClient> mqtt);

private:
    static inline std::shared_ptr<BufferService> buffer_service_;
    static inline std::shared_ptr<yolo::MqttClient> mqtt_client_;
};

}  // namespace hms
