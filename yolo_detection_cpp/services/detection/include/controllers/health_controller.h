#pragma once

#include <drogon/HttpController.h>
#include <memory>

namespace hms {

class BufferService;

class HealthController : public drogon::HttpController<HealthController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(HealthController::getHealth, "/health", drogon::Get);
    ADD_METHOD_TO(HealthController::getSnapshot, "/api/cameras/{camera_id}/snapshot", drogon::Get);
    METHOD_LIST_END

    void getHealth(const drogon::HttpRequestPtr& req,
                   std::function<void(const drogon::HttpResponsePtr&)>&& callback);

    void getSnapshot(const drogon::HttpRequestPtr& req,
                     std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                     const std::string& camera_id);

    static void setBufferService(std::shared_ptr<BufferService> svc);

private:
    static inline std::shared_ptr<BufferService> buffer_service_;
};

}  // namespace hms
