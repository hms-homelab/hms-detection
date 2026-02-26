#pragma once

#include <drogon/HttpController.h>
#include <memory>

namespace hms {

class BufferService;

class DetectionController : public drogon::HttpController<DetectionController> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(DetectionController::detect,
                  "/api/cameras/{camera_id}/detect", drogon::Get);
    ADD_METHOD_TO(DetectionController::annotatedSnapshot,
                  "/api/cameras/{camera_id}/snapshot", drogon::Get);
    METHOD_LIST_END

    void detect(const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                const std::string& camera_id);

    void annotatedSnapshot(const drogon::HttpRequestPtr& req,
                           std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                           const std::string& camera_id);

    static void setBufferService(std::shared_ptr<BufferService> svc);

private:
    static inline std::shared_ptr<BufferService> buffer_service_;
};

}  // namespace hms
