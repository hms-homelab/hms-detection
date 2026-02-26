#include "controllers/health_controller.h"
#include "buffer_service.h"
#include "time_utils.h"

#include <drogon/HttpResponse.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace hms {

void HealthController::setBufferService(std::shared_ptr<BufferService> svc) {
    buffer_service_ = std::move(svc);
}

void HealthController::getHealth(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback)
{
    using json = nlohmann::json;

    auto stats = buffer_service_->getAllStats();
    bool healthy = buffer_service_->isHealthy();

    json cameras_json = json::object();
    for (const auto& s : stats) {
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - s.last_frame_time).count();

        cameras_json[s.camera_id] = {
            {"name", s.camera_name},
            {"buffer_size", s.buffer_size},
            {"max_frames", s.max_frames},
            {"frames_captured", s.frames_captured},
            {"reconnect_count", s.reconnect_count},
            {"consecutive_failures", s.consecutive_failures},
            {"is_connected", s.is_connected},
            {"is_healthy", s.is_healthy},
            {"frame_width", s.frame_width},
            {"frame_height", s.frame_height},
            {"last_frame_ms_ago", s.frames_captured > 0 ? elapsed_ms : -1},
        };
    }

    json result = {
        {"service", "hms-detection"},
        {"status", healthy ? "healthy" : "degraded"},
        {"timestamp", yolo::time_utils::now_iso8601()},
        {"cameras", cameras_json},
    };

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(healthy ? drogon::k200OK : drogon::k503ServiceUnavailable);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(result.dump());
    callback(resp);
}

void HealthController::getSnapshot(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& camera_id)
{
    auto frame = buffer_service_->getLatestFrame(camera_id);
    if (!frame) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k404NotFound);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"No frame available for camera: )" + camera_id + R"("})");
        callback(resp);
        return;
    }

    // Encode BGR24 → JPEG using FFmpeg MJPEG encoder
    const AVCodec* mjpeg_codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!mjpeg_codec) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"MJPEG encoder not available"})");
        callback(resp);
        return;
    }

    AVCodecContext* enc_ctx = avcodec_alloc_context3(mjpeg_codec);
    enc_ctx->width = frame->width;
    enc_ctx->height = frame->height;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
    enc_ctx->time_base = {1, 25};
    enc_ctx->flags |= AV_CODEC_FLAG_QSCALE;
    // Quality: lower = better. 2-5 is good range. ~85% quality ≈ qmin=2, qmax=5
    enc_ctx->qmin = 2;
    enc_ctx->qmax = 5;

    if (avcodec_open2(enc_ctx, mjpeg_codec, nullptr) < 0) {
        avcodec_free_context(&enc_ctx);
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"Failed to open MJPEG encoder"})");
        callback(resp);
        return;
    }

    // Convert BGR24 → YUV420P for MJPEG
    SwsContext* sws = sws_getContext(
        frame->width, frame->height, AV_PIX_FMT_BGR24,
        frame->width, frame->height, AV_PIX_FMT_YUVJ420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVFrame* yuv_frame = av_frame_alloc();
    yuv_frame->format = AV_PIX_FMT_YUVJ420P;
    yuv_frame->width = frame->width;
    yuv_frame->height = frame->height;
    av_frame_get_buffer(yuv_frame, 0);

    const uint8_t* src_data[1] = {frame->pixels.data()};
    int src_linesize[1] = {frame->stride};
    sws_scale(sws, src_data, src_linesize, 0, frame->height,
              yuv_frame->data, yuv_frame->linesize);

    // Encode
    AVPacket* pkt = av_packet_alloc();
    avcodec_send_frame(enc_ctx, yuv_frame);
    int ret = avcodec_receive_packet(enc_ctx, pkt);

    if (ret == 0) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeCode(drogon::CT_NONE);
        resp->addHeader("Content-Type", "image/jpeg");
        resp->setBody(std::string(reinterpret_cast<const char*>(pkt->data), pkt->size));
        callback(resp);
    } else {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"JPEG encoding failed"})");
        callback(resp);
    }

    // Cleanup
    av_packet_free(&pkt);
    av_frame_free(&yuv_frame);
    sws_freeContext(sws);
    avcodec_free_context(&enc_ctx);
}

}  // namespace hms
