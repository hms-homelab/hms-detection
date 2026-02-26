#include "controllers/detection_controller.h"
#include "buffer_service.h"
#include "detection_engine.h"
#include "time_utils.h"

#include <drogon/HttpResponse.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace hms {

void DetectionController::setBufferService(std::shared_ptr<BufferService> svc) {
    buffer_service_ = std::move(svc);
}

void DetectionController::detect(
    const drogon::HttpRequestPtr& /*req*/,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& camera_id)
{
    using json = nlohmann::json;

    if (!buffer_service_) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"Service not initialized"})");
        callback(resp);
        return;
    }

    // Get latest detection result from the worker
    auto result = buffer_service_->getDetectionResult(camera_id);
    if (!result) {
        // Try on-demand detection from latest frame
        auto frame = buffer_service_->getLatestFrame(camera_id);
        if (!frame) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k404NotFound);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            resp->setBody(R"({"error":"No frame available for camera: )" + camera_id + R"("})");
            callback(resp);
            return;
        }

        auto engine = buffer_service_->getDetectionEngine();
        if (!engine || !engine->isLoaded()) {
            auto resp = drogon::HttpResponse::newHttpResponse();
            resp->setStatusCode(drogon::k503ServiceUnavailable);
            resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
            resp->setBody(R"({"error":"Detection model not loaded"})");
            callback(resp);
            return;
        }

        auto start = std::chrono::steady_clock::now();
        auto detections = engine->detect(*frame);
        auto elapsed = std::chrono::steady_clock::now() - start;
        double inference_ms = std::chrono::duration<double, std::milli>(elapsed).count();

        json dets_json = json::array();
        for (const auto& d : detections) {
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

        json response = {
            {"camera_id", camera_id},
            {"timestamp", yolo::time_utils::now_iso8601()},
            {"frame_number", frame->frame_number},
            {"inference_ms", std::round(inference_ms * 10) / 10},
            {"detections", dets_json},
        };

        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(response.dump());
        callback(resp);
        return;
    }

    // Return cached worker result
    json dets_json = json::array();
    for (const auto& d : result->detections) {
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

    auto stats = buffer_service_->getDetectionStats();
    double inference_ms = 0;
    for (const auto& [id, s] : stats) {
        if (id == camera_id) { inference_ms = s.avg_inference_ms; break; }
    }

    json response = {
        {"camera_id", camera_id},
        {"timestamp", yolo::time_utils::now_iso8601()},
        {"frame_number", result->frame_number},
        {"inference_ms", std::round(inference_ms * 10) / 10},
        {"detections", dets_json},
    };

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k200OK);
    resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
    resp->setBody(response.dump());
    callback(resp);
}

/// Draw bounding box rectangles on BGR24 frame data
static void drawBoundingBoxes(std::vector<uint8_t>& pixels, int width, int height, int stride,
                               const std::vector<Detection>& detections) {
    // Simple color palette (BGR)
    static const uint8_t colors[][3] = {
        {0, 255, 0},     // green
        {0, 0, 255},     // red
        {255, 0, 0},     // blue
        {0, 255, 255},   // yellow
        {255, 0, 255},   // magenta
        {255, 255, 0},   // cyan
    };
    constexpr int num_colors = sizeof(colors) / sizeof(colors[0]);
    constexpr int thickness = 2;

    for (const auto& det : detections) {
        int x1 = std::max(0, static_cast<int>(det.x1));
        int y1 = std::max(0, static_cast<int>(det.y1));
        int x2 = std::min(width - 1, static_cast<int>(det.x2));
        int y2 = std::min(height - 1, static_cast<int>(det.y2));

        const uint8_t* color = colors[det.class_id % num_colors];

        // Draw horizontal lines (top and bottom)
        for (int t = 0; t < thickness; ++t) {
            int top_y = y1 + t;
            int bot_y = y2 - t;
            if (top_y >= 0 && top_y < height) {
                for (int x = x1; x <= x2; ++x) {
                    uint8_t* px = pixels.data() + top_y * stride + x * 3;
                    px[0] = color[0]; px[1] = color[1]; px[2] = color[2];
                }
            }
            if (bot_y >= 0 && bot_y < height && bot_y != top_y) {
                for (int x = x1; x <= x2; ++x) {
                    uint8_t* px = pixels.data() + bot_y * stride + x * 3;
                    px[0] = color[0]; px[1] = color[1]; px[2] = color[2];
                }
            }
        }

        // Draw vertical lines (left and right)
        for (int t = 0; t < thickness; ++t) {
            int left_x = x1 + t;
            int right_x = x2 - t;
            if (left_x >= 0 && left_x < width) {
                for (int y = y1; y <= y2; ++y) {
                    uint8_t* px = pixels.data() + y * stride + left_x * 3;
                    px[0] = color[0]; px[1] = color[1]; px[2] = color[2];
                }
            }
            if (right_x >= 0 && right_x < width && right_x != left_x) {
                for (int y = y1; y <= y2; ++y) {
                    uint8_t* px = pixels.data() + y * stride + right_x * 3;
                    px[0] = color[0]; px[1] = color[1]; px[2] = color[2];
                }
            }
        }
    }
}

/// Encode BGR24 frame to JPEG using FFmpeg MJPEG encoder
static std::string encodeJpeg(const uint8_t* pixels, int width, int height, int stride) {
    const AVCodec* mjpeg_codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
    if (!mjpeg_codec) return {};

    AVCodecContext* enc_ctx = avcodec_alloc_context3(mjpeg_codec);
    enc_ctx->width = width;
    enc_ctx->height = height;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUVJ420P;
    enc_ctx->time_base = {1, 25};
    enc_ctx->flags |= AV_CODEC_FLAG_QSCALE;
    enc_ctx->qmin = 2;
    enc_ctx->qmax = 5;

    if (avcodec_open2(enc_ctx, mjpeg_codec, nullptr) < 0) {
        avcodec_free_context(&enc_ctx);
        return {};
    }

    SwsContext* sws = sws_getContext(
        width, height, AV_PIX_FMT_BGR24,
        width, height, AV_PIX_FMT_YUVJ420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVFrame* yuv_frame = av_frame_alloc();
    yuv_frame->format = AV_PIX_FMT_YUVJ420P;
    yuv_frame->width = width;
    yuv_frame->height = height;
    av_frame_get_buffer(yuv_frame, 0);

    const uint8_t* src_data[1] = {pixels};
    int src_linesize[1] = {stride};
    sws_scale(sws, src_data, src_linesize, 0, height,
              yuv_frame->data, yuv_frame->linesize);

    AVPacket* pkt = av_packet_alloc();
    avcodec_send_frame(enc_ctx, yuv_frame);

    std::string result;
    if (avcodec_receive_packet(enc_ctx, pkt) == 0) {
        result.assign(reinterpret_cast<const char*>(pkt->data), pkt->size);
    }

    av_packet_free(&pkt);
    av_frame_free(&yuv_frame);
    sws_freeContext(sws);
    avcodec_free_context(&enc_ctx);

    return result;
}

void DetectionController::annotatedSnapshot(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback,
    const std::string& camera_id)
{
    if (!buffer_service_) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"Service not initialized"})");
        callback(resp);
        return;
    }

    auto frame = buffer_service_->getLatestFrame(camera_id);
    if (!frame) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k404NotFound);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"No frame available for camera: )" + camera_id + R"("})");
        callback(resp);
        return;
    }

    bool annotate = req->getParameter("annotate") == "true";

    if (annotate) {
        // Get detections and draw bounding boxes on a copy of the frame
        std::vector<Detection> detections;

        auto result = buffer_service_->getDetectionResult(camera_id);
        if (result) {
            detections = result->detections;
        } else {
            auto engine = buffer_service_->getDetectionEngine();
            if (engine && engine->isLoaded()) {
                detections = engine->detect(*frame);
            }
        }

        if (!detections.empty()) {
            // Copy pixels so we don't modify the shared frame
            auto annotated_pixels = frame->pixels;
            drawBoundingBoxes(annotated_pixels, frame->width, frame->height,
                              frame->stride, detections);

            auto jpeg = encodeJpeg(annotated_pixels.data(), frame->width,
                                   frame->height, frame->stride);
            if (!jpeg.empty()) {
                auto resp = drogon::HttpResponse::newHttpResponse();
                resp->setStatusCode(drogon::k200OK);
                resp->setContentTypeCode(drogon::CT_NONE);
                resp->addHeader("Content-Type", "image/jpeg");
                resp->setBody(std::move(jpeg));
                callback(resp);
                return;
            }
        }
    }

    // Plain snapshot (no annotation, or annotation failed)
    auto jpeg = encodeJpeg(frame->pixels.data(), frame->width,
                           frame->height, frame->stride);
    if (jpeg.empty()) {
        auto resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k500InternalServerError);
        resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
        resp->setBody(R"({"error":"JPEG encoding failed"})");
        callback(resp);
        return;
    }

    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setStatusCode(drogon::k200OK);
    resp->setContentTypeCode(drogon::CT_NONE);
    resp->addHeader("Content-Type", "image/jpeg");
    resp->setBody(std::move(jpeg));
    callback(resp);
}

}  // namespace hms
