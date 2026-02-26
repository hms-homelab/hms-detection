#include "snapshot_writer.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <ctime>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace fs = std::filesystem;

namespace hms {

void SnapshotWriter::drawBoundingBoxes(std::vector<uint8_t>& pixels,
                                        int width, int height, int stride,
                                        const std::vector<Detection>& detections) {
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

        for (int t = 0; t < thickness; ++t) {
            int top_y = y1 + t, bot_y = y2 - t;
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

        for (int t = 0; t < thickness; ++t) {
            int left_x = x1 + t, right_x = x2 - t;
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

std::string SnapshotWriter::encodeJpeg(const uint8_t* pixels,
                                        int width, int height, int stride) {
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

std::string SnapshotWriter::save(const FrameData& frame,
                                  const std::vector<Detection>& detections,
                                  const std::string& camera_id,
                                  const std::string& output_dir) {
    fs::create_directories(output_dir);

    // Generate timestamp for filename
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
    localtime_r(&t, &tm_buf);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", &tm_buf);

    std::string file_path = output_dir + "/" + camera_id + "_" + ts + ".jpg";

    // Copy pixels and draw bounding boxes
    auto pixels = frame.pixels;
    if (!detections.empty()) {
        drawBoundingBoxes(pixels, frame.width, frame.height, frame.stride, detections);
    }

    // Encode to JPEG
    auto jpeg = encodeJpeg(pixels.data(), frame.width, frame.height, frame.stride);
    if (jpeg.empty()) {
        spdlog::error("SnapshotWriter: JPEG encoding failed for {}", camera_id);
        return {};
    }

    // Write to disk
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs) {
        spdlog::error("SnapshotWriter: failed to open {}", file_path);
        return {};
    }
    ofs.write(jpeg.data(), static_cast<std::streamsize>(jpeg.size()));
    ofs.close();

    spdlog::info("SnapshotWriter: saved {} ({} bytes)", file_path, jpeg.size());
    return file_path;
}

}  // namespace hms
