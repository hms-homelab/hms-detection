#include "rtsp_capture.h"

#include <spdlog/spdlog.h>
#include <algorithm>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace hms {

RtspCapture::RtspCapture(std::string camera_id, std::string rtsp_url,
                         std::shared_ptr<FramePool> frame_pool,
                         FrameCallback on_frame)
    : camera_id_(std::move(camera_id))
    , rtsp_url_(std::move(rtsp_url))
    , frame_pool_(std::move(frame_pool))
    , on_frame_(std::move(on_frame)) {}

RtspCapture::~RtspCapture() {
    stop();
}

void RtspCapture::start() {
    if (running_.exchange(true)) return;  // already running
    thread_ = std::thread(&RtspCapture::captureLoop, this);
    spdlog::info("[{}] Capture thread started", camera_id_);
}

void RtspCapture::stop() {
    running_ = false;
    if (thread_.joinable()) {
        thread_.join();
    }
    closeStream();
    spdlog::info("[{}] Capture thread stopped", camera_id_);
}

RtspCapture::Stats RtspCapture::stats() const {
    return Stats{
        .frames_captured = frames_captured_.load(),
        .reconnect_count = reconnect_count_.load(),
        .consecutive_failures = consecutive_failures_.load(),
        .is_connected = is_connected_.load(),
        .last_frame_time = last_frame_time_.load(),
        .frame_width = frame_width_.load(),
        .frame_height = frame_height_.load(),
    };
}

bool RtspCapture::openStream() {
    // RTSP options: TCP transport, 5s timeout, no buffering
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    av_dict_set(&opts, "stimeout", "5000000", 0);    // 5s connection timeout (microseconds)
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);

    fmt_ctx_ = avformat_alloc_context();
    // Set a shorter interrupt timeout for av_read_frame
    fmt_ctx_->interrupt_callback.opaque = this;
    fmt_ctx_->interrupt_callback.callback = [](void* opaque) -> int {
        auto* self = static_cast<RtspCapture*>(opaque);
        return self->running_.load() ? 0 : 1;  // Return 1 to abort if stopping
    };

    int ret = avformat_open_input(&fmt_ctx_, rtsp_url_.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("[{}] Failed to open RTSP stream: {}", camera_id_, errbuf);
        closeStream();
        return false;
    }

    ret = avformat_find_stream_info(fmt_ctx_, nullptr);
    if (ret < 0) {
        spdlog::error("[{}] Failed to find stream info", camera_id_);
        closeStream();
        return false;
    }

    // Find video stream
    video_stream_idx_ = -1;
    for (unsigned i = 0; i < fmt_ctx_->nb_streams; ++i) {
        if (fmt_ctx_->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx_ = static_cast<int>(i);
            break;
        }
    }
    if (video_stream_idx_ < 0) {
        spdlog::error("[{}] No video stream found", camera_id_);
        closeStream();
        return false;
    }

    auto* codecpar = fmt_ctx_->streams[video_stream_idx_]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        spdlog::error("[{}] Decoder not found for codec {}", camera_id_,
                      avcodec_get_name(codecpar->codec_id));
        closeStream();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(codec_ctx_, codecpar);
    codec_ctx_->thread_count = 1;  // Single thread per camera decoder

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        spdlog::error("[{}] Failed to open codec", camera_id_);
        closeStream();
        return false;
    }

    // Allocate frames and packet
    av_frame_ = av_frame_alloc();
    bgr_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    spdlog::info("[{}] Connected: {}x{} ({})", camera_id_,
                 codec_ctx_->width, codec_ctx_->height,
                 avcodec_get_name(codecpar->codec_id));

    frame_width_ = codec_ctx_->width;
    frame_height_ = codec_ctx_->height;

    return true;
}

void RtspCapture::closeStream() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (packet_) {
        av_packet_free(&packet_);
        packet_ = nullptr;
    }
    if (bgr_frame_) {
        av_frame_free(&bgr_frame_);
        bgr_frame_ = nullptr;
    }
    if (av_frame_) {
        av_frame_free(&av_frame_);
        av_frame_ = nullptr;
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
        codec_ctx_ = nullptr;
    }
    if (fmt_ctx_) {
        avformat_close_input(&fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    video_stream_idx_ = -1;
    is_connected_ = false;
}

void RtspCapture::captureLoop() {
    int backoff_seconds = 5;
    uint64_t frame_counter = 0;

    while (running_) {
        // Connect if needed
        if (!fmt_ctx_) {
            spdlog::info("[{}] Connecting to RTSP stream...", camera_id_);
            if (openStream()) {
                is_connected_ = true;
                consecutive_failures_ = 0;
                backoff_seconds = 5;
            } else {
                ++consecutive_failures_;
                ++reconnect_count_;
                is_connected_ = false;
                spdlog::warn("[{}] Reconnect in {}s (attempt {})", camera_id_,
                             backoff_seconds, consecutive_failures_.load());

                // Exponential backoff with running_ checks
                auto deadline = Clock::now() + std::chrono::seconds(backoff_seconds);
                while (running_ && Clock::now() < deadline) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                backoff_seconds = std::min(backoff_seconds * 2, 60);
                continue;
            }
        }

        // Read frame
        int ret = av_read_frame(fmt_ctx_, packet_);
        if (ret < 0) {
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN)) {
                spdlog::warn("[{}] Stream ended or timeout, reconnecting...", camera_id_);
            } else {
                char errbuf[256];
                av_strerror(ret, errbuf, sizeof(errbuf));
                spdlog::warn("[{}] Read error: {}, reconnecting...", camera_id_, errbuf);
            }
            closeStream();
            ++reconnect_count_;
            continue;
        }

        // Only process video packets
        if (packet_->stream_index != video_stream_idx_) {
            av_packet_unref(packet_);
            continue;
        }

        // Decode
        ret = avcodec_send_packet(codec_ctx_, packet_);
        av_packet_unref(packet_);
        if (ret < 0) continue;

        while (avcodec_receive_frame(codec_ctx_, av_frame_) == 0) {
            int w = av_frame_->width;
            int h = av_frame_->height;

            // Initialize or reinitialize sws if resolution changed
            if (!sws_ctx_ || frame_width_ != w || frame_height_ != h) {
                if (sws_ctx_) sws_freeContext(sws_ctx_);
                sws_ctx_ = sws_getContext(
                    w, h, static_cast<AVPixelFormat>(av_frame_->format),
                    w, h, AV_PIX_FMT_BGR24,
                    SWS_BILINEAR, nullptr, nullptr, nullptr);
                if (!sws_ctx_) {
                    spdlog::error("[{}] Failed to create sws context", camera_id_);
                    continue;
                }
                frame_width_ = w;
                frame_height_ = h;
                spdlog::info("[{}] Resolution: {}x{}", camera_id_, w, h);
            }

            // Acquire frame from pool
            auto frame = frame_pool_->acquire();
            if (!frame) {
                spdlog::warn("[{}] Frame pool exhausted, dropping frame", camera_id_);
                continue;
            }

            // Resize pixel buffer if needed (lazy allocation)
            if (frame->width != w || frame->height != h) {
                frame->resize(w, h);
            }

            // Convert YUV → BGR24
            uint8_t* dst_data[1] = {frame->pixels.data()};
            int dst_linesize[1] = {frame->stride};
            sws_scale(sws_ctx_, av_frame_->data, av_frame_->linesize,
                      0, h, dst_data, dst_linesize);

            frame->timestamp = Clock::now();
            frame->frame_number = ++frame_counter;

            ++frames_captured_;
            last_frame_time_ = Clock::now();

            // Deliver to buffer
            on_frame_(std::move(frame));
        }
    }
}

}  // namespace hms
