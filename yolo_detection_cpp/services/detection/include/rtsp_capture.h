#pragma once

#include "frame_data.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

// Forward-declare FFmpeg types to keep header clean
struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace hms {

/// Per-camera RTSP capture using FFmpeg libav*.
/// Runs a dedicated thread that decodes H.264 → BGR24 and delivers frames via callback.
class RtspCapture {
public:
    using FrameCallback = std::function<void(std::shared_ptr<FrameData>)>;

    struct Stats {
        uint64_t frames_captured = 0;
        uint64_t reconnect_count = 0;
        uint64_t consecutive_failures = 0;
        bool is_connected = false;
        Clock::time_point last_frame_time;
        int frame_width = 0;
        int frame_height = 0;
    };

    RtspCapture(std::string camera_id, std::string rtsp_url,
                std::shared_ptr<FramePool> frame_pool,
                FrameCallback on_frame);
    ~RtspCapture();

    RtspCapture(const RtspCapture&) = delete;
    RtspCapture& operator=(const RtspCapture&) = delete;

    void start();
    void stop();

    Stats stats() const;

private:
    void captureLoop();
    bool openStream();
    void closeStream();

    std::string camera_id_;
    std::string rtsp_url_;
    std::shared_ptr<FramePool> frame_pool_;
    FrameCallback on_frame_;

    std::thread thread_;
    std::atomic<bool> running_{false};

    // FFmpeg state (owned by capture thread)
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    AVFrame* bgr_frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int video_stream_idx_ = -1;

    // Stats (atomic for lock-free reads from HTTP threads)
    std::atomic<uint64_t> frames_captured_{0};
    std::atomic<uint64_t> reconnect_count_{0};
    std::atomic<uint64_t> consecutive_failures_{0};
    std::atomic<bool> is_connected_{false};
    std::atomic<Clock::time_point> last_frame_time_{};
    std::atomic<int> frame_width_{0};
    std::atomic<int> frame_height_{0};
};

}  // namespace hms
