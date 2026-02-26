#pragma once

#include "frame_data.h"

#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace hms {

/// Records BGR24 frames to H.264 MP4 using libavcodec/libavformat.
/// Supports pre-roll frames, post-roll timer, and max duration cap.
class EventRecorder {
public:
    EventRecorder();
    ~EventRecorder();

    EventRecorder(const EventRecorder&) = delete;
    EventRecorder& operator=(const EventRecorder&) = delete;

    /// Start recording. Writes pre-roll frames immediately.
    /// output_dir: directory for MP4 files (e.g. /mnt/ssd/events)
    bool start(const std::string& camera_id,
               const std::vector<std::shared_ptr<FrameData>>& preroll_frames,
               int width, int height, int fps = 10,
               const std::string& output_dir = "/mnt/ssd/events");

    /// Write a single BGR24 frame
    bool writeFrame(const FrameData& frame);

    /// Request stop with post-roll seconds. Recording continues for duration.
    void requestStop(int post_roll_seconds = 5);

    /// Flush encoder, write trailer, close file. Returns true on success.
    bool finalize();

    /// Full file path of the recording
    std::string filePath() const { return file_path_; }

    /// Just the filename (no directory)
    std::string fileName() const;

    /// Whether requestStop() has been called
    bool isStopRequested() const { return stop_requested_; }

    /// Whether post-roll time has elapsed
    bool isPostRollComplete() const;

    /// Whether recording is active
    bool isRecording() const { return recording_; }

    /// Number of frames written so far
    int framesWritten() const { return frames_written_; }

    /// Maximum recording duration in seconds
    static constexpr int MAX_DURATION_SECONDS = 30;

    /// Whether max duration has been reached
    bool isMaxDurationReached() const;

private:
    void cleanup();

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* enc_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    AVFrame* yuv_frame_ = nullptr;
    AVPacket* pkt_ = nullptr;

    std::string file_path_;
    std::string camera_id_;
    int width_ = 0, height_ = 0, fps_ = 10;
    int frames_written_ = 0;
    int64_t pts_ = 0;
    bool recording_ = false;
    bool stop_requested_ = false;
    Clock::time_point stop_requested_time_;
    int post_roll_seconds_ = 5;
};

}  // namespace hms
