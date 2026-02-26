#include "event_recorder.h"

#include <spdlog/spdlog.h>
#include <filesystem>
#include <chrono>
#include <ctime>

namespace fs = std::filesystem;

namespace hms {

EventRecorder::EventRecorder() = default;

EventRecorder::~EventRecorder() {
    if (recording_) {
        finalize();
    }
    cleanup();
}

static std::string makeTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf{};
    localtime_r(&t, &tm_buf);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
    return buf;
}

bool EventRecorder::start(const std::string& camera_id,
                           const std::vector<std::shared_ptr<FrameData>>& preroll_frames,
                           int width, int height, int fps,
                           const std::string& output_dir) {
    camera_id_ = camera_id;
    width_ = width;
    height_ = height;
    fps_ = fps > 0 ? fps : 10;
    frames_written_ = 0;
    pts_ = 0;
    stop_requested_ = false;

    // Create output directory
    fs::create_directories(output_dir);

    // Generate filename: camera_id_YYYYMMDD_HHMMSS.mp4
    file_path_ = output_dir + "/" + camera_id + "_" + makeTimestamp() + ".mp4";

    // Allocate output format context (MP4)
    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", file_path_.c_str());
    if (ret < 0 || !fmt_ctx_) {
        spdlog::error("EventRecorder: failed to create MP4 context for {}", file_path_);
        return false;
    }

    // Find H.264 encoder
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        spdlog::error("EventRecorder: H.264 encoder not found");
        cleanup();
        return false;
    }

    // Create stream
    stream_ = avformat_new_stream(fmt_ctx_, codec);
    if (!stream_) {
        spdlog::error("EventRecorder: failed to create stream");
        cleanup();
        return false;
    }
    stream_->id = 0;

    // Allocate and configure encoder context
    enc_ctx_ = avcodec_alloc_context3(codec);
    enc_ctx_->width = width_;
    enc_ctx_->height = height_;
    enc_ctx_->time_base = {1, fps_};
    enc_ctx_->framerate = {fps_, 1};
    enc_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    enc_ctx_->gop_size = fps_;  // keyframe every second

    // Ultrafast preset, CRF 28 (good quality/size tradeoff)
    av_opt_set(enc_ctx_->priv_data, "preset", "ultrafast", 0);
    av_opt_set(enc_ctx_->priv_data, "crf", "28", 0);

    // For MP4 container compatibility
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        enc_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    ret = avcodec_open2(enc_ctx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("EventRecorder: failed to open H.264 encoder: {}", errbuf);
        cleanup();
        return false;
    }

    // Copy codec parameters to stream
    avcodec_parameters_from_context(stream_->codecpar, enc_ctx_);
    stream_->time_base = enc_ctx_->time_base;

    // Open output file with faststart (moov at beginning for streaming)
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "movflags", "+faststart", 0);

    ret = avio_open(&fmt_ctx_->pb, file_path_.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
        char errbuf[128];
        av_strerror(ret, errbuf, sizeof(errbuf));
        spdlog::error("EventRecorder: failed to open {}: {}", file_path_, errbuf);
        av_dict_free(&opts);
        cleanup();
        return false;
    }

    ret = avformat_write_header(fmt_ctx_, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        spdlog::error("EventRecorder: failed to write header");
        cleanup();
        return false;
    }

    // Allocate YUV frame for encoding
    yuv_frame_ = av_frame_alloc();
    yuv_frame_->format = AV_PIX_FMT_YUV420P;
    yuv_frame_->width = width_;
    yuv_frame_->height = height_;
    av_frame_get_buffer(yuv_frame_, 0);

    // Allocate packet
    pkt_ = av_packet_alloc();

    // BGR24 → YUV420P converter
    sws_ctx_ = sws_getContext(width_, height_, AV_PIX_FMT_BGR24,
                               width_, height_, AV_PIX_FMT_YUV420P,
                               SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        spdlog::error("EventRecorder: failed to create sws context");
        cleanup();
        return false;
    }

    recording_ = true;
    spdlog::info("EventRecorder: started recording {} ({}x{} @ {}fps)",
                 file_path_, width_, height_, fps_);

    // Write pre-roll frames
    for (const auto& frame : preroll_frames) {
        if (frame && frame->width == width_ && frame->height == height_) {
            writeFrame(*frame);
        }
    }

    return true;
}

bool EventRecorder::writeFrame(const FrameData& frame) {
    if (!recording_ || !enc_ctx_ || !fmt_ctx_) return false;

    // Check max duration
    if (isMaxDurationReached()) return false;

    av_frame_make_writable(yuv_frame_);

    // BGR24 → YUV420P
    const uint8_t* src_data[1] = {frame.pixels.data()};
    int src_linesize[1] = {frame.stride};
    sws_scale(sws_ctx_, src_data, src_linesize, 0, height_,
              yuv_frame_->data, yuv_frame_->linesize);

    yuv_frame_->pts = pts_++;

    // Encode
    int ret = avcodec_send_frame(enc_ctx_, yuv_frame_);
    if (ret < 0) return false;

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) return false;

        av_packet_rescale_ts(pkt_, enc_ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;

        ret = av_interleaved_write_frame(fmt_ctx_, pkt_);
        av_packet_unref(pkt_);
        if (ret < 0) return false;
    }

    frames_written_++;
    return true;
}

void EventRecorder::requestStop(int post_roll_seconds) {
    if (!stop_requested_) {
        stop_requested_ = true;
        post_roll_seconds_ = post_roll_seconds;
        stop_requested_time_ = Clock::now();
        spdlog::debug("EventRecorder: stop requested for {}, post-roll {}s",
                      camera_id_, post_roll_seconds);
    }
}

bool EventRecorder::finalize() {
    if (!recording_ || !enc_ctx_ || !fmt_ctx_) return false;

    recording_ = false;

    // Flush encoder
    avcodec_send_frame(enc_ctx_, nullptr);
    while (true) {
        int ret = avcodec_receive_packet(enc_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) break;

        av_packet_rescale_ts(pkt_, enc_ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;
        av_interleaved_write_frame(fmt_ctx_, pkt_);
        av_packet_unref(pkt_);
    }

    av_write_trailer(fmt_ctx_);

    double duration = static_cast<double>(frames_written_) / fps_;
    spdlog::info("EventRecorder: finalized {} ({} frames, {:.1f}s)",
                 file_path_, frames_written_, duration);

    cleanup();
    return true;
}

std::string EventRecorder::fileName() const {
    return fs::path(file_path_).filename().string();
}

bool EventRecorder::isPostRollComplete() const {
    if (!stop_requested_) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        Clock::now() - stop_requested_time_).count();
    return elapsed >= post_roll_seconds_;
}

bool EventRecorder::isMaxDurationReached() const {
    return frames_written_ >= (fps_ * MAX_DURATION_SECONDS);
}

void EventRecorder::cleanup() {
    if (sws_ctx_) { sws_freeContext(sws_ctx_); sws_ctx_ = nullptr; }
    if (yuv_frame_) { av_frame_free(&yuv_frame_); }
    if (pkt_) { av_packet_free(&pkt_); }
    if (enc_ctx_) { avcodec_free_context(&enc_ctx_); }
    if (fmt_ctx_) {
        if (fmt_ctx_->pb) { avio_closep(&fmt_ctx_->pb); }
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    stream_ = nullptr;
}

}  // namespace hms
