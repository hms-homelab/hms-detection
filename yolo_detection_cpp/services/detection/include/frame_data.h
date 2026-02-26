#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace hms {

using Clock = std::chrono::steady_clock;

struct FrameData {
    std::vector<uint8_t> pixels;  // BGR24 interleaved
    int width = 0;
    int height = 0;
    int stride = 0;               // bytes per row (width * 3 for BGR24)
    Clock::time_point timestamp;
    uint64_t frame_number = 0;

    void resize(int w, int h) {
        width = w;
        height = h;
        stride = w * 3;
        pixels.resize(static_cast<size_t>(stride) * h);
    }
};

/// Pre-allocates N FrameData objects and recycles them via shared_ptr custom deleter.
/// Avoids malloc/free churn during steady-state capture.
class FramePool {
public:
    explicit FramePool(size_t capacity)
        : capacity_(capacity) {
        for (size_t i = 0; i < capacity_; ++i) {
            free_list_.push(std::make_unique<FrameData>());
        }
    }

    /// Acquire a frame from the pool. Returns nullptr if exhausted.
    /// The returned shared_ptr automatically recycles back to the pool on destruction.
    std::shared_ptr<FrameData> acquire() {
        std::unique_ptr<FrameData> frame;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (free_list_.empty()) return nullptr;
            frame = std::move(free_list_.front());
            free_list_.pop();
        }

        // Custom deleter recycles frame back to pool
        auto* pool = this;
        auto* raw = frame.release();
        return std::shared_ptr<FrameData>(raw, [pool](FrameData* f) {
            pool->recycle(std::unique_ptr<FrameData>(f));
        });
    }

    size_t capacity() const { return capacity_; }

    size_t available() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return free_list_.size();
    }

    size_t in_use() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return capacity_ - free_list_.size();
    }

private:
    void recycle(std::unique_ptr<FrameData> frame) {
        frame->frame_number = 0;
        std::lock_guard<std::mutex> lock(mutex_);
        free_list_.push(std::move(frame));
    }

    size_t capacity_;
    mutable std::mutex mutex_;
    std::queue<std::unique_ptr<FrameData>> free_list_;
};

}  // namespace hms
