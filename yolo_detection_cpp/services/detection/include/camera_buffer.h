#pragma once

#include "frame_data.h"

#include <shared_mutex>
#include <vector>

namespace hms {

/// Fixed-size ring buffer for frames from a single camera.
/// Writer (capture thread) takes exclusive lock; readers (HTTP handlers) take shared lock.
class CameraBuffer {
public:
    explicit CameraBuffer(size_t capacity)
        : capacity_(capacity), buffer_(capacity), head_(0), count_(0) {}

    /// Push a frame, overwriting the oldest if full. Called from capture thread.
    void push(std::shared_ptr<FrameData> frame) {
        std::unique_lock lock(mutex_);
        buffer_[head_] = std::move(frame);
        head_ = (head_ + 1) % capacity_;
        if (count_ < capacity_) ++count_;
    }

    /// Get the most recent frame, or nullptr if empty.
    std::shared_ptr<FrameData> getLatestFrame() const {
        std::shared_lock lock(mutex_);
        if (count_ == 0) return nullptr;
        size_t idx = (head_ + capacity_ - 1) % capacity_;
        return buffer_[idx];
    }

    /// Get all buffered frames in order from oldest to newest.
    std::vector<std::shared_ptr<FrameData>> getBuffer() const {
        std::shared_lock lock(mutex_);
        std::vector<std::shared_ptr<FrameData>> result;
        result.reserve(count_);
        size_t start = (head_ + capacity_ - count_) % capacity_;
        for (size_t i = 0; i < count_; ++i) {
            result.push_back(buffer_[(start + i) % capacity_]);
        }
        return result;
    }

    size_t size() const {
        std::shared_lock lock(mutex_);
        return count_;
    }

    size_t capacity() const { return capacity_; }

    void clear() {
        std::unique_lock lock(mutex_);
        for (auto& f : buffer_) f.reset();
        head_ = 0;
        count_ = 0;
    }

private:
    size_t capacity_;
    std::vector<std::shared_ptr<FrameData>> buffer_;
    size_t head_;   // next write position
    size_t count_;  // number of valid frames
    mutable std::shared_mutex mutex_;
};

}  // namespace hms
