#pragma once

#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace hms {

/// Lightweight coordinator for GPU resource sharing between EventManager
/// (YOLO + LLaVA) and PeriodicSnapshotManager (moondream).
///
/// Design: Events always have priority. When an event starts, periodic
/// LLaVA/moondream inference should abort quickly so Ollama can evict
/// moondream and load LLaVA for event context.
class GpuCoordinator {
public:
    /// Called by EventManager when a motion event begins processing.
    /// Periodic snapshot LLaVA calls should check abort flag and bail out.
    void eventStarted() {
        event_active_.store(true, std::memory_order_release);
        abort_periodic_.store(true, std::memory_order_release);
    }

    /// Called by EventManager when event processing is fully done
    /// (including LLaVA context). Clears the abort flag so periodic
    /// snapshots can resume moondream inference.
    void eventFinished() {
        event_active_.store(false, std::memory_order_release);
        abort_periodic_.store(false, std::memory_order_release);
    }

    /// True while any motion event is being processed.
    bool isEventActive() const {
        return event_active_.load(std::memory_order_acquire);
    }

    /// True when periodic LLaVA/moondream inference should abort.
    /// Checked by VisionClient via curl progress callback.
    bool shouldAbortPeriodic() const {
        return abort_periodic_.load(std::memory_order_acquire);
    }

    /// Reference to the abort flag — passed to VisionClient for curl progress callback.
    const std::atomic<bool>& abortPeriodicFlag() const {
        return abort_periodic_;
    }

private:
    std::atomic<bool> event_active_{false};
    std::atomic<bool> abort_periodic_{false};
};

}  // namespace hms
