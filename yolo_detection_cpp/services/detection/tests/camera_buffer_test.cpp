#include <catch2/catch_all.hpp>
#include "camera_buffer.h"

#include <thread>
#include <vector>

using namespace hms;

namespace {
std::shared_ptr<FrameData> makeFrame(uint64_t num, int w = 4, int h = 4) {
    auto f = std::make_shared<FrameData>();
    f->frame_number = num;
    f->resize(w, h);
    f->timestamp = Clock::now();
    return f;
}
}  // namespace

TEST_CASE("CameraBuffer push and size", "[camera_buffer]") {
    CameraBuffer buf(5);
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.capacity() == 5);

    buf.push(makeFrame(1));
    REQUIRE(buf.size() == 1);

    buf.push(makeFrame(2));
    buf.push(makeFrame(3));
    REQUIRE(buf.size() == 3);
}

TEST_CASE("CameraBuffer ring overflow oldest-first", "[camera_buffer]") {
    CameraBuffer buf(3);

    buf.push(makeFrame(1));
    buf.push(makeFrame(2));
    buf.push(makeFrame(3));
    REQUIRE(buf.size() == 3);

    // Overflow: should evict frame 1
    buf.push(makeFrame(4));
    REQUIRE(buf.size() == 3);

    auto latest = buf.getLatestFrame();
    REQUIRE(latest->frame_number == 4);

    auto all = buf.getBuffer();
    REQUIRE(all.size() == 3);
    REQUIRE(all[0]->frame_number == 2);  // oldest surviving
    REQUIRE(all[1]->frame_number == 3);
    REQUIRE(all[2]->frame_number == 4);  // newest
}

TEST_CASE("CameraBuffer getLatestFrame on empty", "[camera_buffer]") {
    CameraBuffer buf(5);
    REQUIRE(buf.getLatestFrame() == nullptr);
}

TEST_CASE("CameraBuffer getLatestFrame returns newest", "[camera_buffer]") {
    CameraBuffer buf(10);
    buf.push(makeFrame(10));
    buf.push(makeFrame(20));
    buf.push(makeFrame(30));

    REQUIRE(buf.getLatestFrame()->frame_number == 30);
}

TEST_CASE("CameraBuffer getBuffer returns ordered oldest to newest", "[camera_buffer]") {
    CameraBuffer buf(5);

    for (uint64_t i = 1; i <= 5; ++i) {
        buf.push(makeFrame(i));
    }

    auto all = buf.getBuffer();
    REQUIRE(all.size() == 5);
    for (size_t i = 0; i < 5; ++i) {
        REQUIRE(all[i]->frame_number == i + 1);
    }
}

TEST_CASE("CameraBuffer clear", "[camera_buffer]") {
    CameraBuffer buf(5);
    buf.push(makeFrame(1));
    buf.push(makeFrame(2));
    REQUIRE(buf.size() == 2);

    buf.clear();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.getLatestFrame() == nullptr);
}

TEST_CASE("CameraBuffer concurrent read/write", "[camera_buffer]") {
    CameraBuffer buf(100);
    std::atomic<bool> running{true};
    std::atomic<int> reads{0};
    std::atomic<bool> order_violation{false};

    // Writer thread
    std::thread writer([&]() {
        for (uint64_t i = 0; i < 500; ++i) {
            buf.push(makeFrame(i));
        }
        running = false;
    });

    // Reader threads — avoid Catch2 REQUIRE inside threads (not thread-safe)
    std::vector<std::thread> readers;
    for (int r = 0; r < 3; ++r) {
        readers.emplace_back([&]() {
            while (running.load()) {
                auto frame = buf.getLatestFrame();
                if (frame) ++reads;
                auto all = buf.getBuffer();
                for (size_t i = 1; i < all.size(); ++i) {
                    if (all[i]->frame_number <= all[i - 1]->frame_number) {
                        order_violation = true;
                    }
                }
            }
        });
    }

    writer.join();
    for (auto& t : readers) t.join();

    REQUIRE(buf.size() == 100);
    REQUIRE(reads.load() > 0);
    REQUIRE_FALSE(order_violation.load());
}
