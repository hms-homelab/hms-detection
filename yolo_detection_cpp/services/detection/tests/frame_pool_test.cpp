#include <catch2/catch_all.hpp>
#include "frame_data.h"

#include <thread>
#include <vector>
#include <atomic>

using namespace hms;

TEST_CASE("FramePool basic allocation", "[frame_pool]") {
    FramePool pool(5);

    REQUIRE(pool.capacity() == 5);
    REQUIRE(pool.available() == 5);
    REQUIRE(pool.in_use() == 0);

    auto f1 = pool.acquire();
    REQUIRE(f1 != nullptr);
    REQUIRE(pool.available() == 4);
    REQUIRE(pool.in_use() == 1);

    auto f2 = pool.acquire();
    auto f3 = pool.acquire();
    REQUIRE(pool.available() == 2);
    REQUIRE(pool.in_use() == 3);
}

TEST_CASE("FramePool exhaustion returns nullptr", "[frame_pool]") {
    FramePool pool(3);

    auto f1 = pool.acquire();
    auto f2 = pool.acquire();
    auto f3 = pool.acquire();
    REQUIRE(pool.available() == 0);

    auto f4 = pool.acquire();
    REQUIRE(f4 == nullptr);
}

TEST_CASE("FramePool recycles on shared_ptr destruction", "[frame_pool]") {
    FramePool pool(2);

    auto f1 = pool.acquire();
    auto f2 = pool.acquire();
    REQUIRE(pool.available() == 0);

    // Release f1 — should recycle back to pool
    f1.reset();
    REQUIRE(pool.available() == 1);

    // Should be able to acquire again
    auto f3 = pool.acquire();
    REQUIRE(f3 != nullptr);
    REQUIRE(pool.available() == 0);
}

TEST_CASE("FramePool frame_number reset on recycle", "[frame_pool]") {
    FramePool pool(1);

    auto f1 = pool.acquire();
    f1->frame_number = 42;
    f1.reset();  // recycle

    auto f2 = pool.acquire();
    REQUIRE(f2->frame_number == 0);  // reset by recycle
}

TEST_CASE("FrameData resize", "[frame_data]") {
    FrameData frame;
    frame.resize(640, 480);

    REQUIRE(frame.width == 640);
    REQUIRE(frame.height == 480);
    REQUIRE(frame.stride == 640 * 3);
    REQUIRE(frame.pixels.size() == 640 * 3 * 480);
}

TEST_CASE("FramePool concurrent access", "[frame_pool]") {
    FramePool pool(100);
    std::atomic<int> acquired{0};
    std::atomic<int> failed{0};

    auto worker = [&]() {
        for (int i = 0; i < 50; ++i) {
            auto f = pool.acquire();
            if (f) {
                ++acquired;
                // Simulate brief use
                std::this_thread::sleep_for(std::chrono::microseconds(10));
                f.reset();  // recycle
            } else {
                ++failed;
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();

    // All frames should be recycled back
    REQUIRE(pool.available() == 100);
    // Total attempts: 200, most should succeed
    REQUIRE(acquired.load() > 100);
}
