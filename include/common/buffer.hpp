#pragma once

#include "common/common.h"

template <typename T>
class TripletBuffer
{
public:
    struct Element
    {
        T msg1;
        T msg2;
        std::chrono::milliseconds timestamp;
    };

public:
    TripletBuffer() : buffer_{std::make_shared<Element>(), std::make_shared<Element>(), std::make_shared<Element>()}
    {
    }
    TripletBuffer(const TripletBuffer &) = delete;
    TripletBuffer &operator=(const TripletBuffer &) = delete;
    TripletBuffer(TripletBuffer &&other) noexcept
        : buffer_{std::move(other.buffer_[0]), std::move(other.buffer_[1]), std::move(other.buffer_[2])},
          current_read_{other.current_read_.load()},
          last_written_{other.last_written_.load()}
    {
    }
    TripletBuffer &operator=(TripletBuffer &&other) noexcept
    {
        if (this != &other)
        {
            buffer_[0] = std::move(other.buffer_[0]);
            buffer_[1] = std::move(other.buffer_[1]);
            buffer_[2] = std::move(other.buffer_[2]);
            current_read_.store(other.current_read_.load());
            last_written_.store(other.last_written_.load());
        }
        return *this;
    }
    ~TripletBuffer() = default;

public:
    std::shared_ptr<Element> read()
    {
        size_t latest = last_written_.load(std::memory_order_acquire);
        current_read_.store(latest, std::memory_order_release);
        return buffer_[latest];
    }

    template <class F>
    void update(F &&writer)
    {
        size_t current_read = current_read_.load(std::memory_order_relaxed);
        size_t last_written = last_written_.load(std::memory_order_relaxed);
        size_t write_idx = 0;
        for (size_t i = 0; i < 3; ++i)
        {
            if (i != current_read && i != last_written)
            {
                write_idx = i;
                break;
            }
        }
        if (write_idx == current_read)
        {
            write_idx = (last_written + 1) % 3;
        }

        auto &target = buffer_[write_idx];
        writer(target->msg1, target->msg2);

        auto now = std::chrono::system_clock::now();
        target->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch());

        last_written_.store(write_idx, std::memory_order_release);
    }

private:
    std::shared_ptr<Element> buffer_[3];
    std::atomic<size_t> current_read_{0};
    std::atomic<size_t> last_written_{0};
};