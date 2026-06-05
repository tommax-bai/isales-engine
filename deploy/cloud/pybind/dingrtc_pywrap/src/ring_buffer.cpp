#include "ring_buffer.h"

#include <algorithm>

namespace idingrtc {

FrameRingBuffer::FrameRingBuffer(std::size_t capacity_frames)
    : capacity_(capacity_frames), ring_(capacity_frames) {}

bool FrameRingBuffer::push(PcmFrame frame) {
    std::lock_guard<std::mutex> lk(mu_);
    bool dropped_oldest = false;
    if (size_.load(std::memory_order_relaxed) == capacity_) {
        // overwrite oldest
        tail_ = (tail_ + 1) % capacity_;
        size_.fetch_sub(1, std::memory_order_relaxed);
        dropped_.fetch_add(1, std::memory_order_relaxed);
        dropped_oldest = true;
    }
    ring_[head_] = std::move(frame);
    head_ = (head_ + 1) % capacity_;
    size_.fetch_add(1, std::memory_order_relaxed);
    return !dropped_oldest;
}

std::size_t FrameRingBuffer::drain(std::vector<PcmFrame> &out, std::size_t max_n) {
    std::lock_guard<std::mutex> lk(mu_);
    const std::size_t n = std::min(max_n, size_.load(std::memory_order_relaxed));
    out.reserve(out.size() + n);
    for (std::size_t i = 0; i < n; ++i) {
        out.emplace_back(std::move(ring_[tail_]));
        tail_ = (tail_ + 1) % capacity_;
    }
    size_.fetch_sub(n, std::memory_order_relaxed);
    return n;
}

void FrameRingBuffer::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    head_ = tail_ = 0;
    size_.store(0, std::memory_order_relaxed);
}

}  // namespace idingrtc
