#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace idingrtc {

/// One captured remote audio frame, ready to hand to Python.
///
/// Owns its PCM bytes — the SDK callback's `void*` buffer is copied on the
/// SDK audio thread so the Python drainer task can pick it up safely later.
///
/// Mirror of isales-telephony's windows binding `iartc::PcmFrame` (same
/// fields, different namespace) — Python wrappers can interop via duck
/// typing.
struct PcmFrame {
    std::vector<uint8_t> pcm;       // raw PCM bytes (16-bit signed LE)
    int sample_rate = 0;            // e.g. 16000
    int channels = 0;               // 1 = mono
    int bytes_per_sample = 0;       // 2 for PCM16
    int num_samples = 0;            // per channel
    std::string remote_uid;         // set for OnRemoteUserAudioFrame, empty for OnPlaybackAudioFrame (混流)
    int64_t timestamp = 0;          // DingRTC RtcEngineAudioFrame.timestamp
};

/// Bounded MPSC frame buffer.
///
/// The DingRTC SDK invokes audio observer callbacks from its own audio I/O
/// thread (single-producer). The Python drainer is a single asyncio task
/// (single-consumer). Mutex-protected deque is sufficient at the ~50 Hz frame
/// rate / ~640-byte frame size; lock-free SPSC isn't worth the complexity
/// for v1.0.
///
/// On overflow (Python drainer too slow), the OLDEST frame is dropped and
/// `dropped` is incremented. Drop-oldest keeps freshest audio flowing.
class FrameRingBuffer {
public:
    explicit FrameRingBuffer(std::size_t capacity_frames);

    /// Push a frame. Called from SDK audio I/O thread, GIL NOT held.
    /// Returns true if accepted without dropping, false if oldest was dropped.
    bool push(PcmFrame frame);

    /// Drain up to `max_n` frames into `out`. Called from Python drainer,
    /// caller holds the GIL. Returns the number drained (may be 0).
    std::size_t drain(std::vector<PcmFrame> &out, std::size_t max_n);

    /// Total frames dropped since construction.
    std::uint64_t dropped() const noexcept {
        return dropped_.load(std::memory_order_relaxed);
    }

    /// Current depth (best-effort, no lock).
    std::size_t size_approx() const noexcept {
        return size_.load(std::memory_order_relaxed);
    }

    void clear();

private:
    const std::size_t capacity_;
    std::mutex mu_;
    std::vector<PcmFrame> ring_;
    std::size_t head_ = 0;  // next write
    std::size_t tail_ = 0;  // next read
    std::atomic<std::size_t> size_{0};
    std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace idingrtc
