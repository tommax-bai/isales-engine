#pragma once

#include "ring_buffer.h"

#include <memory>
#include <string>

// RtcEngineAudioFrameObserver is declared in engine_interface.h (not
// engine_types.h, which only has structs / enums / error codes).
// engine_interface.h transitively includes engine_types.h, so RtcEngineAudioFrame
// is reachable too.
#include "engine_interface.h"

namespace idingrtc {

/// RtcEngineAudioFrameObserver implementation that hands inbound PCM to a
/// Python-side ring buffer.
///
/// IMPORTANT: every On*AudioFrame callback runs on the DingRTC SDK audio
/// I/O thread, NOT the Python main thread. Do NOT call into Python here —
/// copy PCM bytes, push to `buffer_`, return.
///
/// For iSales engine (混流模式, v1.0): we register for
/// `RtcEngineAudioObservePositionPlayback` via EnableAudioFrameObserver,
/// and consume frames via OnPlaybackAudioFrame (混流 of all remote users
/// — in 2-user channel that's just the edge peer's audio).
class AudioObserver final : public ding::rtc::RtcEngineAudioFrameObserver {
public:
    AudioObserver(std::shared_ptr<FrameRingBuffer> buffer,
                  std::string expected_remote_uid);
    ~AudioObserver() override = default;

    // RtcEngineAudioFrameObserver overrides — vendor passes by REFERENCE
    // (not pointer); copy bytes before return. OnPlaybackAudioFrame /
    // OnCapturedAudioFrame / OnProcessCapturedAudioFrame / OnPublishAudioFrame
    // are pure virtual (must override even when unused). OnRemoteUserAudioFrame
    // has a default empty body.
    void OnCapturedAudioFrame(ding::rtc::RtcEngineAudioFrame &frame) override;
    void OnProcessCapturedAudioFrame(ding::rtc::RtcEngineAudioFrame &frame) override;
    void OnPublishAudioFrame(ding::rtc::RtcEngineAudioFrame &frame) override;
    void OnPlaybackAudioFrame(ding::rtc::RtcEngineAudioFrame &frame) override;
    void OnRemoteUserAudioFrame(const char *uid,
                                 ding::rtc::RtcEngineAudioFrame &frame) override;

    std::shared_ptr<FrameRingBuffer> buffer() const { return buffer_; }

private:
    std::shared_ptr<FrameRingBuffer> buffer_;
    std::string expected_remote_uid_;  // optional filter; empty = accept all
};

}  // namespace idingrtc
