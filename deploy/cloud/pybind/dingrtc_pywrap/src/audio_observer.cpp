#include "audio_observer.h"

#include <cstring>

namespace idingrtc {

using ding::rtc::RtcEngineAudioFrame;

namespace {

PcmFrame to_pcm_frame(const RtcEngineAudioFrame *raw, const char *uid) {
    PcmFrame f;
    if (!raw) return f;
    const std::size_t total_bytes =
        static_cast<std::size_t>(raw->samples) *
        static_cast<std::size_t>(raw->bytesPerSample) *
        static_cast<std::size_t>(raw->channels);
    if (raw->buffer != nullptr && total_bytes > 0) {
        f.pcm.resize(total_bytes);
        std::memcpy(f.pcm.data(), raw->buffer, total_bytes);
    }
    f.sample_rate = raw->samplesPerSec;
    f.channels = raw->channels;
    f.bytes_per_sample = raw->bytesPerSample;
    f.num_samples = raw->samples;
    f.remote_uid = uid ? std::string(uid) : std::string();
    f.timestamp = raw->timestamp;
    return f;
}

}  // namespace

AudioObserver::AudioObserver(std::shared_ptr<FrameRingBuffer> buffer,
                             std::string expected_remote_uid)
    : buffer_(std::move(buffer)),
      expected_remote_uid_(std::move(expected_remote_uid)) {}

void AudioObserver::OnCapturedAudioFrame(RtcEngineAudioFrame *) {}
void AudioObserver::OnProcessCapturedAudioFrame(RtcEngineAudioFrame *) {}
void AudioObserver::OnPublishAudioFrame(RtcEngineAudioFrame *) {}

void AudioObserver::OnPlaybackAudioFrame(RtcEngineAudioFrame *frame) {
    // 混流 path — iSales engine uses this in v1.0 (2-user channel ⇒ mixed
    // playback ≡ edge peer's audio). DingRTC Linux C++ SDK does not expose
    // per-uid PcmBeforMixing subscription, so this is the only path.
    if (buffer_) {
        buffer_->push(to_pcm_frame(frame, nullptr));
    }
}

void AudioObserver::OnRemoteUserAudioFrame(const char *uid,
                                            RtcEngineAudioFrame *frame) {
    if (!buffer_) return;
    if (!expected_remote_uid_.empty() &&
        (uid == nullptr || expected_remote_uid_ != uid)) {
        return;
    }
    buffer_->push(to_pcm_frame(frame, uid));
}

}  // namespace idingrtc
