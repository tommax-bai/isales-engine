// dingrtc_pywrap — pybind11 binding for DingRTC Linux C++ SDK 3.x.
//
// See openspec/changes/engine-rtc-dingrtc-migration/ for the full design.
//
// API surface mirrors what isales_engine.transport.dingrtc.DingRtcSession
// needs to satisfy isales-common's RtcSession ABC. The binding does NOT
// expose asyncio constructs — thread-safe queue draining happens in Python.
//
// Key API differences from windows-artc-pybind11 (`aliyun_artc_pywrap`):
//   * Namespace `ding::rtc::` (vs `AliRTCSdk::`)
//   * Class `RtcEngine` (vs `AliEngine`)
//   * No `IAliEngineMediaEngine` indirection — `SetExternalAudioSource` +
//     `PushExternalAudioFrame` are directly on `RtcEngine`
//   * No stream_id concept — single external audio source per engine
//   * JoinChannel takes `RtcEngineAuthInfo` (channelId/userId/appId/token/
//     gslbServer); no separate nonce/timestamp — DingRTC v3.0 binary token
//     packs all that into the token string itself
//   * Audio observer registered via `RegisterAudioFrameObserver` +
//     `EnableAudioFrameObserver(true, position)` — position = Playback (8)
//     for v1.0 混流 path

#include <pybind11/pybind11.h>
#include <pybind11/gil.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "audio_observer.h"
#include "engine_listener.h"
#include "ring_buffer.h"

#include "engine_interface.h"  // ding::rtc::RtcEngine
#include "engine_types.h"      // ding::rtc::RtcEngineAuthInfo etc.

#include <atomic>
#include <memory>
#include <mutex>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace idingrtc;

namespace {

// --- Custom exception -------------------------------------------------------

class DingRtcError : public std::runtime_error {
public:
    DingRtcError(int code, const std::string &msg)
        : std::runtime_error(msg), code_(code) {}
    int code() const noexcept { return code_; }

private:
    int code_;
};

// --- Audio frame observer position enum (subset) ----------------------------
// From engine_types.h:
//   RtcEngineAudioObservePositionCapture        = 1
//   RtcEngineAudioObservePositionProcessCapture = 2
//   RtcEngineAudioObservePositionPublish        = 4
//   RtcEngineAudioObservePositionPlayback       = 8    <-- iSales engine
//   RtcEngineAudioObservePositionRemoteUser     = 16
constexpr unsigned int kPositionPlayback = 8;
constexpr unsigned int kPositionRemoteUser = 16;

// --- Engine wrapper ---------------------------------------------------------
//
// EngineHandle owns:
//   * the ding::rtc::RtcEngine* (destroyed via static RtcEngine::Destroy)
//   * registered AudioObserver / EngineListener (kept alive via pybind11
//     keep_alive — see PYBIND11_MODULE binding section)
//
// Lifecycle invariants:
//   * destroy() must NOT race with audio-frame callbacks — caller (Python
//     adapter) must stop pushing frames and stop draining before destroy.
//   * destroy() is idempotent: second call is a no-op.

class EngineHandle {
public:
    EngineHandle() = default;
    ~EngineHandle() { destroy_unlocked(); }

    EngineHandle(const EngineHandle &) = delete;
    EngineHandle &operator=(const EngineHandle &) = delete;

    void create(const std::string &extras) {
        std::lock_guard<std::mutex> lk(mu_);
        if (engine_) {
            throw DingRtcError(/*EALREADY*/ -1, "engine already created");
        }
        py::gil_scoped_release nogil;
        engine_ = ding::rtc::RtcEngine::Create(extras.empty() ? nullptr : extras.c_str());
        if (!engine_) {
            throw DingRtcError(/*ENOMEM*/ -2, "RtcEngine::Create returned nullptr");
        }
    }

    bool is_alive() const {
        std::lock_guard<std::mutex> lk(mu_);
        return engine_ != nullptr;
    }

    void set_event_listener(EngineListener *listener) {
        std::lock_guard<std::mutex> lk(mu_);
        require_alive_locked();
        const int rc = engine_->SetEngineEventListener(listener);
        if (rc != 0) {
            throw DingRtcError(rc, "SetEngineEventListener failed");
        }
    }

    void register_audio_observer(AudioObserver *observer) {
        std::lock_guard<std::mutex> lk(mu_);
        require_alive_locked();
        const int rc = engine_->RegisterAudioFrameObserver(observer);
        if (rc != 0) {
            throw DingRtcError(rc, "RegisterAudioFrameObserver failed");
        }
        registered_observer_ = observer;
    }

    void unregister_audio_observer() {
        std::lock_guard<std::mutex> lk(mu_);
        if (!engine_ || !registered_observer_) return;
        engine_->RegisterAudioFrameObserver(nullptr);
        registered_observer_ = nullptr;
    }

    void enable_audio_frame_observer(bool enabled, unsigned int position) {
        std::lock_guard<std::mutex> lk(mu_);
        require_alive_locked();
        const int rc = engine_->EnableAudioFrameObserver(enabled, position);
        if (rc != 0) {
            throw DingRtcError(rc, "EnableAudioFrameObserver failed");
        }
    }

    void set_external_audio_source(bool enable, int sample_rate, int channels) {
        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->SetExternalAudioSource(enable, sample_rate, channels);
        }
        if (rc < 0) {
            throw DingRtcError(rc, "SetExternalAudioSource failed");
        }
    }

    void push_external_audio(py::bytes pcm, int sample_rate, int channels,
                              int bytes_per_sample, long long timestamp) {
        std::string buf = pcm;
        const int total_bytes = static_cast<int>(buf.size());
        if (channels <= 0 || bytes_per_sample <= 0) {
            throw DingRtcError(-3, "invalid channels/bytes_per_sample");
        }
        // Vendor SDK divides internally by samplesPerSec; passing 0
        // produces SIGFPE inside the audio pipeline (observed on Linux
        // 3.9.0). Reject up-front with a clear error so a caller bug
        // surfaces as an exception, not a crashed process.
        if (sample_rate <= 0) {
            throw DingRtcError(-3, "invalid sample_rate (must be > 0)");
        }
        const int num_samples = total_bytes / (bytes_per_sample * channels);

        ding::rtc::RtcEngineAudioFrame frame{};
        frame.type = ding::rtc::RtcEngineAudioFramePcm16;
        frame.bytesPerSample = bytes_per_sample;
        frame.samplesPerSec = sample_rate;
        frame.channels = channels;
        frame.samples = num_samples;
        frame.buffer = buf.data();
        frame.timestamp = timestamp;

        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->PushExternalAudioFrame(&frame);
        }
        if (rc != 0) {
            throw DingRtcError(rc, "PushExternalAudioFrame failed");
        }
    }

    void publish_local_audio_stream(bool enabled) {
        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->PublishLocalAudioStream(enabled);
        }
        if (rc != 0) {
            throw DingRtcError(rc, "PublishLocalAudioStream failed");
        }
    }

    void publish_local_video_stream(bool enabled) {
        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->PublishLocalVideoStream(enabled);
        }
        if (rc != 0) {
            throw DingRtcError(rc, "PublishLocalVideoStream failed");
        }
    }

    void subscribe_all_remote_audio_streams(bool sub) {
        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->SubscribeAllRemoteAudioStreams(sub);
        }
        if (rc != 0) {
            throw DingRtcError(rc, "SubscribeAllRemoteAudioStreams failed");
        }
    }

    /// JoinChannel via RtcEngineAuthInfo struct.
    ///
    /// `gslb_server` may be empty — DingRTC SDK falls back to its internal
    /// default (the GSLB endpoint is baked into libDingRTC.so per release).
    /// For 3.9.0 the documented value is "https://gslb.dingrtc.com".
    void join_channel(const std::string &channel_id,
                      const std::string &user_id,
                      const std::string &app_id,
                      const std::string &token,
                      const std::string &gslb_server,
                      const std::string &user_name) {
        // Build authInfo on stack — RtcEngine::JoinChannel reads the struct
        // synchronously before returning, so local-scope String members are
        // valid for the duration of the call.
        ding::rtc::RtcEngineAuthInfo authInfo;
        authInfo.channelId = channel_id.c_str();
        authInfo.userId    = user_id.c_str();
        authInfo.appId     = app_id.c_str();
        authInfo.token     = token.c_str();
        if (!gslb_server.empty()) {
            authInfo.gslbServer = gslb_server.c_str();
        }

        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->JoinChannel(authInfo,
                                       user_name.empty() ? nullptr : user_name.c_str());
        }
        if (rc != 0) {
            char buf[96];
            std::snprintf(buf, sizeof(buf),
                          "JoinChannel returned rc=%d (0x%08X)", rc, rc);
            throw DingRtcError(rc, buf);
        }
    }

    void leave_channel() {
        int rc;
        {
            std::lock_guard<std::mutex> lk(mu_);
            require_alive_locked();
            py::gil_scoped_release nogil;
            rc = engine_->LeaveChannel();
        }
        if (rc != 0) {
            throw DingRtcError(rc, "LeaveChannel returned non-zero");
        }
    }

    void destroy() {
        std::lock_guard<std::mutex> lk(mu_);
        destroy_unlocked();
    }

    /// Static helpers (don't require an engine instance).
    static int set_log_dir_path(const std::string &path) {
        return ding::rtc::RtcEngine::SetLogDirPath(path.c_str());
    }

private:
    void require_alive_locked() {
        if (!engine_) throw DingRtcError(-4, "engine is destroyed or not created");
    }
    void destroy_unlocked() {
        if (engine_) {
            if (registered_observer_) {
                engine_->RegisterAudioFrameObserver(nullptr);
                registered_observer_ = nullptr;
            }
            engine_->SetEngineEventListener(nullptr);
            // GIL release: Destroy can block on internal thread joining.
            py::gil_scoped_release nogil;
            ding::rtc::RtcEngine::Destroy(engine_);
            engine_ = nullptr;
        }
    }

    mutable std::mutex mu_;
    ding::rtc::RtcEngine *engine_ = nullptr;
    ding::rtc::RtcEngineAudioFrameObserver *registered_observer_ = nullptr;
};

}  // namespace

PYBIND11_MODULE(dingrtc_pywrap, m) {
    m.doc() = "Project-internal pybind11 binding for the DingRTC Linux C++ "
              "SDK 3.x. See openspec/changes/engine-rtc-dingrtc-migration/.";

    // Module version — bumped manually when binding ABI changes.
    m.attr("__version__") = "0.1.0";

    py::register_exception<DingRtcError>(m, "DingRtcError");

    // Audio frame observer position enum (subset for v1.0 audio-only).
    m.attr("POSITION_PLAYBACK") = kPositionPlayback;
    m.attr("POSITION_REMOTE_USER") = kPositionRemoteUser;

    py::class_<PcmFrame>(m, "PcmFrame")
        .def_readonly("pcm",              &PcmFrame::pcm)
        .def_readonly("sample_rate",      &PcmFrame::sample_rate)
        .def_readonly("channels",         &PcmFrame::channels)
        .def_readonly("bytes_per_sample", &PcmFrame::bytes_per_sample)
        .def_readonly("num_samples",      &PcmFrame::num_samples)
        .def_readonly("remote_uid",       &PcmFrame::remote_uid)
        .def_readonly("timestamp",        &PcmFrame::timestamp);

    py::class_<FrameRingBuffer, std::shared_ptr<FrameRingBuffer>>(m, "FrameRingBuffer")
        .def(py::init<std::size_t>(), py::arg("capacity_frames") = 64)
        .def("drain",
             [](FrameRingBuffer &self, std::size_t max_n) {
                 std::vector<PcmFrame> out;
                 self.drain(out, max_n);
                 return out;
             },
             py::arg("max_n") = 8)
        .def("dropped",      &FrameRingBuffer::dropped)
        .def("size_approx",  &FrameRingBuffer::size_approx)
        .def("clear",        &FrameRingBuffer::clear);

    py::class_<AudioObserver, std::shared_ptr<AudioObserver>>(m, "AudioObserver")
        .def(py::init<std::shared_ptr<FrameRingBuffer>, std::string>(),
             py::arg("buffer"), py::arg("expected_remote_uid") = std::string())
        .def("buffer", &AudioObserver::buffer);

    py::class_<EngineListener, std::shared_ptr<EngineListener>>(m, "EngineListener")
        .def(py::init<>())
        .def("set_on_join",                       &EngineListener::set_on_join)
        .def("set_on_leave",                      &EngineListener::set_on_leave)
        .def("set_on_bye",                        &EngineListener::set_on_bye)
        .def("set_on_error",                      &EngineListener::set_on_error)
        .def("set_on_connection_status_change",   &EngineListener::set_on_connection_status_change)
        .def("clear_callbacks",                   &EngineListener::clear_callbacks);

    py::class_<EngineHandle>(m, "EngineHandle")
        .def(py::init<>())
        .def("create",     &EngineHandle::create, py::arg("extras") = std::string())
        .def("is_alive",   &EngineHandle::is_alive)
        .def("set_event_listener",
             [](EngineHandle &self, std::shared_ptr<EngineListener> l) {
                 self.set_event_listener(l.get());
             },
             py::arg("listener"),
             py::keep_alive<1, 2>())
        .def("register_audio_observer",
             [](EngineHandle &self, std::shared_ptr<AudioObserver> obs) {
                 self.register_audio_observer(obs.get());
             },
             py::arg("observer"),
             py::keep_alive<1, 2>())
        .def("unregister_audio_observer",         &EngineHandle::unregister_audio_observer)
        .def("enable_audio_frame_observer",       &EngineHandle::enable_audio_frame_observer,
             py::arg("enabled"), py::arg("position"))
        .def("set_external_audio_source",         &EngineHandle::set_external_audio_source,
             py::arg("enable"), py::arg("sample_rate") = 16000, py::arg("channels") = 1)
        .def("push_external_audio",               &EngineHandle::push_external_audio,
             py::arg("pcm"), py::arg("sample_rate") = 16000, py::arg("channels") = 1,
             py::arg("bytes_per_sample") = 2, py::arg("timestamp") = 0)
        .def("publish_local_audio_stream",        &EngineHandle::publish_local_audio_stream,
             py::arg("enabled"))
        .def("publish_local_video_stream",        &EngineHandle::publish_local_video_stream,
             py::arg("enabled"))
        .def("subscribe_all_remote_audio_streams", &EngineHandle::subscribe_all_remote_audio_streams,
             py::arg("sub"))
        .def("join_channel",                      &EngineHandle::join_channel,
             py::arg("channel_id"), py::arg("user_id"), py::arg("app_id"),
             py::arg("token"),
             py::arg("gslb_server") = std::string("https://gslb.dingrtc.com"),
             py::arg("user_name") = std::string())
        .def("leave_channel",                     &EngineHandle::leave_channel)
        .def("destroy",                           &EngineHandle::destroy);

    m.def("set_log_dir_path", &EngineHandle::set_log_dir_path, py::arg("path"),
          "Set DingRTC SDK log directory (static, applies before any "
          "EngineHandle.create call).");
}
