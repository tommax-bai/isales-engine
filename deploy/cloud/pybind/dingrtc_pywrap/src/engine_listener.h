#pragma once

#include <pybind11/pybind11.h>

#include <mutex>
#include <string>

#include "engine_types.h"  // ding::rtc::RtcEngineEventListener / RtcEngineStats

namespace idingrtc {

namespace py = pybind11;

/// RtcEngineEventListener subclass — marshals join / leave / connection
/// status / error events back to Python callables.
///
/// Event rate < 10 Hz in steady state; trampoline + GIL acquire overhead
/// is negligible (mirror of windows-artc-pybind11 engine_listener Decision).
///
/// Convention: Python side calls set_on_*(...) registers callables, C++
/// holds references; on callback C++ does `py::gil_scoped_acquire` then
/// invokes callable. Python callable exceptions are captured and logged,
/// NEVER re-thrown into C++ frame (would corrupt SDK state).
class EngineListener final : public ding::rtc::RtcEngineEventListener {
public:
    EngineListener() = default;
    ~EngineListener() override = default;

    // Called from Python (GIL held) to register / unregister callables.
    void set_on_join(py::object cb);
    void set_on_leave(py::object cb);
    void set_on_bye(py::object cb);
    void set_on_error(py::object cb);
    void set_on_connection_status_change(py::object cb);
    void clear_callbacks();

    // Overrides — called from SDK threads (GIL NOT held by default).
    void OnJoinChannelResult(int result, const char *channel, const char *userId,
                              int elapsed) override;
    void OnLeaveChannelResult(int result, ding::rtc::RtcEngineStats stats) override;
    void OnBye(int code) override;
    void OnOccurError(int error, const char *message) override;
    void OnConnectionStatusChange(ding::rtc::RtcEngineConnectionStatus status,
                                   ding::rtc::RtcEngineConnectionStatusChangeReason reason)
        override;

private:
    std::mutex mu_;
    py::object on_join_;
    py::object on_leave_;
    py::object on_bye_;
    py::object on_error_;
    py::object on_connection_status_change_;
};

}  // namespace idingrtc
