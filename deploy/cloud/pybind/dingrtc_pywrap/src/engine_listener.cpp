#include "engine_listener.h"

#include <pybind11/gil.h>

#include <exception>

namespace idingrtc {

namespace {

void invoke_locked(py::object &cb, py::args args) {
    if (!cb || cb.is_none()) return;
    try {
        cb(*args);
    } catch (const py::error_already_set &) {
        // Don't propagate Python exceptions back into the SDK C++ frame.
        // Log via PyErr_WriteUnraisable; the Python adapter wires up
        // logging before registering callbacks.
        PyErr_WriteUnraisable(cb.ptr());
    } catch (const std::exception &) {
        // Swallow native exceptions for the same reason.
    }
}

}  // namespace

void EngineListener::set_on_join(py::object cb) {
    std::lock_guard<std::mutex> lk(mu_);
    on_join_ = std::move(cb);
}

void EngineListener::set_on_leave(py::object cb) {
    std::lock_guard<std::mutex> lk(mu_);
    on_leave_ = std::move(cb);
}

void EngineListener::set_on_bye(py::object cb) {
    std::lock_guard<std::mutex> lk(mu_);
    on_bye_ = std::move(cb);
}

void EngineListener::set_on_error(py::object cb) {
    std::lock_guard<std::mutex> lk(mu_);
    on_error_ = std::move(cb);
}

void EngineListener::set_on_connection_status_change(py::object cb) {
    std::lock_guard<std::mutex> lk(mu_);
    on_connection_status_change_ = std::move(cb);
}

void EngineListener::clear_callbacks() {
    py::gil_scoped_acquire gil;
    std::lock_guard<std::mutex> lk(mu_);
    on_join_ = py::object();
    on_leave_ = py::object();
    on_bye_ = py::object();
    on_error_ = py::object();
    on_connection_status_change_ = py::object();
}

void EngineListener::OnJoinChannelResult(int result, const char *channel,
                                          const char *userId, int elapsed) {
    py::gil_scoped_acquire gil;
    py::object cb;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb = on_join_;
    }
    invoke_locked(cb,
                  py::make_tuple(result,
                                 channel ? channel : "",
                                 userId ? userId : "",
                                 elapsed));
}

void EngineListener::OnLeaveChannelResult(int result, ding::rtc::RtcEngineStats) {
    py::gil_scoped_acquire gil;
    py::object cb;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb = on_leave_;
    }
    invoke_locked(cb, py::make_tuple(result));
}

void EngineListener::OnBye(ding::rtc::RtcEngineOnByeType code) {
    py::gil_scoped_acquire gil;
    py::object cb;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb = on_bye_;
    }
    invoke_locked(cb, py::make_tuple(static_cast<int>(code)));
}

void EngineListener::OnOccurError(int error, const char *message) {
    py::gil_scoped_acquire gil;
    py::object cb;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb = on_error_;
    }
    invoke_locked(cb, py::make_tuple(error, message ? message : ""));
}

void EngineListener::OnConnectionStatusChanged(
    ding::rtc::RtcEngineConnectionStatus status,
    ding::rtc::RtcEngineConnectionStatusChangeReason reason) {
    py::gil_scoped_acquire gil;
    py::object cb;
    {
        std::lock_guard<std::mutex> lk(mu_);
        cb = on_connection_status_change_;
    }
    invoke_locked(cb,
                  py::make_tuple(static_cast<int>(status),
                                 static_cast<int>(reason)));
}

}  // namespace idingrtc
