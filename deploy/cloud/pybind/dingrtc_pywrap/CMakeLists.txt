# dingrtc_pywrap — project-internal pybind11 binding for the DingRTC Linux
# C++ SDK 3.x. Mirror of isales-telephony's windows-artc-pybind11 binding
# layout, but targets DingRTC (RTC PaaS new-generation; alivc-demo-cms.alicdn
# ARTC SDK is the wrong product line — see openspec change
# engine-rtc-dingrtc-migration).
#
# Built out-of-source via deploy/cloud/scripts/build-dingrtc-binding.sh.
# The script invokes:
#   cmake -S deploy/cloud/pybind/dingrtc_pywrap -B build/dingrtc-binding \
#         -DPython3_EXECUTABLE=$(which python3)
#   cmake --build build/dingrtc-binding --config Release
#   cp build/dingrtc-binding/dingrtc_pywrap*.so .venv/lib/python*/site-packages/
#
# Linux-only. macOS edge走 PyObjC binding (audio_bridge/macos_dingrtc_pyobjc.py);
# Windows edge走 deploy/edge/windows/pybind/dingrtc_pywrap (isales-telephony).

cmake_minimum_required(VERSION 3.20)
project(dingrtc_pywrap LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Linux-only. Cloud engine runs on x86_64 Linux (AL3 / Ubuntu 22.04).
if(NOT UNIX OR APPLE OR WIN32)
    message(FATAL_ERROR
        "dingrtc_pywrap targets Linux only. "
        "macOS dev: use isales-telephony/audio_bridge/macos_dingrtc_pyobjc.py. "
        "Windows edge: use isales-telephony/deploy/edge/windows/pybind/dingrtc_pywrap.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Python — must match the venv the binding will be imported from.
# build-dingrtc-binding.sh exports Python3_EXECUTABLE before invoking cmake.
# ---------------------------------------------------------------------------
find_package(Python3 3.11 COMPONENTS Interpreter Development.Module REQUIRED)

message(STATUS "Python3_EXECUTABLE  = ${Python3_EXECUTABLE}")
message(STATUS "Python3_VERSION     = ${Python3_VERSION}")
message(STATUS "Python3_INCLUDE_DIRS= ${Python3_INCLUDE_DIRS}")

# ---------------------------------------------------------------------------
# pybind11 — fetched via `pip install pybind11` and resolved via Python.
# Unlike the Windows binding (which vendors pybind11 as a submodule), we
# use Python-discovered pybind11 here because cloud Linux deploys always
# have pip available in the venv.
# ---------------------------------------------------------------------------
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE pybind11_FIND_RESULT
)
if(NOT pybind11_FIND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "pybind11 not installed in the active venv. "
        "Run `pip install pybind11` before building.")
endif()
set(pybind11_DIR "${pybind11_DIR_OUT}")
find_package(pybind11 REQUIRED CONFIG)

# ---------------------------------------------------------------------------
# DingRTC SDK (vendored binaries, gitignored).
# Layout:
#   $ISALES_DINGRTC_LINUX_SDK_PATH/
#     api/{engine_interface.h,engine_types.h,engine_conf.h,engine_utils.h,...}
#     lib/x86_64/libDingRTC.so
#     lib/x86_64/libffmpeg.so
# Default vendor path: /opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/
# (matches deploy/cloud/STATE.md § "DingRTC SDK vendor")
# ---------------------------------------------------------------------------
if(NOT DEFINED ENV{ISALES_DINGRTC_LINUX_SDK_PATH})
    set(DINGRTC_SDK_ROOT "/opt/isales/vendor/DingRTC_Linux_SDK_3_9_0")
else()
    set(DINGRTC_SDK_ROOT "$ENV{ISALES_DINGRTC_LINUX_SDK_PATH}")
endif()

set(DINGRTC_INCLUDE_DIR "${DINGRTC_SDK_ROOT}/api")
set(DINGRTC_LIB_DIR     "${DINGRTC_SDK_ROOT}/lib/x86_64")
set(DINGRTC_LIB_FILE    "${DINGRTC_LIB_DIR}/libDingRTC.so")

if(NOT EXISTS "${DINGRTC_INCLUDE_DIR}/engine_interface.h")
    message(FATAL_ERROR
        "DingRTC SDK headers not found at ${DINGRTC_INCLUDE_DIR}/engine_interface.h. "
        "Set ISALES_DINGRTC_LINUX_SDK_PATH to the SDK extraction root, "
        "or place SDK at /opt/isales/vendor/DingRTC_Linux_SDK_3_9_0/. "
        "Download: https://dingrtc.oss-cn-zhangjiakou.aliyuncs.com/sdk/linux/3.9.0/DingRTC_Linux_SDK_3_9_0.zip")
endif()

if(NOT EXISTS "${DINGRTC_LIB_FILE}")
    message(FATAL_ERROR
        "DingRTC SDK shared library not found at ${DINGRTC_LIB_FILE}. "
        "vendor dir may be incomplete — re-extract DingRTC_Linux_SDK_3_9_0.zip.")
endif()

message(STATUS "DINGRTC_SDK_ROOT    = ${DINGRTC_SDK_ROOT}")
message(STATUS "DINGRTC_INCLUDE_DIR = ${DINGRTC_INCLUDE_DIR}")
message(STATUS "DINGRTC_LIB_FILE    = ${DINGRTC_LIB_FILE}")

add_library(DingRTC SHARED IMPORTED)
set_target_properties(DingRTC PROPERTIES
    IMPORTED_LOCATION "${DINGRTC_LIB_FILE}"
    INTERFACE_INCLUDE_DIRECTORIES "${DINGRTC_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# pybind11 module.
# ---------------------------------------------------------------------------
pybind11_add_module(dingrtc_pywrap MODULE
    src/bindings.cpp
    src/audio_observer.cpp
    src/engine_listener.cpp
    src/ring_buffer.cpp
)

target_include_directories(dingrtc_pywrap PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${DINGRTC_INCLUDE_DIR}"
)

target_link_libraries(dingrtc_pywrap PRIVATE DingRTC)

# Set RPATH so the loaded .so finds libDingRTC.so at the vendor location
# without forcing LD_LIBRARY_PATH globally. systemd unit may still set
# LD_LIBRARY_PATH for the engine venv binary, which is fine — RPATH is the
# fallback that works in dev / smoke contexts.
set_target_properties(dingrtc_pywrap PROPERTIES
    INSTALL_RPATH "${DINGRTC_LIB_DIR}"
    BUILD_RPATH   "${DINGRTC_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# GCC / Clang flags. -fvisibility=hidden matches pybind11_add_module default.
target_compile_options(dingrtc_pywrap PRIVATE
    -Wall -Wextra -Wno-unused-parameter
    -fvisibility=hidden
)
