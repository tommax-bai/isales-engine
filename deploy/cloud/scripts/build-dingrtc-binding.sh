#!/usr/bin/env bash
# build-dingrtc-binding.sh — build dingrtc_pywrap pybind11 extension.
#
# Mirror of isales-telephony's deploy/edge/windows/build.ps1 for the
# DingRTC Linux C++ binding (engine-rtc-dingrtc-migration § 2).
#
# Usage (from isales-engine repo root):
#   source .venv/bin/activate     # or use $VIRTUAL_ENV detection below
#   pip install pybind11           # one-time per venv
#   ./deploy/cloud/scripts/build-dingrtc-binding.sh
#
# Optional env:
#   ISALES_DINGRTC_LINUX_SDK_PATH  # default /opt/isales/vendor/DingRTC_Linux_SDK_3_9_0
#   BUILD_DIR                       # default build/dingrtc-binding (rel to repo root)

set -euo pipefail

# --- Repo root detection ----------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

BINDING_SRC_DIR="${REPO_ROOT}/deploy/cloud/pybind/dingrtc_pywrap"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build/dingrtc-binding}"

# --- Python / venv ---------------------------------------------------------
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi

if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "ERROR: no Python interpreter found. Activate a venv first." >&2
    exit 1
fi

echo "Using Python: ${PYTHON}"
"${PYTHON}" -c "import pybind11" 2>/dev/null || {
    echo "ERROR: pybind11 not installed in this venv. Run: pip install pybind11" >&2
    exit 1
}

# --- DingRTC SDK ground-truth -----------------------------------------------
DINGRTC_PATH="${ISALES_DINGRTC_LINUX_SDK_PATH:-/opt/isales/vendor/DingRTC_Linux_SDK_3_9_0}"
if [[ ! -f "${DINGRTC_PATH}/api/engine_interface.h" ]]; then
    echo "ERROR: DingRTC Linux SDK not found at ${DINGRTC_PATH}/api/engine_interface.h" >&2
    echo "       Download: https://dingrtc.oss-cn-zhangjiakou.aliyuncs.com/sdk/linux/3.9.0/DingRTC_Linux_SDK_3_9_0.zip" >&2
    echo "       Or set ISALES_DINGRTC_LINUX_SDK_PATH to the SDK extraction root." >&2
    exit 1
fi

# --- CMake configure + build ------------------------------------------------
echo "Configuring CMake (source=${BINDING_SRC_DIR}, build=${BUILD_DIR})..."
cmake -S "${BINDING_SRC_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="${PYTHON}" \
    -DISALES_DINGRTC_LINUX_SDK_PATH="${DINGRTC_PATH}"

# Build
echo "Building..."
cmake --build "${BUILD_DIR}" --config Release -j"$(nproc)"

# --- Install into venv site-packages ----------------------------------------
SO_FILE="$(find "${BUILD_DIR}" -maxdepth 2 -name 'dingrtc_pywrap*.so' | head -1)"
if [[ -z "${SO_FILE}" ]]; then
    echo "ERROR: build succeeded but dingrtc_pywrap*.so not found in ${BUILD_DIR}" >&2
    exit 2
fi

SITE_PACKAGES="$("${PYTHON}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "Installing ${SO_FILE} -> ${SITE_PACKAGES}/"
cp -f "${SO_FILE}" "${SITE_PACKAGES}/"

# --- Smoke test -------------------------------------------------------------
echo "Verifying import..."
"${PYTHON}" - <<'PY'
import dingrtc_pywrap
print(f"  module file: {dingrtc_pywrap.__file__}")
print(f"  version    : {dingrtc_pywrap.__version__}")
print(f"  classes    : {[n for n in dir(dingrtc_pywrap) if not n.startswith('_')]}")
PY

echo "OK — dingrtc_pywrap built and installed."
