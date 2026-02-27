#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="/opt/hms-detection"
BINARY="hms_detection"

# Check binary exists
if [ ! -f "${BUILD_DIR}/services/detection/${BINARY}" ]; then
    echo "ERROR: Binary not found at ${BUILD_DIR}/services/detection/${BINARY}"
    echo "Run build first: cd ${SCRIPT_DIR}/yolo_detection_cpp && cmake --preset system-deps && cmake --build ${BUILD_DIR}"
    exit 1
fi

# Create install dir (needs sudo only on first run)
if [ ! -d "${INSTALL_DIR}" ]; then
    echo "Creating ${INSTALL_DIR} (requires sudo)..."
    sudo mkdir -p "${INSTALL_DIR}"
    sudo chown aamat:aamat "${INSTALL_DIR}"
fi

# Copy binary
cp "${BUILD_DIR}/services/detection/${BINARY}" "${INSTALL_DIR}/${BINARY}"
echo "Installed ${BINARY} → ${INSTALL_DIR}/${BINARY}"

# Copy config if not already present (don't overwrite running config)
if [ ! -f "${INSTALL_DIR}/config.yaml" ]; then
    cp "${SCRIPT_DIR}/config.yaml" "${INSTALL_DIR}/config.yaml"
    echo "Installed config.yaml → ${INSTALL_DIR}/config.yaml"
else
    echo "Config exists at ${INSTALL_DIR}/config.yaml (not overwritten)"
fi

# Copy ONNX model if not already present
if [ ! -f "${INSTALL_DIR}/yolo11s.onnx" ]; then
    if [ -f "${SCRIPT_DIR}/yolo11s.onnx" ]; then
        cp "${SCRIPT_DIR}/yolo11s.onnx" "${INSTALL_DIR}/yolo11s.onnx"
        echo "Installed yolo11s.onnx → ${INSTALL_DIR}/yolo11s.onnx"
    fi
else
    echo "Model exists at ${INSTALL_DIR}/yolo11s.onnx (not overwritten)"
fi

# Create logs dir
mkdir -p "${INSTALL_DIR}/logs"

echo ""
echo "Install complete. Restart service:"
echo "  sudo systemctl restart hms-detection"
