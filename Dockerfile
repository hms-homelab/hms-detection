# Multi-stage Dockerfile for hms-detection
# RTSP capture + ring buffers + /health + /snapshot
# Build context: repo root (docker build -f Dockerfile .)

# ── Stage 1: Build C++ hms-detection service ────────────────────────────────
FROM debian:trixie-slim AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git pkg-config \
    libdrogon-dev libtrantor-dev \
    libpqxx-dev libpq-dev \
    libyaml-cpp-dev libjsoncpp-dev \
    libspdlog-dev libfmt-dev \
    nlohmann-json3-dev \
    uuid-dev libbrotli-dev libsqlite3-dev \
    libhiredis-dev default-libmysqlclient-dev \
    libssl-dev libkrb5-dev \
    libavformat-dev libavcodec-dev libavutil-dev libswscale-dev \
    libcurl4-openssl-dev \
    libonnxruntime-dev \
    libpaho-mqttpp-dev libpaho-mqtt-dev \
    libcatch2-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY yolo_detection_cpp/ ./yolo_detection_cpp/

RUN cmake -S yolo_detection_cpp -B yolo_detection_cpp/build \
        -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTS=OFF \
    && cmake --build yolo_detection_cpp/build --target hms_detection

# ── Stage 2: Runtime image ──────────────────────────────────────────────────
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libdrogon1t64 libtrantor1 \
    libpqxx-7.10 libpq5 \
    libyaml-cpp0.8 libjsoncpp26 \
    libspdlog1.15 libfmt10 \
    libuuid1 libbrotli1 libsqlite3-0 libhiredis1.1.0 libmariadb3 \
    libssl3 libkrb5-3 \
    libavformat61 libavcodec61 libavutil59 libswscale8 \
    libcurl4t64 \
    libonnxruntime1.21 \
    libpaho-mqttpp3-1 libpaho-mqtt1.3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build \
    /src/yolo_detection_cpp/build/services/detection/hms_detection \
    /usr/local/bin/hms_detection

RUN chmod +x /usr/local/bin/hms_detection \
    && mkdir -p /app/config /app/logs /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/hms_detection"]
CMD ["--config", "/app/config/config.yaml"]
