# hms-detection service: RTSP capture + ring buffers + YOLO inference + HTTP API

find_package(PkgConfig REQUIRED)
pkg_check_modules(avformat REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(avcodec  REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(avutil   REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(swscale  REQUIRED IMPORTED_TARGET libswscale)
pkg_check_modules(libcurl  REQUIRED IMPORTED_TARGET libcurl)

# ONNX Runtime — use find_library since Debian doesn't ship pkg-config for it
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include/onnxruntime /usr/local/include/onnxruntime)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Install libonnxruntime-dev.")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")

add_executable(hms_detection
    src/main.cpp
    src/rtsp_capture.cpp
    src/buffer_service.cpp
    src/detection_engine.cpp
    src/detection_worker.cpp
    src/event_recorder.cpp
    src/snapshot_writer.cpp
    src/event_manager.cpp
    src/vision_client.cpp
    src/controllers/health_controller.cpp
    src/controllers/detection_controller.cpp
)

target_include_directories(hms_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/generated
    ${ONNXRUNTIME_INCLUDE_DIR}/..
)

target_link_libraries(hms_detection PRIVATE
    hms_shared
    Drogon::Drogon
    PkgConfig::avformat
    PkgConfig::avcodec
    PkgConfig::avutil
    PkgConfig::swscale
    PkgConfig::libcurl
    ${ONNXRUNTIME_LIBRARY}
)

if(BUILD_TESTS)
    add_executable(detection_tests
        tests/frame_pool_test.cpp
        tests/camera_buffer_test.cpp
        tests/buffer_service_test.cpp
        tests/detection_engine_test.cpp
        tests/vision_client_test.cpp
        tests/event_manager_test.cpp
        src/rtsp_capture.cpp
        src/buffer_service.cpp
        src/detection_engine.cpp
        src/detection_worker.cpp
        src/event_recorder.cpp
        src/snapshot_writer.cpp
        src/event_manager.cpp
        src/vision_client.cpp
    )

    target_include_directories(detection_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated
        ${ONNXRUNTIME_INCLUDE_DIR}/..
    )

    target_link_libraries(detection_tests PRIVATE
        hms_shared
        Catch2::Catch2WithMain
        Drogon::Drogon
        PkgConfig::avformat
        PkgConfig::avcodec
        PkgConfig::avutil
        PkgConfig::swscale
        PkgConfig::libcurl
        ${ONNXRUNTIME_LIBRARY}
    )

    include(Catch)
    catch_discover_tests(detection_tests)
endif()
