# hms-detection service: RTSP capture + ring buffers + /health + /snapshot

find_package(PkgConfig REQUIRED)
pkg_check_modules(avformat REQUIRED IMPORTED_TARGET libavformat)
pkg_check_modules(avcodec  REQUIRED IMPORTED_TARGET libavcodec)
pkg_check_modules(avutil   REQUIRED IMPORTED_TARGET libavutil)
pkg_check_modules(swscale  REQUIRED IMPORTED_TARGET libswscale)

add_executable(hms_detection
    src/main.cpp
    src/rtsp_capture.cpp
    src/buffer_service.cpp
    src/controllers/health_controller.cpp
)

target_include_directories(hms_detection PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(hms_detection PRIVATE
    yolo_shared
    Drogon::Drogon
    PkgConfig::avformat
    PkgConfig::avcodec
    PkgConfig::avutil
    PkgConfig::swscale
)

if(BUILD_TESTS)
    add_executable(detection_tests
        tests/frame_pool_test.cpp
        tests/camera_buffer_test.cpp
        tests/buffer_service_test.cpp
        src/rtsp_capture.cpp
        src/buffer_service.cpp
    )

    target_include_directories(detection_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    target_link_libraries(detection_tests PRIVATE
        yolo_shared
        Catch2::Catch2WithMain
        PkgConfig::avformat
        PkgConfig::avcodec
        PkgConfig::avutil
        PkgConfig::swscale
    )

    include(Catch)
    catch_discover_tests(detection_tests)
endif()
