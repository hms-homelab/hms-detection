# Shared library used by both yolo_detection and yolo_timeline

# Paho MQTT C++ (async_client wrapper)
find_package(eclipse-paho-mqtt-c REQUIRED)
find_package(PahoMqttCpp REQUIRED)

add_library(yolo_shared STATIC
    config/src/config_manager.cpp
    common/src/time_utils.cpp
    db/src/db_pool.cpp
    db/src/api_queries.cpp
    db/src/event_logger.cpp
    mqtt/src/mqtt_client.cpp
)

target_include_directories(yolo_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/config/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/db/include
    ${CMAKE_CURRENT_SOURCE_DIR}/mqtt/include
    ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(yolo_shared PUBLIC
    spdlog::spdlog
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    PkgConfig::pqxx
    eclipse-paho-mqtt-c::paho-mqtt3as
    PahoMqttCpp::paho-mqttpp3
)

if(BUILD_TESTS)
    include(CTest)
    include(Catch)

    add_executable(shared_tests
        config/tests/config_manager_test.cpp
        common/tests/time_utils_test.cpp
        db/tests/api_queries_test.cpp
    )

    target_link_libraries(shared_tests PRIVATE
        yolo_shared
        Catch2::Catch2WithMain
    )

    catch_discover_tests(shared_tests)
endif()
