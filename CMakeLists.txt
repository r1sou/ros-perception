cmake_minimum_required(VERSION 3.20)
project(perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS "-pthread -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")

find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(fmt REQUIRED)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(message_filters REQUIRED)

find_package(hobot_cv REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/dnn/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/nlohmann/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/ThreadPool/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/websocketpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/magic_enum/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/eigen3/include
)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/dnn/lib)

include_directories(
    /usr/hobot/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/hobotcv_neon_blur/include
)
link_directories(
    /usr/hobot/lib/                                                   # Nano2D Nano2Dutil vpf cjson
    /usr/aarch64-linux-gnu/lib                                        # dl
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/hobotcv_neon_blur/lib # hobotcv_neon_blur
)

SET(VPS_LIBS cjson dl)
SET(NEON_LIBS  Nano2D Nano2Dutil vpf hobotcv_neon_blur)

# aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src src_files)
# add_executable(${PROJECT_NAME}_node ${src_files})

add_executable(${PROJECT_NAME}_node 
    src/model/bytetrack/BYTETracker.cpp
    src/model/bytetrack/kalmanFilter.cpp
    src/model/bytetrack/lapjv.cpp
    src/model/bytetrack/STrack.cpp
    src/model/bytetrack/utils.cpp
    src/model/stereonet/stereonet.cpp
    src/model/yolo/yolo.cpp
    src/node/CameraNode.cpp
    src/node/node.cpp
    src/main.cpp
)

ament_target_dependencies(${PROJECT_NAME}_node
    rclcpp
    cv_bridge
    std_msgs
    sensor_msgs
    message_filters
    tf2
    tf2_ros
    geometry_msgs
    tf2_sensor_msgs
    hobot_cv
)

target_link_libraries(${PROJECT_NAME}_node
    ${OpenCV_LIBRARIES}
    ${OPENSSL_LIBRARIES}
    pthread
    fmt
    dnn
)

install(TARGETS ${PROJECT_NAME}_node
    DESTINATION lib/${PROJECT_NAME}
)


file(GLOB LAUNCH_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/launch.py
)
install(FILES ${LAUNCH_FILES} DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY assets/config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY assets/weights DESTINATION share/${PROJECT_NAME})
install(DIRECTORY scripts DESTINATION share/${PROJECT_NAME})

ament_package()