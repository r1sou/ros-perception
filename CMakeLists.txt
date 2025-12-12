cmake_minimum_required(VERSION 3.20)
project(perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# # set(CMAKE_BUILD_TYPE "Debug")

# set(ARCH "aarch")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS "-pthread -O3 -fsee -fomit-frame-pointer -fno-signed-zeros -fno-math-errno -funroll-loops")
    message("Detect arch -> aarch64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "x86_64")
	message("Detect arch -> x86")
else()
    message("Unknow arch")
endif()

find_package(OpenCV REQUIRED)

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

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/fmt/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/ThreadPool/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/nlohmann/include
    ${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/websocketpp/include
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/dnn/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/assets/3rdparty/dnn/lib)


function(add_project_executable target_name)
    add_executable(${target_name} ${ARGN})
    ament_target_dependencies(${target_name}
        rclcpp
        cv_bridge
        std_msgs
        sensor_msgs
        message_filters
        tf2
        tf2_ros
        geometry_msgs
        tf2_sensor_msgs
    )

    target_link_libraries(${target_name}
        ${OpenCV_LIBRARIES}
        ${OPENSSL_LIBRARIES}
        pthread
        fmt
    )

    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
        target_link_libraries(${target_name}
            dnn
        )
        message("link dnn lib")
    endif()

    install(TARGETS ${target_name}
        DESTINATION lib/${PROJECT_NAME}
    )
endfunction()

add_project_executable(inference
    src/imgproc.cpp
    src/main.cpp
)

file(GLOB LAUNCH_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/launch/*launch.py
)
install(FILES ${LAUNCH_FILES} DESTINATION share/${PROJECT_NAME}/launch)
install(DIRECTORY assets/config DESTINATION share/${PROJECT_NAME})
install(DIRECTORY assets/weights DESTINATION share/${PROJECT_NAME})

ament_package()
