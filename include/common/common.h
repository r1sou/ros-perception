#pragma once

#include <iostream>
#include <stdint.h>

#include <vector>
#include <map>
#include <tuple>
#include <utility>
#include <set>
#include <variant>
#include <unordered_set>
#include <string>

#include <algorithm>

#include <iomanip>
#include <fstream>
// #include <experimental/filesystem>
// namespace fs = std::experimental::filesystem;

#include <time.h>
#include <chrono>
#include <cstddef>

#include <unistd.h>
#include <sched.h>
#include <sys/syscall.h>
#include <sys/stat.h>

#include <memory>
#include <atomic>
#include <thread>
#include <pthread.h>
#include <mutex>
#include <future>

#include <functional>

#include "fmt/format.h"

#include "nlohmann/json.hpp"

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

#include "rclcpp/rclcpp.hpp"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Cholesky>

#include "magic_enum/magic_enum.hpp"

class ScopeTimer{
public:
    ScopeTimer(std::string name): name(name){
        start = std::chrono::system_clock::now();
    }
    ~ScopeTimer(){
        auto end = std::chrono::system_clock::now();
        const std::chrono::duration<float, std::milli> duration = end - start;
        std::string info = fmt::format("\033[32mScope {} end, duration: {:.3f}ms\033[0m", name, duration.count());
        RCLCPP_INFO_STREAM(rclcpp::get_logger("ScopeTimer"), info.c_str());
    }
public:
    std::string name;
    std::chrono::system_clock::time_point start;
};
