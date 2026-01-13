#pragma once

#include "common/client.hpp"
#include "node/CameraNode.h"

#include "model/engine.hpp"
#include "task/followme.h"
#include "task/record.h"


class PerceptionNode : public rclcpp::Node
{
public:
    PerceptionNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("perception_node", options)
    {
        parse_declare_parameters();
        configuration();
    }
    ~PerceptionNode(){
        stop();
    }
public:
    void parse_declare_parameters();
    void configuration();
    void configuration_camera();
    void configuration_client();
    void configuration_task();
    void configuration_work();
public:
    void send_camera_status();
public:
    bool common_preprocess(std::shared_ptr<InferenceData_t> infer_data);
    void followme_thread();
    void record_thread();
    void websocket_publish(std::shared_ptr<InferenceData_t> infer_data, bool is_track=false);
public:
    void start();
    void stop();
public:
    void Display();
public:
    std::string project_root_;
    std::string camera_config_path_, client_config_path_, model_config_path_, laser_config_path_, record_config_path_;
    std::string python_;
    bool record_, followme_, show_, debug_;
public:
    nlohmann::json camera_config_, client_config_, model_config_, laser_config_, record_config_;
public:
    std::shared_ptr<CameraNode> camera_node_;
public:
    std::shared_ptr<PerceptionClient> perception_client_;
public:
    std::shared_ptr<Engine> engine_;
    std::shared_ptr<FollowMe> followme_task_;
    std::shared_ptr<Record> record_task_;

    std::atomic<bool> is_followme_running_{false};
    std::atomic<bool> is_record_running_{false};
public:
    CommonThreadPool publish_thread_pool_;
public:
    std::vector<std::shared_ptr<std::thread>> worker_threads_;
private:
    std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
    std::thread spin_thread_;
};
