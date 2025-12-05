#pragma once

#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_sensor_msgs/tf2_sensor_msgs.hpp"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "sensor_msgs/msg/point_field.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "perception_on_stereo/Buffer.h"
#include "perception_on_stereo/Client.h"
#include "perception_on_stereo/Model.h"

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

struct ScopeProcessTime
{
public:
    ScopeProcessTime(const std::string &name) : name_(name)
    {
        begin_ = std::chrono::system_clock::now();
    }
    ~ScopeProcessTime()
    {
        auto end = std::chrono::system_clock::now();
        const std::chrono::duration<float, std::milli> d = end - begin_;
        if (info == "")
        {
            std::string info_ = fmt::format("\033[32moperate: {}, consume: {:.3f}ms\033[0m", name_, d.count());
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info_.c_str());
        }
        else
        {
            std::string info_ = fmt::format("\033[32moperate: {}, {}, consume: {:.3f}ms\033[0m", name_, info, d.count());
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info_.c_str());
        }
    }

public:
    std::string info;

private:
    std::string name_;
    std::chrono::system_clock::time_point begin_;
};

class CameraNode : public rclcpp::Node
{
public:
    CameraNode(
        std::string node_name, nlohmann::json camera_config) : Node(node_name), camera_config(camera_config)
    {
    }
    ~CameraNode() = default;
public:
    void configuration()
    {
        buffer_ = std::make_shared<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>>();

        // left_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        //     shared_from_this(), camera_config["topic"]["left_raw"].get<std::string>());
        // right_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
        //     shared_from_this(), camera_config["topic"]["right_raw"].get<std::string>());
        // syncApproximate_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
        //     SyncPolicy(10), *left_sub_, *right_sub_);
        // syncApproximate_->registerCallback(&CameraNode::ImageCallback, this);

        combine_sub_ = create_subscription<sensor_msgs::msg::Image>(
            camera_config["topic"]["combine_raw"].get<std::string>(), 10,
            [this](sensor_msgs::msg::Image::SharedPtr msg)
            {
                ImageCallback(msg);
            }
        );
    }
    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr &combine_image)
    {
        buffer_->update(
            [combine_image](sensor_msgs::msg::Image::SharedPtr &image)
            {
                image = combine_image;
            }
        );
    }
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>::Element> Read()
    {
        return buffer_->read();
    }
public:
    nlohmann::json camera_config;
private:
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>> buffer_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr combine_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> syncApproximate_;
};

class PerceptionNode : public rclcpp::Node
{
    public:
        PerceptionNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions()) : Node("perception_node", node_options)
        {
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "PerceptionNode start");
            parameter_configuration();
            configuration();
        }
        ~PerceptionNode()
        {
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "PerceptionNode shutdown");
            Stop();
        }

    public:
        void parameter_configuration()
        {
            this->declare_parameter("root", "");
            this->get_parameter("root", root);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "root: " << root);

            this->declare_parameter("camera_config_path", "");
            this->get_parameter("camera_config_path", camera_config_path_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "camera_config_path: " << camera_config_path_);

            this->declare_parameter("laser_config_path", "");
            this->get_parameter("laser_config_path", laser_config_path_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "laser_config_path: " << laser_config_path_);

            this->declare_parameter("client_config_path", "");
            this->get_parameter("client_config_path", client_config_path_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "client_config_path: " << client_config_path_);

            this->declare_parameter("model_config_path", "");
            this->get_parameter("model_config_path", model_config_path_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "model_config_path: " << model_config_path_);

            this->declare_parameter("infer", true);
            this->get_parameter("infer", infer_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "start model infer: " << infer_);

            this->declare_parameter("pub_laser", true);
            this->get_parameter("pub_laser", pub_laser_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "publish laser scan: " << pub_laser_);

            this->declare_parameter("pub_pc", true);
            this->get_parameter("pub_pc", pub_pc_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "publish point cloud: " << pub_pc_);

            this->declare_parameter("show", false);
            this->get_parameter("show", show_);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "show: " << show_);
        }

        void configuration()
        {
            if (camera_config_path_ != "")
            {
                std::ifstream f(camera_config_path_);
                f >> camera_config_;
            }
            if (laser_config_path_ != "")
            {
                std::ifstream f(laser_config_path_);
                f >> laser_config_;
            }
            if (client_config_path_ != "")
            {
                std::ifstream f(client_config_path_);
                f >> client_config_;
            }
            if (model_config_path_ != "")
            {
                std::ifstream f(model_config_path_);
                f >> model_config_;
            }
            {
                for(auto &camera_config : camera_config_["cameras"]){
                    std::string basename = camera_config["name"].get<std::string>();
                    auto camera_node = std::make_shared<CameraNode>(basename + "_camera_node", camera_config);
                    camera_node->configuration();
                    camera_nodes_.push_back(camera_node);
                }
            }
            {
                auto udp_config = client_config_["UDP"];
                udp_client = std::make_shared<UDPClient>(udp_config["ip"], udp_config["port"].get<int>());
                
                auto websocket_config = client_config_["websocket"];
                websocket_client = std::make_shared<WebSocketClient>(websocket_config);
                std::string uri = fmt::format("ws://{}:{}", websocket_config["ip"], websocket_config["port"]);
                websocket_client->Connect(uri);
                RCLCPP_INFO_STREAM(rclcpp::get_logger(""), fmt::format("WebSocketClient connect to \33[32m{}\33[0m", uri));

                key_ = JWTGenerator::generate(
                    websocket_client->m_config["req_id"], websocket_client->m_config["key"]
                );
            }
        }
        void Start()
        {
            executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
            for (auto &camera_node : camera_nodes_)
            {
                executor_->add_node(camera_node);
            }
            spin_thread_ = std::thread(
                [this](){
                    executor_->spin(); 
                }
            );
        }
        void Stop()
        {
            if (executor_)
            {
                executor_->cancel();
            }
            if (spin_thread_.joinable())
            {
                spin_thread_.join();
            }
        }
    public:
        void Run()
        {
            static std::vector<int64_t> last_pub_time_list(camera_nodes_.size(), 0);
            for(int i = 0; i < camera_nodes_.size(); i++)
            {
                auto element = camera_nodes_[i]->Read();
                if (!element || !element->image)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "element is null");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                int64_t img_pub_time = element->image->header.stamp.sec * 1000LL + element->image->header.stamp.nanosec / 1000000;
                int64_t img_sub_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                int64_t img_store_time = element->timestamp.count();

                if (last_pub_time_list[i] != 0 && img_pub_time - last_pub_time_list[i] < 10)
                {
                    return;
                }
                last_pub_time_list[i] = img_pub_time;
                
                std::string info = fmt::format("camera: {}, pub time: {:.3f}, store time: {:.3f}, sub time: {:.3f}",
                    camera_nodes_[i]->camera_config["name"], img_pub_time / 1000.0, img_store_time / 1000.0, img_sub_time / 1000.0);
                RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());

                if(show_){
                    cv::imshow("left", cv_bridge::toCvShare(element->image, "bgr8")->image);
                    cv::waitKey(1);
                }
            }
        }
    public:
        std::shared_ptr<UDPClient> udp_client;
        std::shared_ptr<WebSocketClient> websocket_client;

    private:
        std::string root;
        std::string camera_config_path_, laser_config_path_, client_config_path_, model_config_path_;
        nlohmann::json camera_config_, laser_config_, client_config_, model_config_;
        bool infer_, pub_laser_, pub_pc_, show_;

        std::string key_;

    private:
        std::vector<std::shared_ptr<CameraNode>> camera_nodes_;

    private:
        std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
        std::thread spin_thread_;
};

