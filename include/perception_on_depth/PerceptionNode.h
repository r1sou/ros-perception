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

#include "perception_on_depth/Buffer.h"
#include "perception_on_depth/Client.h"
#include "perception_on_depth/Model.h"

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
        std::string node_name, nlohmann::json camera_config, nlohmann::json laser_config) : Node(node_name), camera_config(camera_config), laser_config(laser_config)
    {
    }
    ~CameraNode(){
        if(pub_thread_.joinable()){
            pub_thread_.join();
        }
    }

public:
    void InitCameraNode(bool pub_laser, bool pub_pc, bool ms)
    {
        {
            buffer_ = std::make_shared<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>>();
            pub_buffer_ = std::make_shared<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>>();
            image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                shared_from_this(), camera_config["topic"]["image_raw"].get<std::string>());
            depth_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                shared_from_this(), camera_config["topic"]["depth_raw"].get<std::string>());
            syncApproximate_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
                SyncPolicy(10), *image_sub_, *depth_sub_);
            syncApproximate_->registerCallback(&CameraNode::ImageCallback, this);
        }
        {
            target_frame_ = laser_config.value("target_frame", "");
            tolerance_ = laser_config.value("tolerance", 0.01);
            input_queue_size_ = this->declare_parameter(
                "queue_size", static_cast<int>(std::thread::hardware_concurrency()));
            min_height_ = laser_config.value("min_height", std::numeric_limits<double>::min());
            max_height_ = laser_config.value("max_height", std::numeric_limits<double>::max());
            angle_min_ = laser_config.value("angle_min", -M_PI);
            angle_max_ = laser_config.value("angle_max", M_PI);
            angle_increment_ = laser_config.value("angle_increment", M_PI / 180.0);
            scan_time_ = laser_config.value("scan_time", 1.0 / 30.0);
            range_min_ = laser_config.value("range_min", 0.0);
            range_max_ = laser_config.value("range_max", std::numeric_limits<double>::max());
            inf_epsilon_ = laser_config.value("inf_epsilon", 1.0);
            use_inf_ = laser_config.value("use_inf", true);

            laserscan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
                camera_config["name"].get<std::string>() + "_laserscan",
                rclcpp::SensorDataQoS());

            int down_sample_ratio = laser_config.value("down_sample_ratio", 1);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "down_sample_ratio: " << down_sample_ratio);
        }
        {
            point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
                camera_config["name"].get<std::string>() + "_point_cloud", 10);
        }
        {
            pub_laser_ = pub_laser;
            pub_pc_ = pub_pc;
            ms_ = ms;
        }
    }
    void InitClient(std::string ip, int port)
    {
        client = std::make_shared<UDPClient>(ip, port);
        thread_pool.init(1,1);
    }
    void ImageCallback(
        const sensor_msgs::msg::Image::SharedPtr &image_msg,
        const sensor_msgs::msg::Image::SharedPtr &depth_msg)
    {
//         thread_pool.enqueue(
//             [this, image_msg, depth_msg]()
//             {
// #ifdef __aarch64__
//                 Depth2LaserScan(image_msg,depth_msg);
// #else
//                 Depth2LaserScan(image_msg,depth_msg);
// #endif
//             });

        buffer_->update(
            [image_msg, depth_msg](sensor_msgs::msg::Image::SharedPtr &image, sensor_msgs::msg::Image::SharedPtr &depth)
            {
                image = image_msg;
                depth = depth_msg;
            });
        pub_buffer_->update(
            [image_msg, depth_msg](sensor_msgs::msg::Image::SharedPtr &image, sensor_msgs::msg::Image::SharedPtr &depth)
            {
                image = image_msg;
                depth = depth_msg;
            });
    }

    void Start(){
        pub_thread_ = std::thread([this]()
        {
            while (rclcpp::ok())
            {
                auto element = pub_buffer_->read();    
                if (!element || !element->image || !element->depth)
                {
                    RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "element is null");
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    continue;
                }
                Depth2LaserScan(element->image, element->depth);
            }
        });
        pub_thread_.detach();
    }

    void Depth2LaserScan(sensor_msgs::msg::Image::SharedPtr image_msg,sensor_msgs::msg::Image::SharedPtr depth_msg)
    {
        static int64_t last_pub_time = 0;

        cv::Mat depth = cv_bridge::toCvShare(depth_msg, "16UC1")->image;
        cv::Mat image = cv_bridge::toCvShare(image_msg, "bgr8")->image;


        int64_t depth_pub_time = image_msg->header.stamp.sec * 1000LL + image_msg->header.stamp.nanosec / 1000000;
        int64_t depth_sub_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        
        // std::string info = fmt::format(
        //     "depth pub time: {:.3f} ms, depth sub time: {:.3f} ms, delay: {} ms",
        //     depth_pub_time / 1000.0, depth_sub_time / 1000.0, depth_sub_time - depth_pub_time
        // );

        // RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());
        // return;

        if (last_pub_time != 0 && depth_pub_time - last_pub_time < 10)
        {
            return;
        }
        last_pub_time = depth_pub_time;

        int down_sample_ratio = laser_config.value("down_sample_ratio", 1);

        auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
        auto pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>(
            rosidl_runtime_cpp::MessageInitialization::SKIP
        );
        sensor_msgs::msg::PointCloud2 &point_cloud_msg = *pc_msg;

        scan_msg->header = depth_msg->header;
        pc_msg->header = depth_msg->header;

        {
            scan_msg->angle_min = angle_min_;
            scan_msg->angle_max = angle_max_;
            scan_msg->angle_increment = angle_increment_;
            scan_msg->time_increment = 0.0;
            scan_msg->scan_time = scan_time_;
            scan_msg->range_min = range_min_;
            scan_msg->range_max = range_max_;
            // determine amount of rays to create
            uint32_t ranges_size = std::ceil(
                (scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);
            // determine if laserscan rays with no obstacle data will evaluate to infinity or max_range
            if (use_inf_)
            {
                scan_msg->ranges.assign(ranges_size, std::numeric_limits<double>::infinity());
            }
            else
            {
                scan_msg->ranges.assign(ranges_size, scan_msg->range_max + inf_epsilon_);
            }
        }
        {
            point_cloud_msg.header = depth_msg->header;
            point_cloud_msg.is_dense = false;
            point_cloud_msg.fields.resize(4);
            point_cloud_msg.fields[0].name = "x";
            point_cloud_msg.fields[0].offset = 0;
            point_cloud_msg.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
            point_cloud_msg.fields[0].count = 1;
            point_cloud_msg.fields[1].name = "y";
            point_cloud_msg.fields[1].offset = 4;
            point_cloud_msg.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
            point_cloud_msg.fields[1].count = 1;
            point_cloud_msg.fields[2].name = "z";
            point_cloud_msg.fields[2].offset = 8;
            point_cloud_msg.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
            point_cloud_msg.fields[2].count = 1;
            point_cloud_msg.fields[3].name = "rgb";
            point_cloud_msg.fields[3].offset = 12;
            point_cloud_msg.fields[3].datatype = sensor_msgs::msg::PointField::UINT32;
            point_cloud_msg.fields[3].count = 1;

            point_cloud_msg.height = 1;
            point_cloud_msg.point_step = 16;

            point_cloud_msg.data.resize((depth.cols / down_sample_ratio) * (depth.rows / down_sample_ratio) * point_cloud_msg.point_step * point_cloud_msg.height);
        }
        uint32_t point_size = 0;
        float *pcd_data_ptr = reinterpret_cast<float *>(point_cloud_msg.data.data());

        double fx = camera_config["calibration"]["fx"].get<double>();
        double fy = camera_config["calibration"]["fy"].get<double>();
        double cx = camera_config["calibration"]["cx"].get<double>();
        double cy = camera_config["calibration"]["cy"].get<double>();

        {
            ScopeProcessTime t("calculate");
            for (int v = 0; v < depth.rows; v += down_sample_ratio)
            {
                for (int u = 0; u < depth.cols; u += down_sample_ratio)
                {
                    double z = static_cast<double>(depth.at<uint16_t>(v, u)) / 1000.0;
                    if (z > 0)
                    {
                        double x = z * (cx - u) / fx;
                        double y = z * (cy - v) / fy;

                        if (std::isnan(x) || std::isnan(y) || std::isnan(z))
                        {
                            continue;
                        }
                        if(pub_pc_ && y < 5.0 && y > -5.0){
                            *pcd_data_ptr++ = z;
                            *pcd_data_ptr++ = x;
                            *pcd_data_ptr++ = y;
                            cv::Vec3b pixel = image.at<cv::Vec3b>(v, u);
                            *(uint32_t *)pcd_data_ptr++ = (pixel[2] << 16) | (pixel[1] << 8) | (pixel[0] << 0);
                            point_size++;
                        }
                        if (y > max_height_ || y < min_height_)
                        {
                            continue;
                        }
                        double range = hypot(x, z);
                        if (range > range_max_ || range < range_min_)
                        {
                            continue;
                        }

                        double angle = std::atan2(x, z);
                        if (angle > angle_max_ || angle < angle_min_)
                        {
                            continue;
                        }
                        int index = (angle - scan_msg->angle_min) / scan_msg->angle_increment;
                        if (range < scan_msg->ranges[index])
                        {
                            scan_msg->ranges[index] = range;
                        }
                    }
                }
            }
        }
        {
            if(pub_pc_){
                point_cloud_msg.width = point_size;
                point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
                point_cloud_msg.data.resize(point_size * point_cloud_msg.point_step *
                                            point_cloud_msg.height);
            }
        }
        int64_t laser_pub_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();
        {
            nlohmann::json message;
            message["cmd_code"] = 0x01;
            message["device_id"] = camera_config["device_id"].get<int>();
            time_t timestamp = time(NULL);
            message["time_stamp"] = timestamp;
            message["data"] = nlohmann::json::object();
            message["data"]["angle_min"] = scan_msg->angle_min;
            message["data"]["angle_max"] = scan_msg->angle_max;
            message["data"]["angle_increment"] = scan_msg->angle_increment;
            message["data"]["time_increment"] = scan_msg->time_increment;
            message["data"]["scan_time"] = scan_msg->scan_time;
            message["data"]["range_min"] = scan_msg->range_min;
            message["data"]["range_max"] = scan_msg->range_max;
            message["data"]["ranges"] = scan_msg->ranges;

            if(ms_){
                message["depth_pub_time"] = depth_pub_time / 1000.0;  // 相机采集到发布深度图的时间戳  
                message["depth_sub_time"] = depth_sub_time / 1000.0;  // 订阅深度图话题的时间戳 
                message["laser_pub_time"] = laser_pub_time / 1000.0;  // 将深度图处理成laserscan到最后发布的时间戳
            }

            client->SendMsg(message.dump());
        }
        {
            if(pub_laser_){
                laserscan_pub_->publish(std::move(scan_msg));
            }
            if(pub_pc_){
                point_cloud_pub_->publish(std::move(pc_msg));
            }
        }
        {
            std::string info = fmt::format(
                "depth pub time: \33[32m{:.3f}\33[0m s, depth sub time: \33[32m{:.3f}\33[0m s, laser pub time: \33[32m{:.3f}\33[0m s, delay: \33[31m{:3d}\33[0m ms",
                depth_pub_time / 1000.0,
                depth_sub_time / 1000.0,
                laser_pub_time / 1000.0, 
                laser_pub_time - depth_pub_time
            );
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());
        }
    }
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>::Element> Read()
    {
        return buffer_->read();
    }

public:
    int input_queue_size_;
    std::string target_frame_;
    double tolerance_;
    double min_height_, max_height_, angle_min_, angle_max_, angle_increment_, scan_time_, range_min_, range_max_;
    bool use_inf_;
    double inf_epsilon_;

    std::shared_ptr<UDPClient> client;
    ThreadPool thread_pool;

private:
    std::vector<double> angle_map_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::LaserScan>> laserscan_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> point_cloud_pub_;

    bool pub_laser_, pub_pc_, ms_;

    std::thread pub_thread_;

public:
    nlohmann::json camera_config;
    nlohmann::json laser_config;

private:
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>> buffer_;
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>> pub_buffer_;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> syncApproximate_;
};

class PerceptionNode : public rclcpp::Node
{
public:
    PerceptionNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions()) : Node("perception_node", node_options)
    {
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "PerceptionNode start");
        parameter_configuration();
        init_camera();

#ifdef __aarch64__
        init_model();
#endif
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

        this->declare_parameter("ms", false);
        this->get_parameter("ms", ms_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "ms level precision: " << ms_);

        this->declare_parameter("show", false);
        this->get_parameter("show", show_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "show: " << show_);

        this->declare_parameter("save", false);
        this->get_parameter("save", save_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "save: " << save_);

        this->declare_parameter("save_dir", "");
        this->get_parameter("save_dir", save_dir_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "save_dir: " << save_dir_);

        this->declare_parameter("save_name", "");
        this->get_parameter("save_name", save_name_);
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "save_name: " << save_name_);
    }

    void init_camera()
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

        for (auto &camera_config : camera_config_["cameras"])
        {
            std::string basename = camera_config["name"].get<std::string>();
            {
                auto camera_node = std::make_shared<CameraNode>(basename + "_camera_node", camera_config, laser_config_["laser-config"]);
                camera_node->InitCameraNode(pub_laser_, pub_pc_, ms_);
                camera_node->InitClient(client_config_["UDP"]["ip"].get<std::string>(), client_config_["UDP"]["port"].get<int>());
                camera_nodes_.push_back(camera_node);
            }
        }
        {
            client = std::make_shared<WebSocketClient>(client_config_["websocket"]);
            std::string uri = fmt::format("ws://{}:{}", client_config_["websocket"]["ip"], client_config_["websocket"]["port"]);
            client->Connect(uri);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), fmt::format("WebSocketClient connect to \33[32m{}\33[0m", uri));

            key_ = JWTGenerator::generate(
                client->m_config["req_id"], client->m_config["key"]
            );
        }
    }

#ifdef __aarch64__
    void init_model()
    {
        if (model_config_path_ != "")
        {
            std::ifstream f(model_config_path_);
            f >> model_config_;
        }
        for (auto &model_config : model_config_["models"])
        {
            if(!model_config["launch"].get<bool>()){
                continue;
            }
            
            std::string model_name = model_config["model"].get<std::string>();
            std::string model_path = root + model_config["path"].get<std::string>();
            if (model_name == "yolo")
            {
                auto yolo_model = std::make_shared<BaseYoloModel>(packed_dnn_handle_, model_path, model_config["param"]);
                yolo_models_.push_back(yolo_model);
            }
        }
        preprocessor_thread_pool_.init(4);
        postprocessor_thread_pool_.init(4);
        publisher_thread_pool_.init(4);
    }
#endif

    void Start()
    {
        executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
        for (auto &camera_node : camera_nodes_)
        {
            executor_->add_node(camera_node);
        }
        spin_thread_ = std::thread([this]()
                                   { executor_->spin(); });
        
        for (auto &camera_node : camera_nodes_)
        {
            camera_node->Start();
        }
    }

    void make_save_directory()
    {
        if (!save_){
            return;
        }
        save_path_ = save_dir_ + "/" + save_name_;
        std::filesystem::create_directories(save_path_);
    }

    void Inference()
    {
#ifdef __aarch64__
        if (!infer_)
        {
            Display();
            return;
        }
        for (int i = 0; i < camera_nodes_.size(); i++)
        {
            InferenceSingleCamera(i);
        }
#else
        Display();
#endif
    }

#ifdef __aarch64__
    void InferenceSingleCamera(int index)
    {
        ScopeProcessTime t(fmt::format("inference camera {}", index));
        
        auto element = camera_nodes_[index]->Read();
        if (!element || !element->image || !element->depth)
        {
            return;
        }
        
        cv::Mat image = cv_bridge::toCvShare(element->image, "bgr8")->image;
        cv::Mat depth = cv_bridge::toCvShare(element->depth, "16UC1")->image;

        if(save_){
            int64_t subscribe_time = element->timestamp.count();
            std::string filename = save_path_ + "/" + fmt::format("{:.3f}.jpg", subscribe_time / 1000.0);
            cv::imwrite(filename, image);
        }

        std::vector<YoloModelOutput> outputs(yolo_models_.size());
        std::vector<std::future<void>> preprocess_task, postprocess_task;
        for (int i = 0; i < yolo_models_.size(); i++)
        {
            preprocess_task.push_back(
                preprocessor_thread_pool_.enqueue(
                    [&, i]()
                    {
                        yolo_models_[i]->Preprocess(image, i);
                    }
                )
            );
        }
        for (int i = 0; i < yolo_models_.size(); i++)
        {
            preprocess_task[i].get();
            yolo_models_[i]->Inference(i);
            postprocess_task.push_back(
                postprocessor_thread_pool_.enqueue(
                    [&, i]()
                    {
                        yolo_models_[i]->Postprocess(i, outputs[i]);
                    }
                )
            );
        }
        for (int i = 0; i < yolo_models_.size(); i++)
        {
            postprocess_task[i].get();
        }

        std::shared_ptr<YoloModelOutput> output = std::make_shared<YoloModelOutput>();
        {
            for (auto &out : outputs)
            {
                if (out.bboxes.size())
                {
                    output->bboxes.insert(
                        output->bboxes.end(),
                        std::make_move_iterator(out.bboxes.begin()),
                        std::make_move_iterator(out.bboxes.end()));
                    output->names.insert(
                        output->names.end(),
                        std::make_move_iterator(out.names.begin()),
                        std::make_move_iterator(out.names.end()));
                    output->scores.insert(
                        output->scores.end(),
                        std::make_move_iterator(out.scores.begin()),
                        std::make_move_iterator(out.scores.end()));
                }
            }
        }
        {
            if (show_){
                ImageRender::DrawBox(image, output->bboxes, output->names);
                cv::imshow(camera_nodes_[index]->camera_config["name"].get<std::string>() + "_rgb", image);
                cv::waitKey(1);
            }
        }
        {
            std::string info = fmt::format("detections {} object", output->bboxes.size());
            t.info = info;
        }
        {
            publisher_thread_pool_.enqueue(
                [&, output, depth, index]()
                {
                    PublishObject(output, depth, index);
                });
        }

        // camera_nodes_[index]->Depth2LaserScan(element->image, element->depth);
    }
    void PublishObject(const std::shared_ptr<YoloModelOutput> &output,const cv::Mat &depth,int index)
    {
        // 目前不考虑旋转
        nlohmann::json &camera_config = camera_nodes_[index]->camera_config;
        nlohmann::json &calib = camera_config["calibration"];

        nlohmann::json message;
        {
            message["cmd_code"] = 0x12;
            message["device_id"] = camera_config["device_id"].get<int>();
            time_t timestamp = time(NULL);

            message["time_stamp"] = timestamp;
            message["key"] = JWTGenerator::generate(client->m_config["req_id"], client->m_config["key"]);
        }
        {
            auto data = nlohmann::json::array();
            for(int i = 0; i < output->bboxes.size(); i++){
                std::string object_name = output->names[i];
                if(model_config_["detect"].contains(object_name)){
                    if(model_config_["detect"][object_name]["depth"] == "center"){
                        int box_center_x = static_cast<int>(output->bboxes[i][0] + (output->bboxes[i][2] - output->bboxes[i][0]) / 2.0f);
                        int box_center_y = static_cast<int>(output->bboxes[i][1] + (output->bboxes[i][3] - output->bboxes[i][1]) / 2.0f);
                        int box_w = static_cast<int>(output->bboxes[i][2] - output->bboxes[i][0]);
                        int box_h = static_cast<int>(output->bboxes[i][3] - output->bboxes[i][1]);

                        float X = static_cast<float>(depth.at<uint16_t>(box_center_y, box_center_x)) / 1000.0;
                        float Y = (calib["cx"].get<float>() - 1.0 * box_center_x) * X / calib["fx"].get<float>();
                        float Z = (calib["cy"].get<float>() - 1.0 * box_center_y) * X / calib["fy"].get<float>();

                        if (X < 0.05)
                        {
                            continue;
                        }

                        float H = 1.0 * box_h * X / calib["fy"].get<float>();
                        float W = 1.0 * box_w * X / calib["fx"].get<float>();

                        data.push_back(
                            {{"name", output->names[i]},
                            {"obj_type", model_config_["detect"][object_name]["obj_type"].get<int>()},
                            {"obj_code", model_config_["detect"][object_name]["obj_code"].get<int>()},
                            {"loc", fmt::format("{:.2f},{:.2f},{:2f}", X, Y, Z)},
                            {"size", fmt::format("{:.2f},{:.2f}", W, H)}}
                        );
                    }
                }
            }
            message["data"] = data;
        }
        if(message["data"].size()){
            client->SendMsg(message.dump());
        }
    }
#endif
    void Display()
    {
        static int64_t last_receive_time = 0;
        for (auto &camera_node : camera_nodes_)
        {
            auto element = camera_node->Read();
            if (!element || !element->image || !element->depth)
            {
                continue;
            }
            int64_t subscribe_time = element->timestamp.count();
            cv::Mat image = cv_bridge::toCvShare(element->image, "bgr8")->image;
            if(save_){
                std::string filename = save_path_ + "/" + fmt::format("{:.3f}.jpg", subscribe_time / 1000.0);
                cv::imwrite(filename, image);
            }
            {
                // int64_t rgb_publist_time = element->image->header.stamp.sec * 1000LL + element->image->header.stamp.nanosec / 1000000;
                // int64_t dep_publist_time = element->depth->header.stamp.sec * 1000LL + element->depth->header.stamp.nanosec / 1000000;
                // int64_t subscribe_time = element->timestamp.count();
                // int64_t receive_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                //                            std::chrono::system_clock::now().time_since_epoch())
                //                            .count();
                // if (last_receive_time == 0 || subscribe_time - last_receive_time == 0)
                // {
                //     last_receive_time = subscribe_time;
                //     continue;
                // }
                // std::string info = fmt::format(
                //     "rgb publish_time: {:.3f}, dep publish_time: {:.3f}, subscribe_time: {:.3f}, receive_time: {:.3f}",
                //     rgb_publist_time / 1000.0, dep_publist_time / 1000.0, subscribe_time / 1000.0, receive_time / 1000.0);
                // RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());
                // last_receive_time = subscribe_time;
            }
            // {
            //     camera_node->Depth2LaserScan(element->image, element->depth);
            // }
            {
                if (show_){
                    cv::imshow(camera_node->camera_config["name"].get<std::string>() + "_image", image);
                    cv::waitKey(1);
                }
            }
        }
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
#ifdef __aarch64__
        hbDNNRelease(packed_dnn_handle_);
#endif
    }

public:
    std::string root;
    bool save_;
    std::string save_dir_;
    std::string save_name_;
    std::string save_path_;

private:
    std::string camera_config_path_;
    nlohmann::json camera_config_;

    std::string laser_config_path_;
    nlohmann::json laser_config_;

    std::string client_config_path_;
    nlohmann::json client_config_;

    std::string model_config_path_;
    nlohmann::json model_config_;

    bool infer_,pub_laser_,pub_pc_,ms_, show_;

    std::string key_;

public:
    std::shared_ptr<WebSocketClient> client;

private:
    ThreadPool preprocessor_thread_pool_, postprocessor_thread_pool_;
    ThreadPool publisher_thread_pool_;
private:
    std::vector<std::shared_ptr<CameraNode>> camera_nodes_;
#ifdef __aarch64__
    std::vector<std::shared_ptr<BaseYoloModel>> yolo_models_;
    hbPackedDNNHandle_t packed_dnn_handle_;
#endif

private:
    std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
    std::thread spin_thread_;
};
