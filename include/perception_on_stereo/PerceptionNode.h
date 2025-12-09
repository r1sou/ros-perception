#pragma once

#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"

#include "tf2_ros/buffer.h"
#include "tf2_ros/message_filter.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/static_transform_broadcaster.h"
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

        laserscan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
            camera_config["name"].get<std::string>() + "_laserscan",rclcpp::SensorDataQoS()
        );

        point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            camera_config["name"].get<std::string>() + "_point_cloud", 10
        );

        frame_id = "camera_link";

        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->now();
        t.header.frame_id = "camera_link";
        t.child_frame_id = "camera_link_child";

        t.transform.translation.x = 0.0;
        t.transform.translation.y = 0.0;
        t.transform.translation.z = 0.0;

        static_broadcaster_->sendTransform(t);

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
    std::string frame_id;
    nlohmann::json camera_config;
private:
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>> buffer_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr combine_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> left_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> right_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> syncApproximate_;
public:
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::LaserScan>> laserscan_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> point_cloud_pub_;

    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
};

class PerceptionNode : public rclcpp::Node
{
    public:
        PerceptionNode(const rclcpp::NodeOptions &node_options = rclcpp::NodeOptions()) : Node("perception_node", node_options)
        {
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "PerceptionNode start");
            parameter_configuration();
            configuration();
            {
                preprocess_pool_.init(4);
                postprocess_pool_.init(4);
                udp_pool_.init(4);
                web_pool_.init(4);
            }
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
                RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "camera init finish");
            }
            {
                auto udp_config = client_config_["UDP"];
                udp_client = std::make_shared<UDPClient>(udp_config["ip"].get<std::string>(), udp_config["port"].get<int>());
                RCLCPP_INFO_STREAM(rclcpp::get_logger(""), fmt::format("UDPClient connect to \33[32m{}:{}\33[0m", udp_config["ip"], udp_config["port"].get<int>()));
                
                auto websocket_config = client_config_["websocket"];
                websocket_client = std::make_shared<WebSocketClient>(websocket_config);
                std::string uri = fmt::format("ws://{}:{}", websocket_config["ip"], websocket_config["port"]);
                websocket_client->Connect(uri);
                RCLCPP_INFO_STREAM(rclcpp::get_logger(""), fmt::format("WebSocketClient connect to \33[32m{}\33[0m", uri));

                // key_ = JWTGenerator::generate(websocket_client->m_config["req_id"], websocket_client->m_config["key"]);
            }
            {
                for(auto &model_config : model_config_["models"]){
                    std::string model_name = model_config["model"].get<std::string>();
                    std::string model_path = root + model_config["path"].get<std::string>();
                    if(!model_config["launch"].get<bool>()){
                        continue;
                    }

                    if (model_name == "yolo"){
                        auto model = std::make_shared<YOLO>(model_name);
                        model->configuration(packed_dnn_handle_, model_path, model_config["param"]);
                        models_.push_back(model);
                    }
                    else if (model_name == "stereonet"){
                        auto model = std::make_shared<StereoNet>(model_name);
                        model->configuration(packed_dnn_handle_, model_path, model_config["param"]);
                        models_.push_back(model);
                    }
                    std::string info = fmt::format("\33[32mload model: {}\33[0m", model_path);
                    RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());
                }
            }
            {
                auto laser_config = laser_config_["laser-config"];
                target_frame_ = laser_config.value("target_frame", "");
                tolerance_ = laser_config.value("tolerance", 0.01);
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
            hbDNNRelease(packed_dnn_handle_);
        }
    public:
        void Run(){
            if(infer_){
                Inference();
            }
            else{
                Display();
            }
        }
        void Display()
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
                    cv::Mat combine = cv_bridge::toCvShare(element->image, "bgr8")->image;
                    cv::Mat left = combine(cv::Rect(0, 0, combine.cols, combine.rows / 2));
                    cv::Mat right = combine(cv::Rect(0, combine.rows / 2, combine.cols, combine.rows / 2));
                    cv::imshow("left", left);
                    cv::imshow("right", right);
                    cv::waitKey(1);
                }
            }
        }
        void Inference(){
            static std::vector<int64_t> last_pub_time_list(camera_nodes_.size(), 0);

            for(int i = 0; i < camera_nodes_.size(); i++){
                auto &camera_node = camera_nodes_[i];
                auto element = camera_node->Read();
                if (!element || !element->image){
                    continue;
                }

                int64_t img_pub_time = element->image->header.stamp.sec * 1000LL + element->image->header.stamp.nanosec / 1000000;
                int64_t img_store_time = element->timestamp.count();
                int64_t img_sub_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                if (last_pub_time_list[i] != 0 && img_pub_time - last_pub_time_list[i] < 10)
                {
                    return;
                }
                last_pub_time_list[i] = img_pub_time;

                auto output = std::make_shared<ModelOutput>();
                Inference(element, output);    

                int64_t infer_end_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                
                std::string info = fmt::format(
                    "camera: {}, pub time: {:.3f}, store time: {:.3f}, sub time: {:.3f}, infer time: {:.3f}",
                    camera_node->camera_config["name"], 
                    img_pub_time / 1000.0, 
                    img_store_time / 1000.0, 
                    img_sub_time / 1000.0, 
                    infer_end_time / 1000.0
                );
                // RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());
                output->info = info;
                udp_pool_.enqueue(
                    [this,&output,i](){
                        PublishLaser(output, i);
                    }
                );
                {
                    cv::Mat combine = cv_bridge::toCvShare(element->image, "bgr8")->image;
                    cv::Mat left = combine(cv::Rect(0, 0, combine.cols, combine.rows / 2));
                    ImageRender::DrawBox(left, output->bboxes, output->names);
                    cv::imshow("left", left);
                    cv::waitKey(1);
                }
            }
        }
        // 暂时不考虑极致性能，目前正常推理
        void Inference(std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>::Element> element, std::shared_ptr<ModelOutput> output){
            cv::Mat combine = cv_bridge::toCvShare(element->image, "bgr8")->image;
            cv::Mat left = combine(cv::Rect(0, 0, combine.cols, combine.rows / 2));
            cv::Mat right = combine(cv::Rect(0, combine.rows / 2, combine.cols, combine.rows / 2));

            std::vector<std::future<void>> preprocess_tasks, postprocess_tasks;
            std::vector<ModelOutput> outputs(models_.size());

            for(int i = 0; i < models_.size(); i++){
                preprocess_tasks.push_back(
                    preprocess_pool_.enqueue(
                        [this,&left,&right,i](){
                            std::vector<cv::Mat> inputs = {left, right};
                            models_[i]->preprocess(inputs);
                        }   
                    )
                );
            }
            for(int i = 0; i < models_.size(); i++){
                preprocess_tasks[i].get();
                models_[i]->inference();
                postprocess_tasks.push_back(
                    postprocess_pool_.enqueue(
                        [this,&outputs,i](){
                            models_[i]->postprocess(outputs[i]);
                        }   
                    )
                );
            }
            for(int i = 0; i < models_.size(); i++){
                postprocess_tasks[i].get();
            }
            for(int i = 0; i < models_.size(); i++){
                output->bboxes.insert(output->bboxes.end(), outputs[i].bboxes.begin(), outputs[i].bboxes.end());
                output->scores.insert(output->scores.end(), outputs[i].scores.begin(), outputs[i].scores.end());
                output->names.insert(output->names.end(), outputs[i].names.begin(), outputs[i].names.end());
                if(outputs[i].disparity.size() > 0){
                    output->disparity = std::move(outputs[i].disparity);
                }
            }
            output->rgb = left;
        }

        void PublishLaser(std::shared_ptr<ModelOutput> output, int index){
            if(!pub_laser_ && !pub_pc_){
                RCLCPP_INFO_STREAM(rclcpp::get_logger(""), output->info.c_str());
                return;
            }

            int down_sample_ratio = laser_config_["laser-config"].value("down_sample_ratio", 1);

            auto scan_msg = std::make_unique<sensor_msgs::msg::LaserScan>();
            auto pc_msg = std::make_unique<sensor_msgs::msg::PointCloud2>(
                rosidl_runtime_cpp::MessageInitialization::SKIP
            );
            sensor_msgs::msg::PointCloud2 &point_cloud_msg = *pc_msg;

            scan_msg->header.frame_id = camera_nodes_[index]->frame_id;
            scan_msg->header.stamp = this->get_clock()->now();
            {
                scan_msg->angle_min = angle_min_;
                scan_msg->angle_max = angle_max_;
                scan_msg->angle_increment = angle_increment_;
                scan_msg->time_increment = 0.0;
                scan_msg->scan_time = scan_time_;
                scan_msg->range_min = range_min_;
                scan_msg->range_max = range_max_;
                uint32_t ranges_size = std::ceil(
                    (scan_msg->angle_max - scan_msg->angle_min) / scan_msg->angle_increment);
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
                point_cloud_msg.header = scan_msg->header;
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

                point_cloud_msg.data.resize((output->rgb.cols / down_sample_ratio) * (output->rgb.rows / down_sample_ratio) * point_cloud_msg.point_step * point_cloud_msg.height);
            }
            uint32_t point_size = 0;
            float *pcd_data_ptr = reinterpret_cast<float *>(point_cloud_msg.data.data());

            float fx = camera_nodes_[index]->camera_config["calibration"]["fx"].get<float>();
            float fy = camera_nodes_[index]->camera_config["calibration"]["fy"].get<float>();
            float cx = camera_nodes_[index]->camera_config["calibration"]["cx"].get<float>();
            float cy = camera_nodes_[index]->camera_config["calibration"]["cy"].get<float>();
            float baseline = camera_nodes_[index]->camera_config["calibration"]["baseline"].get<float>();

            {
                int H = output->rgb.rows, W = output->rgb.cols;
                ScopeProcessTime t("calculate");
                for (int v = 0; v < H; v += down_sample_ratio)
                {
                    for (int u = 0; u < W; u += down_sample_ratio)
                    {;
                        float z = fx * baseline / output->disparity[v * W + u];
                        if (z > 0 && z < 5.0)
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
                                cv::Vec3b pixel = output->rgb.at<cv::Vec3b>(v, u);
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
                nlohmann::json message;
                message["cmd_code"] = 0x01;
                message["device_id"] = camera_nodes_[index]->camera_config["device_id"].get<int>();
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
                
                udp_client->SendMsg(message.dump());
            }
            if(pub_laser_){
                camera_nodes_[index]->laserscan_pub_->publish(std::move(scan_msg));
            }
            if(pub_pc_){
                point_cloud_msg.width = point_size;
                point_cloud_msg.row_step = point_cloud_msg.point_step * point_cloud_msg.width;
                point_cloud_msg.data.resize(point_size * point_cloud_msg.point_step *
                                            point_cloud_msg.height);
                camera_nodes_[index]->point_cloud_pub_->publish(std::move(pc_msg));
            }

            int64_t pub_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            output->info += fmt::format(", pub time: {:.3f}", pub_time / 1000.0);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), output->info.c_str());
        }

    public:
        std::string target_frame_;
        double tolerance_;
        double min_height_, max_height_, angle_min_, angle_max_, angle_increment_, scan_time_, range_min_, range_max_;
        bool use_inf_;
        double inf_epsilon_;

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
        ThreadPool preprocess_pool_,postprocess_pool_;
        ThreadPool udp_pool_,web_pool_;

    private:
        hbPackedDNNHandle_t packed_dnn_handle_;
        std::vector<std::shared_ptr<CameraNode>> camera_nodes_;
        std::vector<std::shared_ptr<BaseModel>> models_;

    private:
        std::shared_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
        std::thread spin_thread_;
};

