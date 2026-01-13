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

#include "common/buffer.hpp"
#include "common/image_conversion.hpp"

using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image, sensor_msgs::msg::Image>;

class CameraNode : public rclcpp::Node
{
public:
    CameraNode(const rclcpp::NodeOptions &options = rclcpp::NodeOptions()) : Node("camera_sub_node", options)
    {
    }
    ~CameraNode()
    {
    }

public:
    void configuration(nlohmann::json &camera_config);
    void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg1, const sensor_msgs::msg::Image::SharedPtr msg2);
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>::Element> read();

private:
    std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>> buffer_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> sub1_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> sub2_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> syncApproximate_;

private:
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::LaserScan>> laser_scan_pub_;
    std::shared_ptr<rclcpp::Publisher<sensor_msgs::msg::PointCloud2>> point_cloud_pub_;
};