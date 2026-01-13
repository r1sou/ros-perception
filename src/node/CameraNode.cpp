#include "node/CameraNode.h"

void CameraNode::configuration(nlohmann::json &camera_config)
{
    {
        buffer_ = std::make_shared<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>>();
    }
    {
        sub1_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            shared_from_this(), camera_config["camera"]["topic"]["image1"].get<std::string>());
        sub2_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
            shared_from_this(), camera_config["camera"]["topic"]["image2"].get<std::string>());
        syncApproximate_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), *sub1_, *sub2_);
        syncApproximate_->registerCallback(&CameraNode::ImageCallback, this);
    }
    {
        laser_scan_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>(
            "/stereo/laserscan", 10);
        point_cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/stereo/pointcloud", 10);
    }
    {
        std::string frame_id_ = "stereo_camera_link";
        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = this->get_clock()->now();
        t.header.frame_id = frame_id_;
        t.child_frame_id = "stereo_camera_link_child";

        t.transform.translation.x = 0.0;
        t.transform.translation.y = 0.0;
        t.transform.translation.z = 0.0;

        static_broadcaster_->sendTransform(t);
    }
}

void CameraNode::ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg1, const sensor_msgs::msg::Image::SharedPtr msg2)
{
    buffer_->update(
        [msg1, msg2](sensor_msgs::msg::Image::SharedPtr &msg1_, sensor_msgs::msg::Image::SharedPtr &msg2_)
        {
            msg1_ = msg1;
            msg2_ = msg2;
        }
    );
}

std::shared_ptr<TripletBuffer<sensor_msgs::msg::Image::SharedPtr>::Element> CameraNode::read()
{
    return buffer_->read();
}