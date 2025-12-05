#include "perception_on_stereo/PerceptionNode.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<PerceptionNode>();

    node->Start();    
    rclcpp::WallRate loop_rate(30);
    
    while (rclcpp::ok())
    {
        node->Run();   
        // loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
