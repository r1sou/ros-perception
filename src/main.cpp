#include "perception_on_depth/PerceptionNode.h"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<PerceptionNode>();

    node->Start();    
    rclcpp::WallRate loop_rate(25);
    
    while (rclcpp::ok())
    {
        node->Inference();   
        loop_rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
