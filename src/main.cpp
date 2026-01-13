#include "node/node.h"

int main(int argc, char **argv){
    rclcpp::init(argc, argv);

    auto node = std::make_shared<PerceptionNode>();
    
    node->start();

    rclcpp::WallRate loop_rate(10);

    while(rclcpp::ok()){
        // node->run();
        loop_rate.sleep();
    }

    rclcpp::shutdown();
    cv::destroyAllWindows();
    return 0;
}