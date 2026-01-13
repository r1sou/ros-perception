#include "node/node.h"


void PerceptionNode::parse_declare_parameters()
{
    this->declare_parameter("project_root", "");
    this->get_parameter("project_root", project_root_);
    RCLCPP_INFO_STREAM(this->get_logger(), "project_root: " << project_root_);

    this->declare_parameter("camera_config_path", "");
    this->get_parameter("camera_config_path", camera_config_path_);
    camera_config_path_ = project_root_ + "/" + "config" + "/" + camera_config_path_;
    RCLCPP_INFO_STREAM(this->get_logger(), "camera_config_path: " << camera_config_path_);

    this->declare_parameter("client_config_path", "");
    this->get_parameter("client_config_path", client_config_path_);
    client_config_path_ = project_root_ + "/" + "config" + "/" + client_config_path_;
    RCLCPP_INFO_STREAM(this->get_logger(), "client_config_path: " << client_config_path_);

    this->declare_parameter("model_config_path", "");
    this->get_parameter("model_config_path", model_config_path_);
    model_config_path_ = project_root_ + "/" + "config" + "/" + model_config_path_;
    RCLCPP_INFO_STREAM(this->get_logger(), "model_config_path: " << model_config_path_);

    this->declare_parameter("laser_config_path", "");
    this->get_parameter("laser_config_path", laser_config_path_);
    laser_config_path_ = project_root_ + "/" + "config" + "/" + laser_config_path_;
    RCLCPP_INFO_STREAM(this->get_logger(), "laser_config_path: " << laser_config_path_);

    this->declare_parameter("record_config_path", "");
    this->get_parameter("record_config_path", record_config_path_);
    record_config_path_ = project_root_ + "/" + "config" + "/" + record_config_path_;
    RCLCPP_INFO_STREAM(this->get_logger(), "record_config_path: " << record_config_path_);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    this->declare_parameter("record", false);
    this->get_parameter("record", record_);
    RCLCPP_INFO_STREAM(this->get_logger(), "record: " << record_);

    this->declare_parameter("followme", false);
    this->get_parameter("followme", followme_);
    RCLCPP_INFO_STREAM(this->get_logger(), "followme: " << followme_);
    ///////////////////////////////////////////////////////////////////////////////////////////////

    this->declare_parameter("show", false);
    this->get_parameter("show", show_);
    RCLCPP_INFO_STREAM(this->get_logger(), "show: " << show_);

    this->declare_parameter("debug", false);
    this->get_parameter("debug", debug_);
    RCLCPP_INFO_STREAM(this->get_logger(), "debug: " << debug_);
}

void PerceptionNode::configuration()
{
    {
        int ret;
        ret = std::system(
            fmt::format("python {}/scripts/camera.py --file {}/config/camera.json", project_root_, project_root_).c_str()
        );
        if (ret != 0)
        {
            RCLCPP_ERROR_STREAM(this->get_logger(), "\33[31mpython camera.py failed\33[0m");
        }
        ret = std::system(
            fmt::format("python {}/scripts/client.py --file {}/config/client.json", project_root_, project_root_).c_str()
        );
        if (ret != 0)
        {
            RCLCPP_ERROR_STREAM(this->get_logger(), "\33[31mpython client.py failed\33[0m");
        }
    }
    {
        camera_config_ = nlohmann::json::parse(std::ifstream(camera_config_path_));
        client_config_ = nlohmann::json::parse(std::ifstream(client_config_path_));
        model_config_ = nlohmann::json::parse(std::ifstream(model_config_path_));
        laser_config_ = nlohmann::json::parse(std::ifstream(laser_config_path_));
        record_config_ = nlohmann::json::parse(std::ifstream(record_config_path_));
    }
    {
        configuration_camera();
        configuration_client();
        configuration_task();
        configuration_work();
    }
}

void PerceptionNode::configuration_camera()
{
    camera_node_ = std::make_shared<CameraNode>();
    camera_node_->configuration(camera_config_);
    RCLCPP_INFO_STREAM(this->get_logger(), "config camera Finish");
}

void PerceptionNode::configuration_client()
{
    std::string uri = fmt::format(
        "ws://{}:{}", 
        client_config_["client"]["ip"].get<std::string>(), 
        client_config_["client"]["WebSocket"]["port"].get<int>()
    );
    perception_client_ = std::make_shared<PerceptionClient>(uri, client_config_);
    RCLCPP_INFO_STREAM(this->get_logger(), "config client Finish");
}

void PerceptionNode::configuration_task()
{
    {
        engine_ = std::make_shared<Engine>();
        engine_->configuration(model_config_, project_root_);
    }
    {
        followme_task_ = std::make_shared<FollowMe>();
        followme_task_->init_task();
    }
    {
        record_task_ = std::make_shared<Record>();
        record_task_->init_task(record_config_);
    }
    RCLCPP_INFO_STREAM(this->get_logger(), "config task Finish");
}

void PerceptionNode::configuration_work()
{
    worker_threads_.emplace_back(
        std::make_shared<std::thread>(
            [this](){
                rclcpp::WallRate rate(1.0 / 2);
                while(rclcpp::ok()){
                    send_camera_status();
                    rate.sleep();
                }
                RCLCPP_INFO_STREAM(this->get_logger(), "send status thread exit");
            }
        )
    );
    worker_threads_.emplace_back(
        std::make_shared<std::thread>(
            [this](){
                rclcpp::WallRate rate(10);
                while(rclcpp::ok()){
                    followme_thread();
                    rate.sleep();
                }
                RCLCPP_INFO_STREAM(this->get_logger(), "followme thread exit");
            }
        )
    );
    worker_threads_.emplace_back(
        std::make_shared<std::thread>(
            [this](){
                int fps_ = 10;
                rclcpp::WallRate rate(fps_);
                while(rclcpp::ok()){
                    record_thread();
                    rate.sleep();
                }
                RCLCPP_INFO_STREAM(this->get_logger(), "record thread exit");
            }
        )
    );
    RCLCPP_INFO_STREAM(this->get_logger(), "config work Finish");
}

void PerceptionNode::send_camera_status()
{
    if(!perception_client_->connected.load()){
        return;
    }
    nlohmann::json message;
    message["device_id"] = camera_config_["camera"]["device_id"].get<int>();
    message["cmd_code"] = 0x16;
    time_t timestamp = time(NULL);

    message["data"]["follow_collect_status"] = perception_client_->start_follow.load() ? 2 : 0;
    message["data"]["identify_collect_status"] = 0;
    message["data"]["cam_record_status"] = perception_client_->start_cam_record.load() ? 1 : 0;

    message["key"] = JWTGenerator::generate(client_config_["client"]["JWT"]["req_id"], client_config_["client"]["JWT"]["key"]);
    perception_client_->send_message(message.dump());
}

void PerceptionNode::start()
{
    executor_ = std::make_shared<rclcpp::executors::MultiThreadedExecutor>();
    executor_->add_node(camera_node_);
    spin_thread_ = std::thread(
        [this](){
            executor_->spin(); 
        }
    );
    perception_client_->start();
}

void PerceptionNode::stop()
{
    if (executor_)
    {
        executor_->cancel();
        RCLCPP_INFO_STREAM(this->get_logger(), "cancel executor Finish");
    }
    if (spin_thread_.joinable())
    {
        spin_thread_.join();
        RCLCPP_INFO_STREAM(this->get_logger(), "join spin thread Finish");
    }
    for(auto &t : worker_threads_){
        if(t->joinable()){
            t->join();
        }
    }
    worker_threads_.clear();
}

void PerceptionNode::Display()
{
    auto element = camera_node_->read();
    if(!element || !element->msg1 || !element->msg2){
        RCLCPP_INFO_STREAM(this->get_logger(), "image is empty");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }

    int64_t pub_time = element->msg1->header.stamp.sec * 1000LL + element->msg1->header.stamp.nanosec / 1000000;
    int64_t sub_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
    int64_t store_time = element->timestamp.count();

    std::string info = fmt::format("pub: {:.3f}, store: {:.3f}, sub: {:.3f}",pub_time / 1000.0, store_time / 1000.0, sub_time / 1000.0);
    RCLCPP_INFO_STREAM(this->get_logger(), info.c_str());
}

bool PerceptionNode::common_preprocess(std::shared_ptr<InferenceData_t> infer_data){
    auto element = camera_node_->read();
    if(!element || !element->msg1 || !element->msg2){
        RCLCPP_INFO_STREAM(this->get_logger(), "image is empty");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return false;
    }
    std::string encoding = element->msg1->encoding;
    if(encoding == "bgr8" || encoding == "BGR8"){
        infer_data->input.image_type = INPUT_IMAGE_TYPE::BGR;
        infer_data->input.images.push_back(cv_bridge::toCvShare(element->msg1, element->msg1->encoding)->image);
        infer_data->input.images.push_back(cv_bridge::toCvShare(element->msg2, element->msg2->encoding)->image);
        infer_data->input.image_H = element->msg1->height;
        infer_data->input.image_W = element->msg1->width;
    }
    else if(encoding == "rgb8" || encoding == "RGB8"){
        infer_data->input.image_type = INPUT_IMAGE_TYPE::RGB;
        infer_data->input.images.push_back(cv_bridge::toCvShare(element->msg1, element->msg1->encoding)->image);
        infer_data->input.images.push_back(cv_bridge::toCvShare(element->msg2, element->msg2->encoding)->image);
        infer_data->input.image_H = element->msg1->height;
        infer_data->input.image_W = element->msg1->width;
    }
    else if(encoding == "nv12" || encoding == "NV12"){
        infer_data->input.image_type = INPUT_IMAGE_TYPE::NV12;
        infer_data->input.images.push_back(cv::Mat(element->msg1->height * 3 / 2, element->msg1->width, CV_8UC1, element->msg1->data.data(), element->msg1->step));
        infer_data->input.images.push_back(cv::Mat(element->msg2->height * 3 / 2, element->msg2->width, CV_8UC1, element->msg2->data.data(), element->msg2->step));
        infer_data->input.image_H = element->msg1->height;
        infer_data->input.image_W = element->msg1->width;
    }
    return true;
}

void PerceptionNode::followme_thread(){
    if(perception_client_->connected.load()){
        if((perception_client_->start_follow.load() && !is_followme_running_.load()) || followme_){
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            is_followme_running_.store(true);
        }
        if((!perception_client_->start_follow.load() && is_followme_running_.load()) && !followme_){
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            is_followme_running_.store(false);
            engine_->reset_track();
            followme_task_->reset();
        }
        if(!is_followme_running_.load()){
            return;
        }
        auto infer_data = std::make_shared<InferenceData_t>();
        if(!common_preprocess(infer_data)){
            return;
        }
        {
            ScopeTimer t("follow me inference");
            followme_task_->run(infer_data, engine_);
        }
        {
            if(show_){
                // track
                if(infer_data->output.track_output.bboxes.size() > 0){
                    auto &track_output = infer_data->output.track_output;
                    image_render::draw_box(infer_data->input.render_image, track_output.bboxes, track_output.track_names);
                }
                // detect
                else{
                    for(auto &detect_output: infer_data->output.detect_output){
                        image_render::draw_box(infer_data->input.render_image, detect_output.bboxes, detect_output.names);
                    }
                }
            }
            if(show_){
                cv::imshow("inference", infer_data->input.render_image);
                cv::waitKey(1);
            }
        }
        {
            publish_thread_pool_.thread_pool_->detach_task(
                [this, infer_data](){
                    websocket_publish(infer_data, true);
                }
            );
        }
    }
}

void PerceptionNode::record_thread(){
    if(perception_client_->connected.load()){
        if((perception_client_->start_cam_record.load() && !is_record_running_.load()) || record_){
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            is_record_running_.store(true);
        }
        if((!perception_client_->start_cam_record.load() && is_record_running_.load()) && !record_){
            // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            is_record_running_.store(false);
            record_task_->release();
        }
        if(!is_record_running_.load()){
            return;
        }
        auto infer_data = std::make_shared<InferenceData_t>();
        if(!common_preprocess(infer_data)){
            return;
        }
        {
            // ScopeTimer t("record inference");
            record_task_->run(infer_data, engine_);
        }
    }
}

void PerceptionNode::websocket_publish(std::shared_ptr<InferenceData_t> infer_data, bool is_track){
    if(!perception_client_->connected.load()){
        RCLCPP_ERROR_STREAM(this->get_logger(), "websocket is not connected cannot publish");
        return;
    }
    if(infer_data->output.stereo_output.disparity.empty()){
        RCLCPP_ERROR_STREAM(this->get_logger(), "disparity is empty cannot publish");
    }
    nlohmann::json message;
    {
        message["cmd_code"] = 0x12;
        message["device_id"] = camera_config_["camera"]["device_id"].get<int>();
        time_t timestamp = time(NULL);
        message["time_stamp"] = timestamp;
        message["key"] = JWTGenerator::generate(client_config_["client"]["JWT"]["req_id"], client_config_["client"]["JWT"]["key"]);
    }

    float fx = camera_config_["camera"]["calib"]["fx"].get<float>();
    float fy = camera_config_["camera"]["calib"]["fy"].get<float>();
    float cx = camera_config_["camera"]["calib"]["cx"].get<float>();
    float cy = camera_config_["camera"]["calib"]["cy"].get<float>();
    float baseline = camera_config_["camera"]["calib"]["baseline"].get<float>();

    if(is_track){
        auto data = nlohmann::json::array();
        for(int i = 0; i < infer_data->output.track_output.bboxes.size(); i++){
            if(infer_data->output.track_output.track_ids[i] != followme_task_->target_id){
                continue;
            }
            auto &track_output = infer_data->output.track_output;
            int box_center_x = static_cast<int>(track_output.bboxes[i][0] + (track_output.bboxes[i][2] - track_output.bboxes[i][0]) / 2.0f);
            int box_center_y = static_cast<int>(track_output.bboxes[i][1] + (track_output.bboxes[i][3] - track_output.bboxes[i][1]) / 2.0f);
            int box_w = static_cast<int>(track_output.bboxes[i][2] - track_output.bboxes[i][0]);
            int box_h = static_cast<int>(track_output.bboxes[i][3] - track_output.bboxes[i][1]);

            float X = fx * baseline / infer_data->output.stereo_output.disparity.at<float>(box_center_y, box_center_x);
            float Y = (cx - box_center_x) * X / fx;
            float Z = (cy - box_center_y) * X / fy;

            float H = 1.0 * box_h * X / fy;
            float W = 1.0 * box_w * X / fx;

            Z = 0.1, H = 0.1, W = 0.1;

            data.push_back(
                {{"name", "person"},
                {"obj_type", model_config_["detect"]["person"]["obj_type"].get<int>()},
                {"obj_code", model_config_["detect"]["person"]["obj_code"].get<int>()},
                {"loc", fmt::format("{:.2f},{:.2f},{:2f}", X, Y, Z)},
                {"size", fmt::format("{:.2f},{:.2f}", W, H)}}
            );
            break;
        }
        if(data.size() > 0){
            message["data"] = data;
            perception_client_->send_message(message.dump());
        }

    }
}