#pragma once

#include "task/base.hpp"

class Record : public BasicTask {
public:
    Record() = default;
    ~Record(){
        release();
    }
public:
    void init_task(nlohmann::json task_config = nlohmann::json()) override{
        /*
            config: 
                suffix: str, default: ".mp4"
                fourcc: int, default: 2
                save_dir: str, default: "/home/sunrise/Desktop/dataset"
                fps: int, default: 10
        */
        suffix_ = task_config["record"].value("suffix", ".mp4");
        fourcc_ = task_config["record"].value("fourcc", 0);

        if(fourcc_ == 0){
            fourcc_ = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
        } else if(fourcc_ == 1){
            fourcc_ = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        } else if(fourcc_ == 2){
            fourcc_ = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        } else if(fourcc_ == 3){
            fourcc_ = cv::VideoWriter::fourcc('h', 'v', 'c', '1');
        }

        save_dir_ = task_config["record"].value("save_dir", "/home/sunrise/Desktop/dataset");
        std::filesystem::create_directories(save_dir_);

        fps_ = task_config["record"].value("fps", 10);
    }
    void reset() override{
    }
    void run(std::shared_ptr<InferenceData_t> infer_data, std::shared_ptr<Engine> engine) override{
        if(!video_writer_){
            input_H = infer_data->input.image_H;
            input_W = infer_data->input.image_W;
            filename = save_dir_ + "/" + get_string_date(2) + suffix_;
            video_writer_  = std::make_shared<cv::VideoWriter>(filename, fourcc_, fps_, cv::Size(input_W, input_H));
            RCLCPP_INFO(rclcpp::get_logger("Record-Task"), "create video writer: %s", filename.c_str());
            frame_count = 0;
        }
        cv::Mat image;
        if(infer_data->input.image_type == INPUT_IMAGE_TYPE::BGR || infer_data->input.image_type == INPUT_IMAGE_TYPE::RGB){
            image = infer_data->input.images[0];
        }
        else if(infer_data->input.image_type == INPUT_IMAGE_TYPE::NV12){
            image_conversion::nv12_to_bgr(infer_data->input.images[0], image);
        }
        video_writer_->write(image);
    }
    void release() override{
        if(video_writer_){
            video_writer_->release();
        }
        RCLCPP_INFO(rclcpp::get_logger("Record-Task"), "release video writer, write total frame: %d", frame_count);
    }
public:
    std::string get_string_date(int level) {
        time_t timestamp = time(NULL);
        struct tm *tm_time = localtime(&timestamp);
        std::ostringstream oss;
        if(level == 0){
            oss << std::put_time(tm_time, "%Y-%m-%d");
        }
        else if(level == 1){
            oss << std::put_time(tm_time, "%m-%d");
        }
        else if(level == 2){
            oss << std::put_time(tm_time, "%H-%M");
        }
        return oss.str();
    }
public:
    int input_H, input_W, fps_;
    std::string save_dir_ = "";
    std::string suffix_ = ".mp4";
    int fourcc_ = 0;
    std::shared_ptr<cv::VideoWriter> video_writer_;
public:
    int frame_count = 0;
    std::string filename;
};
