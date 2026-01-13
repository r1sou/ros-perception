#pragma once

#include "model/base/base.hpp"

class YOLO : public BasicModel
{
public:
    YOLO() = default;
    ~YOLO(){
        RCLCPP_INFO_STREAM(rclcpp::get_logger("YOLO"), "tensor release");
    }

public:
    void configuration(
        hbPackedDNNHandle_t &packed_dnn_handle, 
        std::string model_path, 
        nlohmann::json config = nlohmann::json()
    ) override;
    void configuration_config(nlohmann::json config) override;
    void preprocess(std::shared_ptr<InferenceData_t> infer_data, int index) override;
    void postprocess(std::shared_ptr<InferenceData_t> infer_data, int index) override;
public:
    void DecodeBox(
        int H, int W, float stride,
        float *cls_ptr, int32_t *box_ptr, float *scale_ptr,
        std::vector<std::vector<cv::Rect2d>> &bboxes,std::vector<std::vector<float>> &scores
    );
public:
    YOLO_CONFIG model_config_;
};
