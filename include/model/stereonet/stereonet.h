#pragma once

#include "model/base/base.hpp"

class StereoNet : public BasicModel
{
public:
    StereoNet() = default;
    ~StereoNet(){
        RCLCPP_INFO_STREAM(rclcpp::get_logger("StereoNet"), "tensor release");
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
    int postprocess_convex_upsampling(std::vector<hbDNNTensor> &tensors, cv::Mat &disparity);
    int postprocess_convex_upsampling_with_interp(std::vector<hbDNNTensor> &tensors, cv::Mat &disparity);
public:
    STEREO_CONFIG model_config_;
};