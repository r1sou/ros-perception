#pragma once

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

#include "model/base/type.hpp"

#include "common/image_conversion.hpp"

class BasicModel{
public:
    BasicModel() = default;
    ~BasicModel(){
        release();
    }
public:
    virtual void configuration_model(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path){
        const char *model_path_c = model_path.c_str();
        int model_count = 0;
        const char **model_name_list;
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path_c, 1);
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);
    }
    virtual void configuration_tensor(){
        hbDNNGetInputCount(&input_count, dnn_handle);
        hbDNNGetOutputCount(&output_count, dnn_handle);

        RCLCPP_INFO_STREAM(rclcpp::get_logger("Model"), fmt::format("input  count: {}", input_count).c_str());
        RCLCPP_INFO_STREAM(rclcpp::get_logger("Model"), fmt::format("output count: {}", output_count).c_str());

        for (int i = 0; i < input_count; i++)
        {
            hbDNNTensor tensor;
            hbDNNGetInputTensorProperties(&tensor.properties, dnn_handle, i);
            tensor.properties.alignedShape = tensor.properties.validShape;
            hbSysAllocCachedMem(&tensor.sysMem[0], tensor.properties.alignedByteSize);
            input_tensor.push_back(tensor);

            input_N = tensor.properties.validShape.dimensionSize[0];
            input_C = tensor.properties.validShape.dimensionSize[1];
            input_H = tensor.properties.validShape.dimensionSize[2];
            input_W = tensor.properties.validShape.dimensionSize[3];
            RCLCPP_INFO_STREAM(rclcpp::get_logger("Model"), fmt::format("input  {} shape: {}x{}x{}x{}", i, input_N, input_C, input_H, input_W).c_str());
        }
        for (int i = 0; i < output_count; i++)
        {
            hbDNNTensor tensor;
            hbDNNGetOutputTensorProperties(&tensor.properties, dnn_handle, i);
            hbSysAllocCachedMem(&tensor.sysMem[0], tensor.properties.alignedByteSize);
            output_tensor.push_back(tensor);

            int output_N = tensor.properties.validShape.dimensionSize[0];
            int output_C = tensor.properties.validShape.dimensionSize[1];
            int output_H = tensor.properties.validShape.dimensionSize[2];
            int output_W = tensor.properties.validShape.dimensionSize[3];
            RCLCPP_INFO_STREAM(rclcpp::get_logger("Model"), fmt::format("output {} shape: {}x{}x{}x{}", i, output_N, output_C, output_H, output_W).c_str());
        }
    }
    virtual void inference(){
        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

        hbDNNTensor *in_ptr = &input_tensor[0];
        hbDNNTensor *out_ptr = &output_tensor[0];
        hbDNNInfer(&task_handle, &out_ptr, in_ptr, dnn_handle, &infer_ctrl_param);

        hbDNNWaitTaskDone(task_handle, 0);
        hbDNNReleaseTask(task_handle);
    }
    virtual void release(){
        for (auto &tensor : input_tensor)
        {
            hbSysFreeMem(&tensor.sysMem[0]);
        }

        for (auto &tensor : output_tensor)
        {
            hbSysFreeMem(&tensor.sysMem[0]);
        }

    }
public:
    virtual void configuration(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path, nlohmann::json config = nlohmann::json()){

    }
    virtual void reset(){
        
    }
    virtual void configuration_config(nlohmann::json config) = 0;
    virtual void preprocess(std::shared_ptr<InferenceData_t> infer_data, int index) = 0;
    virtual void postprocess(std::shared_ptr<InferenceData_t> infer_data,int index) = 0;
public:
    int input_count, output_count;
    int input_N, input_C, input_H, input_W;
public:
    hbDNNHandle_t dnn_handle;
    std::vector<hbDNNTensor> input_tensor;
    std::vector<hbDNNTensor> output_tensor;
};