#include "model/stereonet/stereonet.h"

void StereoNet::configuration(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path, nlohmann::json config)
{
    configuration_model(packed_dnn_handle, model_path);
    configuration_tensor();
    configuration_config(config);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("StereoNet"), fmt::format("\33[32mload model from {} success!\33[0m", model_path).c_str());
}

void StereoNet::configuration_config(nlohmann::json config)
{
    model_config_.input_H = input_H;
    model_config_.input_W = input_W;
}

void StereoNet::preprocess(std::shared_ptr<InferenceData_t> infer_data, int index)
{
    cv::Mat left_input, right_input;

    if (infer_data->input.image_type == INPUT_IMAGE_TYPE::BGR)
    {
        image_conversion::bgr_to_nv12(infer_data->input.images[0], left_input);
        image_conversion::bgr_to_nv12(infer_data->input.images[1], right_input);
    }
    else if (infer_data->input.image_type == INPUT_IMAGE_TYPE::RGB)
    {
        cv::Mat left_bgr, right_bgr;
        image_conversion::rgb_to_bgr(infer_data->input.images[0], left_bgr);
        image_conversion::rgb_to_bgr(infer_data->input.images[1], right_bgr);

        image_conversion::bgr_to_nv12(left_bgr, left_input);
        image_conversion::bgr_to_nv12(right_bgr, right_input);
    }
    else if (infer_data->input.image_type == INPUT_IMAGE_TYPE::NV12)
    {
        left_input = infer_data->input.images[0];
        right_input = infer_data->input.images[1];
    }

    if (infer_data->input.image_H != model_config_.input_H || infer_data->input.image_W != model_config_.input_W)
    {
        cv::Mat left_resize, right_resize;
        image_conversion::hobotcv_resize(
            left_input,
            infer_data->input.image_H,
            infer_data->input.image_W,
            left_resize,
            model_config_.input_H,
            model_config_.input_W);
        image_conversion::hobotcv_resize(
            right_input,
            infer_data->input.image_H,
            infer_data->input.image_W,
            right_resize,
            model_config_.input_H,
            model_config_.input_W);
        left_input = left_resize;
        right_input = right_resize;
    }

    hbSysWriteMem(&input_tensor[0].sysMem[0], (char *)left_input.data, left_input.rows * left_input.cols);
    hbSysFlushMem(&input_tensor[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    hbSysWriteMem(&input_tensor[1].sysMem[0], (char *)right_input.data, right_input.rows * right_input.cols);
    hbSysFlushMem(&input_tensor[1].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
}

void StereoNet::postprocess(std::shared_ptr<InferenceData_t> infer_data, int index)
{
    for (auto &tensor : output_tensor)
    {
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    auto disp_tensor = output_tensor[0];
    auto spx_tensor = output_tensor[1];
    const int32_t *disp_shape = disp_tensor.properties.validShape.dimensionSize;
    int disp_h_dim = disp_shape[2];
    int disp_w_dim = disp_shape[3];
    const int32_t *spx_shape = spx_tensor.properties.validShape.dimensionSize;
    int spx_h_dim = spx_shape[2];
    int spx_w_dim = spx_shape[3];

    if (disp_h_dim == spx_h_dim && disp_w_dim == spx_w_dim)
    {
        int ret = postprocess_convex_upsampling(output_tensor, infer_data->output.stereo_output.disparity);
    }
    else if (disp_h_dim * 4 == spx_h_dim && disp_w_dim * 4 == spx_w_dim)
    {
        int ret = postprocess_convex_upsampling_with_interp(output_tensor, infer_data->output.stereo_output.disparity);
    }
}

int StereoNet::postprocess_convex_upsampling(std::vector<hbDNNTensor> &tensors, cv::Mat &disparity)
{
    // get shape info
    const int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
    int c_dim = disp_shape[1];
    int h_dim = disp_shape[2];
    int w_dim = disp_shape[3];

    // calc disp
    Eigen::MatrixXf result = Eigen::MatrixXf::Zero(h_dim, w_dim);
    if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
        tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32)
    {
        // get tensor info
        auto disp = reinterpret_cast<float *>(tensors[0].sysMem[0].virAddr);
        auto spx = reinterpret_cast<float *>(tensors[1].sysMem[0].virAddr);

        // multiply element-wise and then add in the c channel
        for (int i = 0; i < c_dim; ++i)
        {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                         w_dim);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                        w_dim);
            result.noalias() += matrix_disp.cwiseProduct(matrix_spx);
        }
    }
    else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
    {
        // get tensor info
        auto disp = reinterpret_cast<float *>(tensors[0].sysMem[0].virAddr);
        auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);

        // multiply element-wise and then add in the c channel
        for (int i = 0; i < c_dim; ++i)
        {
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                         w_dim);
            Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                          w_dim);
            result.noalias() += matrix_disp.cwiseProduct(matrix_spx.cast<float>());
        }
    }
    else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
    {
        // get tensor info
        auto disp = reinterpret_cast<int32_t *>(tensors[0].sysMem[0].virAddr);
        auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);

        // multiply element-wise and then add in the c channel
        for (int i = 0; i < c_dim; ++i)
        {
            Eigen::Map<Eigen::Matrix<int32_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                           w_dim);
            Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                          w_dim);
            result.noalias() += matrix_disp.cast<float>().cwiseProduct(matrix_spx.cast<float>());
        }
    }
    else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S16 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
    {
        // get tensor info
        auto disp = reinterpret_cast<int16_t *>(tensors[0].sysMem[0].virAddr);
        auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);

        // multiply element-wise and then add in the c channel
        for (int i = 0; i < c_dim; ++i)
        {
            Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_disp(disp + i * h_dim * w_dim, h_dim,
                                                                                           w_dim);
            Eigen::Map<Eigen::Matrix<int16_t, Eigen::Dynamic, Eigen::Dynamic>> matrix_spx(spx + i * h_dim * w_dim, h_dim,
                                                                                          w_dim);
            result.noalias() += matrix_disp.cast<float>().cwiseProduct(matrix_spx.cast<float>());
        }
    }
    else
    {
        std::string info = fmt::format(
            "=> output tensor type unsupported! tensor[0]: {} tensor[1]: {}",
            static_cast<std::string>(magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[0].properties.tensorType))),
            static_cast<std::string>(magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[1].properties.tensorType))));
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet"), info.c_str());
        return -1;
    }
    // get scale info
    float scale_constant = 1.0;
    float scale_factor;
    float *disp_scale = &scale_constant;
    float *spx_scale = &scale_constant;
    if (tensors[0].properties.quantiType == SCALE)
    {
        disp_scale = tensors[0].properties.scale.scaleData;
    }
    if (tensors[1].properties.quantiType == SCALE)
    {
        spx_scale = tensors[1].properties.scale.scaleData;
    }
    scale_factor = (*disp_scale * *spx_scale);
    if (std::abs(scale_factor - 1.f) > 1e-2)
    {
        result *= (*disp_scale * *spx_scale);
    }
    disparity = cv::Mat::zeros(h_dim, w_dim, CV_32FC1);
    memcpy(disparity.data, result.data(), h_dim * w_dim * sizeof(float));
    return 0;
}

int StereoNet::postprocess_convex_upsampling_with_interp(std::vector<hbDNNTensor> &tensors, cv::Mat &disparity)
{
    const int32_t *disp_shape = tensors[0].properties.validShape.dimensionSize;
    // int disp_c_dim = disp_shape[1];
    int disp_h_dim = disp_shape[2];
    int disp_w_dim = disp_shape[3];
    int total_disp_size = disp_h_dim * disp_w_dim;

    const int32_t *spx_shape = tensors[1].properties.validShape.dimensionSize;
    int spx_c_dim = spx_shape[1];
    int spx_h_dim = spx_shape[2];
    int spx_w_dim = spx_shape[3];
    int total_size = spx_h_dim * spx_w_dim;
    int32_t scale_h = spx_h_dim / disp_h_dim, scale_w = spx_w_dim / disp_w_dim;

    // get scale info
    float scale_constant = 1.0;
    float scale_factor;
    float *disp_scale = &scale_constant;
    float *spx_scale = &scale_constant;
    if (tensors[0].properties.quantiType == SCALE)
    {
        disp_scale = tensors[0].properties.scale.scaleData;
    }
    if (tensors[1].properties.quantiType == SCALE)
    {
        spx_scale = tensors[1].properties.scale.scaleData;
    }
    scale_factor = (*disp_scale * *spx_scale);
    // calc disp
    disparity = cv::Mat::zeros(spx_h_dim, spx_w_dim, CV_32FC1);
    float *result_ptr = reinterpret_cast<float *>(disparity.data);
    if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 &&
        tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
    {
        auto disp = reinterpret_cast<int32_t *>(tensors[0].sysMem[0].virAddr);
        auto spx = reinterpret_cast<int16_t *>(tensors[1].sysMem[0].virAddr);

        for (int32_t i = 0; i < spx_c_dim; ++i)
        {
            for (int32_t y = 0; y < spx_h_dim; ++y)
            {
                // compute y-index for low-res disparity (nearest-neighbor sampling)
                int32_t idx_y = y / scale_h;
                // offset of this output row in result_ptr
                int32_t output_offset = spx_w_dim * y;
                for (int32_t x = 0; x < spx_w_dim; x += 4)
                {
                    // compute x-index for low-res disparity (nearest-neighbor sampling)
                    int32_t idx_x = x / scale_w;

                    // load spx
                    int16x4_t spx_s16 = vld1_s16(&spx[y * spx_w_dim + x]);
                    int32x4_t spx_s32 = vmovl_s16(spx_s16);

                    // load disp
                    int32_t disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
                    int32x4_t disp_s32 = vdupq_n_s32(disp_val_scalar);

                    // convert to float
                    float32x4_t spx_f32 = vcvtq_f32_s32(spx_s32);
                    float32x4_t disp_f32 = vcvtq_f32_s32(disp_s32);

                    // disp * spx
                    float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);

                    // accumulate into output buffer
                    float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
                    float32x4_t updated_output = vaddq_f32(current_output, mul_result);
                    vst1q_f32(&result_ptr[output_offset + x], updated_output);
                }
            }
            // move to next disparity row
            disp += total_disp_size;
            // move to next spx row
            spx += total_size;
        }

        // result * scale_factor
        if (scale_factor != 1.0f)
        {
            for (int32_t j = 0; j < total_size; j += 4)
            {
                vst1q_f32(result_ptr + j, vmulq_n_f32(vld1q_f32(result_ptr + j), scale_factor));
            }
        }
    }
    else if (tensors[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 &&
             tensors[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32)
    {
        auto disp = reinterpret_cast<float *>(tensors[0].sysMem[0].virAddr);
        auto spx = reinterpret_cast<float *>(tensors[1].sysMem[0].virAddr);

        for (int32_t i = 0; i < spx_c_dim; ++i)
        {
            for (int32_t y = 0; y < spx_h_dim; ++y)
            {
                // compute y-index for low-res disparity (nearest-neighbor sampling)
                int32_t idx_y = y / scale_h;
                // offset of this output row in result_ptr
                int32_t output_offset = spx_w_dim * y;
                for (int32_t x = 0; x < spx_w_dim; x += 4)
                {
                    // compute x-index for low-res disparity (nearest-neighbor sampling)
                    int32_t idx_x = x / scale_w;

                    // load spx
                    float32x4_t spx_f32 = vld1q_f32(&spx[y * spx_w_dim + x]);

                    // load disp
                    float disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
                    float32x4_t disp_f32 = vdupq_n_f32(disp_val_scalar);

                    // disp * spx
                    float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);

                    // accumulate into output buffer
                    float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
                    float32x4_t updated_output = vaddq_f32(current_output, mul_result);
                    vst1q_f32(&result_ptr[output_offset + x], updated_output);
                }
            }
            // move to next disparity row
            disp += total_disp_size;
            // move to next spx row
            spx += total_size;
        }

        // result * scale_factor
        if (scale_factor != 1.0f)
        {
            for (int32_t j = 0; j < total_size; j += 4)
            {
                float32x4_t cur = vld1q_f32(result_ptr + j);
                float32x4_t scaled = vmulq_n_f32(cur, scale_factor);
                vst1q_f32(result_ptr + j, scaled);
            }
        }
    }
    else
    {
        std::string info = fmt::format(
            "=> output tensor type unsupported! tensor[0]: {} tensor[1]: {}",
            static_cast<std::string>(magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[0].properties.tensorType))),
            static_cast<std::string>(magic_enum::enum_name(static_cast<hbDNNDataType>(tensors[1].properties.tensorType))));
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("stereonet"), info.c_str());
        return -1;
    }
    return 0;
}