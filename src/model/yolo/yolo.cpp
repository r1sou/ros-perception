#include "model/yolo/yolo.h"

void YOLO::configuration(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path, nlohmann::json config)
{
    configuration_model(packed_dnn_handle, model_path);
    configuration_tensor();
    configuration_config(config);
    RCLCPP_INFO_STREAM(rclcpp::get_logger("YOLO"), fmt::format("\33[32mload model from {} success!\33[0m", model_path).c_str());
}

void YOLO::configuration_config(nlohmann::json config)
{
    model_config_.input_H        = input_H;
    model_config_.input_W        = input_W;
    if (!config.size()){
        return;
    }
    model_config_.class_names    = config.value("class_names", model_config_.class_names);
    model_config_.class_num      = model_config_.class_names.size();
    model_config_.conf_threshold = config.value("conf_threshold", model_config_.conf_threshold);
    model_config_.iou_threshold  = config.value("iou_threshold", model_config_.iou_threshold);
    model_config_.reg_max        = config.value("reg_max", model_config_.reg_max);
    model_config_.nms_topk       = config.value("nms_topk", model_config_.nms_topk);
}

void YOLO::preprocess(std::shared_ptr<InferenceData_t> infer_data, int index)
{
    cv::Mat input, bgr, resize;

    if (infer_data->input.image_type == INPUT_IMAGE_TYPE::BGR)
    {
        bgr = infer_data->input.images[0];
    }
    else if (infer_data->input.image_type == INPUT_IMAGE_TYPE::RGB)
    {
        image_conversion::rgb_to_bgr(infer_data->input.images[0], bgr);
    }
    else if (infer_data->input.image_type == INPUT_IMAGE_TYPE::NV12)
    {
        image_conversion::nv12_to_bgr(infer_data->input.images[0], bgr);
    }

    if(infer_data->input.render_image.empty()){
        infer_data->input.render_image = bgr;
    }

    if (infer_data->input.image_H != model_config_.input_H || infer_data->input.image_W != model_config_.input_W)
    {
        image_conversion::letterbox(
            bgr,
            resize,
            model_config_.input_W,
            model_config_.input_H,
            infer_data->output.detect_output[index].scaler);
    }
    else
    {
        resize = bgr;
    }

    image_conversion::bgr_to_nv12(resize, input);

    hbSysWriteMem(&input_tensor[0].sysMem[0], (char *)input.data, input.rows * input.cols);
    hbSysFlushMem(&input_tensor[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
}

void YOLO::postprocess(std::shared_ptr<InferenceData_t> infer_data, int index)
{
    for (auto &tensor : output_tensor)
    {
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
    }

    std::vector<std::vector<cv::Rect2d>> bboxes;
    std::vector<std::vector<float>> scores;

    bboxes.resize(model_config_.class_num);
    scores.resize(model_config_.class_num);

    DecodeBox(
        80, 80, 8.0,
        reinterpret_cast<float *>(output_tensor[0].sysMem[0].virAddr),
        reinterpret_cast<int32_t *>(output_tensor[1].sysMem[0].virAddr),
        reinterpret_cast<float *>(output_tensor[1].properties.scale.scaleData),
        bboxes, scores);
    DecodeBox(
        40, 40, 16.0,
        reinterpret_cast<float *>(output_tensor[2].sysMem[0].virAddr),
        reinterpret_cast<int32_t *>(output_tensor[3].sysMem[0].virAddr),
        reinterpret_cast<float *>(output_tensor[3].properties.scale.scaleData),
        bboxes, scores);
    DecodeBox(
        20, 20, 32.0,
        reinterpret_cast<float *>(output_tensor[4].sysMem[0].virAddr),
        reinterpret_cast<int32_t *>(output_tensor[5].sysMem[0].virAddr),
        reinterpret_cast<float *>(output_tensor[5].properties.scale.scaleData),
        bboxes, scores);

    std::vector<std::vector<int>> indices(model_config_.class_num);
    for (int i = 0; i < model_config_.class_num; i++)
    {
        cv::dnn::NMSBoxes(bboxes[i], scores[i], model_config_.conf_threshold, model_config_.iou_threshold, indices[i], 1.f, model_config_.nms_topk);
    }

    auto scaler = infer_data->output.detect_output[index].scaler;
    auto detect_labels = infer_data->output.detect_output[index].detect_labels;
    
    for (int cls_id = 0; cls_id < model_config_.class_num; cls_id++)
    {
        for (auto idx : indices[cls_id])
        {
            std::string name = model_config_.class_names[cls_id];
            if (detect_labels.size() && !detect_labels.contains(name))
            {
                continue;
            }

            float x1 = (bboxes[cls_id][idx].x - scaler[2]) / scaler[0];
            float y1 = (bboxes[cls_id][idx].y - scaler[3]) / scaler[1];
            float x2 = x1 + (bboxes[cls_id][idx].width) / scaler[0];
            float y2 = y1 + (bboxes[cls_id][idx].height) / scaler[1];

            infer_data->output.detect_output[index].bboxes.push_back(std::vector<float>{x1, y1, x2, y2});
            infer_data->output.detect_output[index].scores.push_back(scores[cls_id][idx]);
            infer_data->output.detect_output[index].names.push_back(name);
        }
    }
}

void YOLO::DecodeBox(
    int H, int W, float stride,
    float *cls_ptr, int32_t *box_ptr, float *scale_ptr,
    std::vector<std::vector<cv::Rect2d>> &bboxes, std::vector<std::vector<float>> &scores)
{
    float conf_thres = -log(1 / model_config_.conf_threshold - 1);
    for (int h = 0; h < H; ++h)
    {
        for (int w = 0; w < W; ++w)
        {
            int cls_id = 0;
            for (int i = 1; i < model_config_.class_num; ++i)
                if (cls_ptr[i] > cls_ptr[cls_id])
                    cls_id = i;
            if (cls_ptr[cls_id] < conf_thres)
            {
                cls_ptr += model_config_.class_num;
                box_ptr += model_config_.reg_max * 4;
                continue;
            }
            float score = 1.f / (1.f + std::exp(-cls_ptr[cls_id]));
            float ltrb[4] = {0.f};
            for (int k = 0; k < 4; ++k)
            {
                float sum = 0.f;
                for (int j = 0; j < model_config_.reg_max; ++j)
                {
                    float d = std::exp(float(box_ptr[k * model_config_.reg_max + j]) * scale_ptr[k * model_config_.reg_max + j]);
                    ltrb[k] += d * j;
                    sum += d;
                }
                ltrb[k] /= sum;
            }
            if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
            {
                cls_ptr += model_config_.class_num;
                box_ptr += model_config_.reg_max * 4;
                continue;
            }
            float x1 = (w + 0.5f - ltrb[0]) * stride;
            float y1 = (h + 0.5f - ltrb[1]) * stride;
            float x2 = (w + 0.5f + ltrb[2]) * stride;
            float y2 = (h + 0.5f + ltrb[3]) * stride;

            bboxes[cls_id].emplace_back(x1, y1, x2 - x1, y2 - y1);
            scores[cls_id].push_back(score);
            cls_ptr += model_config_.class_num;
            box_ptr += model_config_.reg_max * 4;
        }
    }
}