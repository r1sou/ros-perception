#pragma once

#include "perception_on_depth/imgproc.h"

#ifdef __aarch64__

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

struct YoloModelConfig
{
    int class_num = 80;
    std::vector<std::string> class_names = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
                                            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
                                            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                                            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                                            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
                                            "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard",
                                            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                                            "teddy bear", "hair drier", "toothbrush"};
    int input_H = 640, input_W = 640;
    float conf_threshold = 0.5, iou_threshold = 0.7;
    int reg_max = 16, nms_topk = 300;
};

struct YoloModelOutput
{
    std::vector<float> scores;
    std::vector<std::string> names;
    std::vector<std::vector<float>> bboxes;
};

class BaseYoloModel
{
public:
    BaseYoloModel(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path, nlohmann::json &config_){
        InitModel(packed_dnn_handle, model_path, config_);
    }
    ~BaseYoloModel()
    {
        release();
    }

public:
    void InitModel(hbPackedDNNHandle_t &packed_dnn_handle, std::string model_path, nlohmann::json &config_)
    {
        {
            {
                const char *model_path_c = model_path.c_str();
                int model_count = 0;
                const char **model_name_list;
                hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path_c, 1);
                hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
                hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

                MallocTensor();
            }
            {
                config.class_num = config_.value("class_num", config.class_num);
                config.class_names = config_.value("class_names", config.class_names);
                config.input_H = config_.value("input_H", config.input_H);
                config.input_W = config_.value("input_W", config.input_W);
                config.conf_threshold = config_.value("conf_threshold", config.conf_threshold);
                config.iou_threshold = config_.value("iou_threshold", config.iou_threshold);
                config.reg_max = config_.value("reg_max", config.reg_max);
                config.nms_topk = config_.value("nms_topk", config.nms_topk);
            }
            std::string info = fmt::format("\033[32mload model from {} success!\033[0m", model_path);
            RCLCPP_INFO_STREAM(rclcpp::get_logger(""), info.c_str());
        }
    }
    void MallocTensor(int batch_size = 4)
    {
        scalers.resize(batch_size);
        batch_input_tensor.resize(batch_size);
        batch_output_tensor.resize(batch_size);

        int input_count;
        hbDNNGetInputCount(&input_count, dnn_handle);

        for (auto &input_tensor : batch_input_tensor)
        {
            for (int i = 0; i < input_count; i++)
            {
                hbDNNTensor tensor;
                hbDNNGetInputTensorProperties(&tensor.properties, dnn_handle, i);
                if (tensor.properties.tensorType == HB_DNN_IMG_TYPE_NV12)
                {
                    int32_t batch = tensor.properties.alignedShape.dimensionSize[0];
                    int32_t batch_size = tensor.properties.alignedByteSize / batch;

                    tensor.properties.alignedByteSize = batch_size;
                    tensor.properties.validShape.dimensionSize[0] = 1;
                    tensor.properties.alignedShape = tensor.properties.validShape;

                    for (int j = 0; j < batch; j++)
                    {
                        hbSysAllocCachedMem(&tensor.sysMem[0], batch_size);
                        input_tensor.push_back(tensor);
                    }
                }
                else if (tensor.properties.tensorType == HB_DNN_IMG_TYPE_NV12_SEPARATE)
                {
                    int32_t batch = tensor.properties.alignedShape.dimensionSize[0];
                    int32_t batch_size = tensor.properties.alignedByteSize / batch;

                    tensor.properties.alignedByteSize = batch_size;
                    tensor.properties.validShape.dimensionSize[0] = 1;
                    tensor.properties.alignedShape = tensor.properties.validShape;

                    for (int j = 0; j < batch; j++)
                    {
                        hbSysAllocCachedMem(&tensor.sysMem[0], batch_size * 2 / 3);
                        hbSysAllocCachedMem(&tensor.sysMem[1], batch_size * 1 / 3);
                        input_tensor.push_back(tensor);
                    }
                }
                else
                {
                    int input_memSize = tensor.properties.alignedByteSize;
                    hbSysAllocCachedMem(&tensor.sysMem[0], input_memSize);
                    tensor.properties.alignedShape = tensor.properties.validShape;
                    input_tensor.push_back(tensor);
                }
            }
        }

        int output_count;
        hbDNNGetOutputCount(&output_count, dnn_handle);
        for (auto &output_tensor : batch_output_tensor)
        {
            for (int i = 0; i < output_count; i++)
            {
                hbDNNTensor tensor;
                hbDNNGetOutputTensorProperties(&tensor.properties, dnn_handle, i);
                int output_memSize = tensor.properties.alignedByteSize;
                hbSysAllocCachedMem(&tensor.sysMem[0], output_memSize);
                output_tensor.push_back(tensor);
            }
        }
    }
    void release()
    {
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "\033[32mrelease model\033[0m");
        for (auto &input_tensor : batch_input_tensor)
        {
            for (auto &tensor : input_tensor)
            {
                hbSysFreeMem(&tensor.sysMem[0]);
            }
        }
        for (auto &output_tensor : batch_output_tensor)
        {
            for (auto &tensor : output_tensor)
            {
                hbSysFreeMem(&tensor.sysMem[0]);
            }
        }
    }
    void Preprocess(cv::Mat &image, int batch_idx)
    {
        auto &scaler = scalers[batch_idx];
        cv::Mat resize_image, input_image;
        ImageProc::LetterBox(image, resize_image, config.input_W, config.input_H, scaler);
        ImageProc::BGR2NV12(resize_image, input_image);
        auto &tensor = batch_input_tensor[batch_idx][0];
        hbSysWriteMem(&tensor.sysMem[0], (char *)input_image.data, input_image.rows * input_image.cols);
        hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
    }
    void Inference(int batch_idx)
    {
        auto &input_tensor = batch_input_tensor[batch_idx];
        auto &output_tensor = batch_output_tensor[batch_idx];

        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

        hbDNNTensor *in_ptr = &input_tensor[0];
        hbDNNTensor *out_ptr = &output_tensor[0];
        hbDNNInfer(&task_handle, &out_ptr, in_ptr, dnn_handle, &infer_ctrl_param);

        hbDNNWaitTaskDone(task_handle, 0);
        hbDNNReleaseTask(task_handle);
    }
    void Postprocess(int batch_idx, YoloModelOutput &output)
    {
        auto &output_tensor = batch_output_tensor[batch_idx];
        for (auto &tensor : output_tensor)
        {
            hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        }

        auto scaler = scalers[batch_idx];
        std::vector<std::vector<cv::Rect2d>> bboxes;
        std::vector<std::vector<float>> scores;
        bboxes.resize(config.class_num);
        scores.resize(config.class_num);
        DecodeBox(
            reinterpret_cast<float *>(output_tensor[0].sysMem[0].virAddr),
            reinterpret_cast<int32_t *>(output_tensor[1].sysMem[0].virAddr),
            reinterpret_cast<float *>(output_tensor[1].properties.scale.scaleData),
            80, 80, 8.0,
            bboxes, scores);
        DecodeBox(
            reinterpret_cast<float *>(output_tensor[2].sysMem[0].virAddr),
            reinterpret_cast<int32_t *>(output_tensor[3].sysMem[0].virAddr),
            reinterpret_cast<float *>(output_tensor[3].properties.scale.scaleData),
            40, 40, 16.0,
            bboxes, scores);
        DecodeBox(
            reinterpret_cast<float *>(output_tensor[4].sysMem[0].virAddr),
            reinterpret_cast<int32_t *>(output_tensor[5].sysMem[0].virAddr),
            reinterpret_cast<float *>(output_tensor[5].properties.scale.scaleData),
            20, 20, 32.0,
            bboxes, scores);
        std::vector<std::vector<int>> indices(config.class_num);
        for (int i = 0; i < config.class_num; i++)
        {
            cv::dnn::NMSBoxes(bboxes[i], scores[i], config.conf_threshold, config.iou_threshold, indices[i], 1.f, config.nms_topk);
        }
        for (int cls_id = 0; cls_id < config.class_num; cls_id++)
        {
            for (auto index : indices[cls_id])
            {
                std::string name = config.class_names[cls_id];

                float x1 = (bboxes[cls_id][index].x - scaler[2]) / scaler[0];
                float y1 = (bboxes[cls_id][index].y - scaler[3]) / scaler[1];
                float x2 = x1 + (bboxes[cls_id][index].width) / scaler[0];
                float y2 = y1 + (bboxes[cls_id][index].height) / scaler[1];

                output.bboxes.push_back(std::vector<float>{x1, y1, x2, y2});
                output.scores.push_back(scores[cls_id][index]);
                output.names.push_back(name);
            }
        }
    }

    void DecodeBox(
        float *cls_ptr, int32_t *box_ptr, float *scale_ptr,
        int H, int W, float stride,
        std::vector<std::vector<cv::Rect2d>> &bboxes,
        std::vector<std::vector<float>> &scores)
    {
        float conf_thres = -log(1 / config.conf_threshold - 1);
        for (int h = 0; h < H; ++h)
        {
            for (int w = 0; w < W; ++w)
            {
                int cls_id = 0;
                for (int i = 1; i < config.class_num; ++i)
                    if (cls_ptr[i] > cls_ptr[cls_id])
                        cls_id = i;
                if (cls_ptr[cls_id] < conf_thres)
                {
                    cls_ptr += config.class_num;
                    box_ptr += config.reg_max * 4;
                    continue;
                }
                float score = 1.f / (1.f + std::exp(-cls_ptr[cls_id]));
                float ltrb[4] = {0.f};
                for (int k = 0; k < 4; ++k)
                {
                    float sum = 0.f;
                    for (int j = 0; j < config.reg_max; ++j)
                    {
                        float d = std::exp(float(box_ptr[k * config.reg_max + j]) * scale_ptr[k * config.reg_max + j]);
                        ltrb[k] += d * j;
                        sum += d;
                    }
                    ltrb[k] /= sum;
                }
                if (ltrb[2] + ltrb[0] <= 0 || ltrb[3] + ltrb[1] <= 0)
                {
                    cls_ptr += config.class_num;
                    box_ptr += config.reg_max * 4;
                    continue;
                }
                float x1 = (w + 0.5f - ltrb[0]) * stride;
                float y1 = (h + 0.5f - ltrb[1]) * stride;
                float x2 = (w + 0.5f + ltrb[2]) * stride;
                float y2 = (h + 0.5f + ltrb[3]) * stride;

                bboxes[cls_id].emplace_back(x1, y1, x2 - x1, y2 - y1);
                scores[cls_id].push_back(score);
                cls_ptr += config.class_num;
                box_ptr += config.reg_max * 4;
            }
        }
    }

public:
    YoloModelConfig config;
    std::vector<std::vector<float>> scalers;

private:
    hbDNNHandle_t dnn_handle;
    std::vector<std::vector<hbDNNTensor>> batch_input_tensor;
    std::vector<std::vector<hbDNNTensor>> batch_output_tensor;
};

#endif
