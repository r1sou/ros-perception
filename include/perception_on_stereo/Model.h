#pragma once

#include "perception_on_stereo/imgproc.h"

#ifdef __aarch64__

#include "dnn/hb_dnn.h"
#include "dnn/hb_sys.h"

struct ModelConfig{
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

struct ModelOutput{
    std::vector<std::vector<float>> bboxes;
    std::vector<float> scores;
    std::vector<std::string> names;
    std::vector<float> disparity;
    cv::Mat rgb;
    std::string info;

    void clear(){
        bboxes.clear();
        scores.clear();
        names.clear();
        disparity.clear();
    }
};

class BaseModel{
public:
    BaseModel(std::string model_name):model_name(model_name){

    }
    ~BaseModel(){
        release();
    }
public:
    virtual void configuration(hbPackedDNNHandle_t &packed_dnn_handle,std::string model_path, nlohmann::json model_config) = 0;
    void allocate(){
        hbDNNGetInputCount(&input_count, dnn_handle);
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
        hbDNNGetOutputCount(&output_count, dnn_handle);
        for (int i = 0; i < output_count; i++)
        {
            hbDNNTensor tensor;
            hbDNNGetOutputTensorProperties(&tensor.properties, dnn_handle, i);
            int output_memSize = tensor.properties.alignedByteSize;
            hbSysAllocCachedMem(&tensor.sysMem[0], output_memSize);
            output_tensor.push_back(tensor);
        }
    }
    void release()
    {
        RCLCPP_INFO_STREAM(rclcpp::get_logger(""), "\033[32mrelease model\033[0m");
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
    virtual void preprocess(std::vector<cv::Mat> &inputs) = 0;
    void inference(){
        hbDNNTaskHandle_t task_handle = nullptr;
        hbDNNInferCtrlParam infer_ctrl_param;
        HB_DNN_INITIALIZE_INFER_CTRL_PARAM(&infer_ctrl_param);

        hbDNNTensor *in_ptr = &input_tensor[0];
        hbDNNTensor *out_ptr = &output_tensor[0];
        hbDNNInfer(&task_handle, &out_ptr, in_ptr, dnn_handle, &infer_ctrl_param);

        hbDNNWaitTaskDone(task_handle, 0);
        hbDNNReleaseTask(task_handle);
    }
    virtual void postprocess(ModelOutput &output) = 0;
    virtual void DecodeBox(
        int H, int W, float stride,
        float *cls_ptr, int32_t *box_ptr, float *scale_ptr,
        std::vector<std::vector<cv::Rect2d>> &bboxes,std::vector<std::vector<float>> &scores
    ){

    }
public:
    ModelConfig model_config_;
    std::string model_name;
public:
    int input_count, output_count;
    std::vector<float> scale;
public:
    hbDNNHandle_t dnn_handle;
    std::vector<hbDNNTensor> input_tensor;
    std::vector<hbDNNTensor> output_tensor;
};

class YOLO: public BaseModel{
public:
    YOLO(std::string model_name):BaseModel(model_name){
    }
    ~YOLO() = default;
public:
    void configuration(hbPackedDNNHandle_t &packed_dnn_handle,std::string model_path, nlohmann::json model_config) override{
        const char *model_path_c = model_path.c_str();
        int model_count = 0;
        const char **model_name_list;
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path_c, 1);
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

        allocate();

        model_config_.class_num = model_config.value("class_num", model_config_.class_num);
        model_config_.class_names = model_config.value("class_names", model_config_.class_names);
        model_config_.input_H = model_config.value("input_H", model_config_.input_H);
        model_config_.input_W = model_config.value("input_W", model_config_.input_W);
        model_config_.conf_threshold = model_config.value("conf_threshold", model_config_.conf_threshold);
        model_config_.iou_threshold = model_config.value("iou_threshold", model_config_.iou_threshold);
        model_config_.reg_max = model_config.value("reg_max", model_config_.reg_max);
        model_config_.nms_topk = model_config.value("nms_topk", model_config_.nms_topk);
    }
    void preprocess(std::vector<cv::Mat> &inputs) override{
        cv::Mat resize_image,input_image;

        ImageProc::LetterBox(inputs[0], resize_image, model_config_.input_W, model_config_.input_H, scale);
        ImageProc::BGR2NV12(resize_image, input_image);
        hbSysWriteMem(&input_tensor[0].sysMem[0], (char *)input_image.data, input_image.rows * input_image.cols);
        hbSysFlushMem(&input_tensor[0].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);

    }

    void postprocess(ModelOutput &output) override{
        std::vector<std::vector<cv::Rect2d>> bboxes;
        std::vector<std::vector<float>> scores;

        bboxes.resize(model_config_.class_num);
        scores.resize(model_config_.class_num);

        for (auto &tensor : output_tensor)
        {
            hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        }

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
        for (int cls_id = 0; cls_id < model_config_.class_num; cls_id++)
        {
            for (auto index : indices[cls_id])
            {
                std::string name = model_config_.class_names[cls_id];

                float x1 = (bboxes[cls_id][index].x - scale[2]) / scale[0];
                float y1 = (bboxes[cls_id][index].y - scale[3]) / scale[1];
                float x2 = x1 + (bboxes[cls_id][index].width) / scale[0];
                float y2 = y1 + (bboxes[cls_id][index].height) / scale[1];

                output.bboxes.push_back(std::vector<float>{x1, y1, x2, y2});
                output.scores.push_back(scores[cls_id][index]);
                output.names.push_back(name);
            }
        }
    }
    void DecodeBox(
        int H, int W, float stride,
        float *cls_ptr, int32_t *box_ptr, float *scale_ptr,
        std::vector<std::vector<cv::Rect2d>> &bboxes,std::vector<std::vector<float>> &scores
    ) override{

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
};

class StereoNet: public BaseModel{
public:
    StereoNet(std::string model_name):BaseModel(model_name){
    }
    ~StereoNet() = default;
public:
    void configuration(hbPackedDNNHandle_t &packed_dnn_handle,std::string model_path, nlohmann::json model_config) override{
        const char *model_path_c = model_path.c_str();
        int model_count = 0;
        const char **model_name_list;
        hbDNNInitializeFromFiles(&packed_dnn_handle, &model_path_c, 1);
        hbDNNGetModelNameList(&model_name_list, &model_count, packed_dnn_handle);
        hbDNNGetModelHandle(&dnn_handle, packed_dnn_handle, model_name_list[0]);

        allocate();

        model_config_.input_H = model_config.value("input_H", model_config_.input_H);
        model_config_.input_W = model_config.value("input_W", model_config_.input_W);
    }
    void preprocess(std::vector<cv::Mat> &inputs) override{
        for(int i = 0;i < input_count; i++){
            cv::Mat resize_image,input_image;
            ImageProc::Resize(inputs[i], resize_image, model_config_.input_W, model_config_.input_H);
            ImageProc::BGR2NV12(resize_image, input_image);
            hbSysWriteMem(&input_tensor[i].sysMem[0], (char *)input_image.data, input_image.rows * input_image.cols);
            hbSysFlushMem(&input_tensor[i].sysMem[0], HB_SYS_MEM_CACHE_CLEAN);
        }
    }
    void postprocess(ModelOutput &output) override{
        for (auto &tensor : output_tensor)
        {
            hbSysFlushMem(&tensor.sysMem[0], HB_SYS_MEM_CACHE_INVALIDATE);
        }

        int32_t *disp_shape = output_tensor[0].properties.validShape.dimensionSize;
        int disp_c_dim = disp_shape[1];
        int disp_h_dim = disp_shape[2];
        int disp_w_dim = disp_shape[3];
        int total_disp_size = disp_h_dim * disp_w_dim;

        int32_t *spx_shape = output_tensor[1].properties.validShape.dimensionSize;
        int spx_c_dim = spx_shape[1];
        int spx_h_dim = spx_shape[2];
        int spx_w_dim = spx_shape[3];
        int total_size = spx_h_dim * spx_w_dim;
        int32_t scale_h = spx_h_dim / disp_h_dim, scale_w = spx_w_dim / disp_w_dim;

        float scale_constant = 1.0;
        float scale_factor;
        float *disp_scale = &scale_constant;
        float *spx_scale = &scale_constant;
        if (output_tensor[0].properties.quantiType == SCALE)
        {
            disp_scale = output_tensor[0].properties.scale.scaleData;
        }
        if (output_tensor[1].properties.quantiType == SCALE)
        {
            spx_scale = output_tensor[1].properties.scale.scaleData;
        }
        scale_factor = (*disp_scale * *spx_scale);

        output.disparity.resize(total_size, 0.f);
        float *result_ptr = output.disparity.data();
        if (output_tensor[0].properties.tensorType == HB_DNN_TENSOR_TYPE_S32 && output_tensor[1].properties.tensorType == HB_DNN_TENSOR_TYPE_S16)
        {
            int32_t *disp = reinterpret_cast<int32_t *>(output_tensor[0].sysMem[0].virAddr);
            int16_t *spx = reinterpret_cast<int16_t *>(output_tensor[1].sysMem[0].virAddr);

            for (int32_t i = 0; i < spx_c_dim; ++i)
            {
                for (int32_t y = 0; y < spx_h_dim; ++y)
                {
                    int32_t idx_y = y / scale_h;
                    int32_t output_offset = spx_w_dim * y;
                    for (int32_t x = 0; x < spx_w_dim; x += 4)
                    {
                        int32_t idx_x = x / scale_w;

                        int16x4_t spx_s16 = vld1_s16(&spx[y * spx_w_dim + x]);
                        int32x4_t spx_s32 = vmovl_s16(spx_s16);

                        int32_t disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
                        int32x4_t disp_s32 = vdupq_n_s32(disp_val_scalar);

                        float32x4_t spx_f32 = vcvtq_f32_s32(spx_s32);
                        float32x4_t disp_f32 = vcvtq_f32_s32(disp_s32);

                        float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);
                        float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
                        float32x4_t updated_output = vaddq_f32(current_output, mul_result);
                        vst1q_f32(&result_ptr[output_offset + x], updated_output);
                    }
                }
                disp += total_disp_size;
                spx += total_size;
            }

            if (scale_factor != 1.0f)
            {
                for (int32_t j = 0; j < total_size; j += 4)
                {
                    vst1q_f32(result_ptr + j, vmulq_n_f32(vld1q_f32(result_ptr + j), scale_factor));
                }
            }
        }
        else if (output_tensor[0].properties.tensorType == HB_DNN_TENSOR_TYPE_F32 && output_tensor[1].properties.tensorType == HB_DNN_TENSOR_TYPE_F32)
        {
            float *disp = reinterpret_cast<float *>(output_tensor[0].sysMem[0].virAddr);
            float *spx = reinterpret_cast<float *>(output_tensor[1].sysMem[0].virAddr);

            for (int32_t i = 0; i < spx_c_dim; ++i)
            {
                for (int32_t y = 0; y < spx_h_dim; ++y)
                {
                    int32_t idx_y = y / scale_h;
                    int32_t output_offset = spx_w_dim * y;
                    for (int32_t x = 0; x < spx_w_dim; x += 4)
                    {
                        int32_t idx_x = x / scale_w;

                        float32x4_t spx_f32 = vld1q_f32(&spx[y * spx_w_dim + x]);

                        float disp_val_scalar = disp[idx_y * disp_w_dim + idx_x];
                        float32x4_t disp_f32 = vdupq_n_f32(disp_val_scalar);

                        float32x4_t mul_result = vmulq_f32(disp_f32, spx_f32);
                        float32x4_t current_output = vld1q_f32(&result_ptr[output_offset + x]);
                        float32x4_t updated_output = vaddq_f32(current_output, mul_result);
                        vst1q_f32(&result_ptr[output_offset + x], updated_output);
                    }
                }

                disp += total_disp_size;
                spx += total_size;
            }
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
    }
};

#endif