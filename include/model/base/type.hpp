#pragma once

#include "common/common.h"

enum INPUT_IMAGE_TYPE{
    NV12,
    BGR,
    RGB
};

struct MODEL_INPUT{
    INPUT_IMAGE_TYPE image_type;
    int image_W;
    int image_H;
    std::vector<cv::Mat> images;
    cv::Mat render_image;
};

// yolo
struct DETECT_OUTPUT{
    std::vector<float> scaler;
    std::vector<std::vector<float>> bboxes;
    std::vector<float> scores;
    std::vector<std::string> names;
    nlohmann::json detect_labels;
};
// stereo
struct STEREO_OUTPUT{
    cv::Mat disparity;
    cv::Mat colormap;
};
// track
struct TRACK_OUTPUT{
    std::vector<std::vector<float>> bboxes;   
    std::vector<int> track_ids;
    std::vector<std::string> names;
    std::vector<int> box_areas;
    
    std::vector<std::string> track_names;
    nlohmann::json track_labels;
};
// output
struct MODEL_OUTPUT{
    std::vector<DETECT_OUTPUT> detect_output;
    STEREO_OUTPUT stereo_output;
    TRACK_OUTPUT track_output;
};

struct InferenceData_t{
    MODEL_INPUT input;
    MODEL_OUTPUT output;
};

struct YOLO_CONFIG{
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

struct STEREO_CONFIG{
    int input_H = 352, input_W = 640;
};

struct TRACK_CONFIG{
    float track_thresh = 0.5;
    float high_thresh = 0.6;
    float match_thresh = 0.8;
    int track_buffer = 30;
};

// struct MODEL_CONFIG{
//     YOLO_CONFIG yolo_config;
//     STEREO_CONFIG stereo_config;
//     TRACK_CONFIG track_config;
// };
