#pragma once

#include <opencv2/opencv.hpp>

#include <arm_neon.h>
#include "hobot_cv/hobotcv_imgproc.h"

#include "common/common.h"

struct image_conversion
{
    static void bgr_to_nv12(cv::Mat &bgr, cv::Mat &nv12)
    {
        hobot_cv::hobotcv_color(bgr, nv12, hobot_cv::DCOLOR_BGR2YUV_NV12);
    }

    static void nv12_to_bgr(cv::Mat &nv12, cv::Mat &bgr)
    {
        hobot_cv::hobotcv_color(nv12, bgr, hobot_cv::DCOLOR_YUV2BGR_NV12);
    }

    static void rgb_to_bgr(cv::Mat &rgb, cv::Mat &bgr)
    {
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    }

    static void opencv_resize(cv::Mat &src, int src_height, int src_width, cv::Mat &dst, int dst_height, int dst_width)
    {
        if (src_height == dst_height && src_width == dst_width)
        {
            dst = src;
        }
        else{
            cv::resize(src, dst, cv::Size(dst_width, dst_height));  
        } 
    }

    static void hobotcv_resize(cv::Mat &src, int src_height, int src_width, cv::Mat &dst, int dst_height, int dst_width)
    {
        // the src_h and src_w of src are both in bgr/rgb format
        hobot_cv::hobotcv_resize(src, src_height, src_width, dst, dst_height, dst_width);
    }

    static void letterbox(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scaler,int value = 127)
    {
        float ratio = std::min(static_cast<float>(width) / src.cols, static_cast<float>(height) / src.rows);

        int new_width = static_cast<int>(std::round(src.cols * ratio));
        int new_height = static_cast<int>(std::round(src.rows * ratio));

        float dw = static_cast<float>(width - new_width);
        float dh = static_cast<float>(height - new_height);
        int left = static_cast<int>(std::round(dw / 2.0 - 0.1));
        int right = static_cast<int>(std::round(dw / 2.0 + 0.1));
        int top = static_cast<int>(std::round(dh / 2.0 - 0.1));
        int bottom = static_cast<int>(std::round(dh / 2.0 + 0.1));

        cv::Size targetSize(new_width, new_height);
        cv::resize(src, dst, targetSize, cv::INTER_LINEAR);
        cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(value, value, value));

        scaler = std::vector<float>{ratio, ratio, static_cast<float>(left), static_cast<float>(top)};
    }
};

struct image_render{
    static void draw_box(cv::Mat &img, std::vector<std::vector<float>> &bboxes, std::vector<std::string> &names, std::string format = "xyxy")
    {
        // x1 y1 x2 y2 format
        for(int i = 0; i < bboxes.size(); i++){
            cv::Scalar color = names[i].find("target") != std::string::npos ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            if(format == "xyxy"){
                cv::rectangle(img, cv::Point(bboxes[i][0], bboxes[i][1]), cv::Point(bboxes[i][2], bboxes[i][3]), color, 2);
                cv::putText(img, names[i], cv::Point(bboxes[i][0], bboxes[i][1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
            else if(format == "xywh"){
                cv::rectangle(img, cv::Point(bboxes[i][0], bboxes[i][1]), cv::Point(bboxes[i][0] + bboxes[i][2], bboxes[i][1] + bboxes[i][3]), color, 2);
                cv::putText(img, names[i], cv::Point(bboxes[i][0], bboxes[i][1]), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            }
        }
    }
    static void disp2colormap(cv::Mat &disparity, cv::Mat &colormap)
    {
        if (disparity.empty())
        {
            return;
        }
        cv::Mat normalized;
        cv::normalize(disparity, normalized, 0.0, 1.0, cv::NORM_MINMAX);
        normalized.convertTo(normalized, CV_8U, 255.0);
        cv::applyColorMap(normalized, colormap, cv::COLORMAP_JET);
    }
};