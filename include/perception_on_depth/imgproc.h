#pragma once

#include "perception_on_depth/common.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

struct ImageProc
{
    static void Resize(cv::Mat &src, cv::Mat &dst, int width, int height);
    static void Resize(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scale);

    static void LetterBox(cv::Mat &src, cv::Mat &dst, int width, int height);
    static void LetterBox(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scale);

    static void BGR2RGB(cv::Mat &src, cv::Mat &dst);

    static void MinMaxNormalize(cv::Mat &src);
    static void MinMaxNormalize(cv::Mat &src, cv::Mat &dst);

#ifdef __aarch64__
    static void BGR2NV12_neon(uint8_t *src, uint8_t *dst, int width, int height);
    static void BGR2NV12(cv::Mat &src, cv::Mat &dst);
#endif
};

struct ImageRender
{
    static void DrawBox(cv::Mat &img, std::vector<float> &box, std::string &label);
    static void DrawBox(cv::Mat &img, std::vector<std::vector<float>> &boxes, std::vector<std::string> &labels);
    static void DepthToColorMap(std::vector<float> &depth, int H, int W, cv::Mat &color_map, bool smooth_for_visualization = true);
    static void DepthToColorMap(cv::Mat &depth, cv::Mat &color_map);
};

struct Image3D
{
    static void DisparityToDepth(
        std::vector<float> &disparity, cv::Mat &depth,
        int H, int W, float fx, float fy, float cx, float cy, float baseline)
    {
    }
    static void DepthToPointCloud()
    {
    }
    static void DisparityToPointCloud(std::vector<float> &depth, int H, int W)
    {
    }
    static void PointCloudToLaserscan()
    {
    }
};