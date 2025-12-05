#include "perception_on_depth/imgproc.h"

void ImageProc::Resize(cv::Mat &src, cv::Mat &dst, int width, int height)
{
    if (src.cols == width && src.rows == height)
    {
        dst = src.clone();
        return;
    }
    cv::resize(src, dst, cv::Size(width, height));
}

void ImageProc::Resize(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scale)
{
    float scale_x = static_cast<float>(width) / src.cols;
    float scale_y = static_cast<float>(height) / src.rows;
    scale = {scale_x, scale_y, 0.0, 0.0};

    Resize(src, dst, width, height);
}

void ImageProc::LetterBox(cv::Mat &src, cv::Mat &dst, int width, int height)
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
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    // scales = std::vector<float>{ratio, ratio, static_cast<float>(left), static_cast<float>(top)};
}

void ImageProc::LetterBox(cv::Mat &src, cv::Mat &dst, int width, int height, std::vector<float> &scale)
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
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    scale = std::vector<float>{ratio, ratio, static_cast<float>(left), static_cast<float>(top)};
}

void ImageProc::BGR2RGB(cv::Mat &src, cv::Mat &dst)
{
    cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
}

void ImageProc::MinMaxNormalize(cv::Mat &src)
{
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);

    src = (src - minVal) / (maxVal - minVal);
}

void ImageProc::MinMaxNormalize(cv::Mat &src, cv::Mat &dst)
{
    double minVal, maxVal;
    cv::minMaxLoc(src, &minVal, &maxVal);

    dst = (src - minVal) / (maxVal - minVal);
}

#ifdef __aarch64__
void ImageProc::BGR2NV12_neon(uint8_t *src, uint8_t *dst, int width, int height)
{
    int frameSize = width * height;
    int yIndex = 0;
    int uvIndex = frameSize;
    const uint16x8_t u16_rounding = vdupq_n_u16(128);
    const int16x8_t s16_rounding = vdupq_n_s16(128);
    const int8x8_t s8_rounding = vdup_n_s8(128);
    const uint8x16_t offset = vdupq_n_u8(16);
    const uint16x8_t mask = vdupq_n_u16(255);

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width >> 4; i++)
        {
            // Load rgb
            uint8x16x3_t pixel_rgb;
            pixel_rgb = vld3q_u8(src);
            src += 48;

            uint8x8x2_t uint8_r;
            uint8x8x2_t uint8_g;
            uint8x8x2_t uint8_b;
            uint8_r.val[0] = vget_low_u8(pixel_rgb.val[2]);
            uint8_r.val[1] = vget_high_u8(pixel_rgb.val[2]);
            uint8_g.val[0] = vget_low_u8(pixel_rgb.val[1]);
            uint8_g.val[1] = vget_high_u8(pixel_rgb.val[1]);
            uint8_b.val[0] = vget_low_u8(pixel_rgb.val[0]);
            uint8_b.val[1] = vget_high_u8(pixel_rgb.val[0]);

            uint16x8x2_t uint16_y;
            uint8x8_t scalar = vdup_n_u8(66);
            uint8x16_t y;

            uint16_y.val[0] = vmull_u8(uint8_r.val[0], scalar);
            uint16_y.val[1] = vmull_u8(uint8_r.val[1], scalar);
            scalar = vdup_n_u8(129);
            uint16_y.val[0] = vmlal_u8(uint16_y.val[0], uint8_g.val[0], scalar);
            uint16_y.val[1] = vmlal_u8(uint16_y.val[1], uint8_g.val[1], scalar);
            scalar = vdup_n_u8(25);
            uint16_y.val[0] = vmlal_u8(uint16_y.val[0], uint8_b.val[0], scalar);
            uint16_y.val[1] = vmlal_u8(uint16_y.val[1], uint8_b.val[1], scalar);

            uint16_y.val[0] = vaddq_u16(uint16_y.val[0], u16_rounding);
            uint16_y.val[1] = vaddq_u16(uint16_y.val[1], u16_rounding);

            y = vcombine_u8(vqshrn_n_u16(uint16_y.val[0], 8), vqshrn_n_u16(uint16_y.val[1], 8));
            y = vaddq_u8(y, offset);

            vst1q_u8(dst + yIndex, y);
            yIndex += 16;

            // Compute u and v in the even row
            if (j % 2 == 0)
            {
                int16x8_t u_scalar = vdupq_n_s16(-38);
                int16x8_t v_scalar = vdupq_n_s16(112);

                int16x8_t r = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_u8(pixel_rgb.val[2]), mask));
                int16x8_t g = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_u8(pixel_rgb.val[1]), mask));
                int16x8_t b = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_u8(pixel_rgb.val[0]), mask));

                int16x8_t u;
                int16x8_t v;
                uint8x8x2_t uv;

                u = vmulq_s16(r, u_scalar);
                v = vmulq_s16(r, v_scalar);

                u_scalar = vdupq_n_s16(-74);
                v_scalar = vdupq_n_s16(-94);
                u = vmlaq_s16(u, g, u_scalar);
                v = vmlaq_s16(v, g, v_scalar);

                u_scalar = vdupq_n_s16(112);
                v_scalar = vdupq_n_s16(-18);
                u = vmlaq_s16(u, b, u_scalar);
                v = vmlaq_s16(v, b, v_scalar);

                u = vaddq_s16(u, s16_rounding);
                v = vaddq_s16(v, s16_rounding);

                uv.val[0] = vreinterpret_u8_s8(vadd_s8(vqshrn_n_s16(u, 8), s8_rounding));
                uv.val[1] = vreinterpret_u8_s8(vadd_s8(vqshrn_n_s16(v, 8), s8_rounding));

                vst2_u8(dst + uvIndex, uv);

                uvIndex += 16;
            }
        }
    }
}
#endif

#ifdef __aarch64__
void ImageProc::BGR2NV12(cv::Mat &src, cv::Mat &dst)
{
    int width = src.cols;
    int height = src.rows;
    dst = cv::Mat(height * 3 / 2, width, CV_8UC1);
    BGR2NV12_neon(src.data, dst.data, width, height);
}
#endif

void ImageRender::DrawBox(cv::Mat &img, std::vector<float> &box, std::string &label)
{

    cv::rectangle(img, cv::Point(box[0], box[1]), cv::Point(box[2], box[3]), cv::Scalar(0, 255, 0), 2);
    cv::putText(img, label, cv::Point(box[0], box[1] - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}

void ImageRender::DrawBox(cv::Mat &img, std::vector<std::vector<float>> &boxes, std::vector<std::string> &labels)
{
    for (size_t i = 0; i < boxes.size(); i++)
    {
        ImageRender::DrawBox(img, boxes[i], labels[i]);
    }
}

void ImageRender::DepthToColorMap(std::vector<float> &depth, int H, int W, cv::Mat &color_map,bool smooth_for_visualization)
{
    cv::Mat grayMat(H, W, CV_32F, const_cast<float *>(depth.data()), W * sizeof(float));
    cv::Mat normalized;
    cv::normalize(grayMat, normalized, 0.0, 1.0, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U, 255.0);
    cv::applyColorMap(normalized, color_map, cv::COLORMAP_JET);
}

void ImageRender::DepthToColorMap(cv::Mat &depth, cv::Mat &color_map)
{
    cv::Mat depth_nomallize;
    cv::normalize(depth, depth_nomallize, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(depth_nomallize, color_map, cv::COLORMAP_JET);
}
