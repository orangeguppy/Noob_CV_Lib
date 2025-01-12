#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H
#include <opencv2/opencv.hpp>

/*
 * This function is a manual implementation of the Convolution operation
 */
void manualFilter2D(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, int stride, int depthChannelsCode, const std::string& paddingMode);
#endif