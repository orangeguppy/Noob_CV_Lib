#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H
#include <opencv2/opencv.hpp>

/*
 * This function takes an RGB CV MAT IMage and returns its greyscaled version
 */
void applyFourierTransform(const cv::InputArray &src, cv::OutputArray &dst);
#endif