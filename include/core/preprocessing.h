#ifndef PREPROCESSING_H
#define PREPROCESSING_H
#include <opencv2/opencv.hpp>

/*
 * This function takes an RGB CV MAT IMage and returns its greyscaled version
 */
void convertToGreyscale(const cv::InputArray &src, cv::OutputArray &dst);

/*
 * Read image from file
 */
cv::Mat readImage(const std::string filepath, int color = cv::IMREAD_COLOR);

/*
 * Write image to file
 */
void writeImage(const std::string filepath, cv::Mat& image);

#endif