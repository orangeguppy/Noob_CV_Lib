#ifndef IMAGE_PROCESSING_II_H
#define IMAGE_PROCESSING_II_H
#include <opencv2/opencv.hpp>

/*
 * This function takes an RGB CV MAT IMage and returns its greyscaled version
 */
void applyFourierTransformSingleChannel(cv::Mat& src, cv::Mat& dst);
void plotMagnitudeSpectrum(const cv::Mat &complexImage);
#endif