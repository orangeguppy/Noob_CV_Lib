#ifndef FREQ_FILTERS_H
#define FREQ_FILTERS_H
#include <opencv2/opencv.hpp>

/*
 * This function takes an RGB CV MAT IMage and returns its greyscaled version
 */
void applyFourierTransformSingleChannel(const cv::Mat& src, cv::Mat& dst);
void invertFourierTransformSingleChannel(const cv::Mat& src, cv::Mat& dst);
void plotMagnitudeSpectrum(const cv::Mat &complexImage);
void applyFrequencyFilter(cv::Mat& complexImage, float radius, bool isLow);
#endif