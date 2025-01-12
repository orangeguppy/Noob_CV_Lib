#ifndef SIMPLE_FILTERS_H
#define SIMPLE_FILTERS_H
#include <opencv2/opencv.hpp>

/*
 * This function applies the Sobel filter to an RGB image
 */
void applySobel(const cv::Mat &src, cv::Mat &gradX, cv::Mat &gradY, cv::OutputArray &dst);
void calculateSobelGradientMagnitudes(cv::Mat gradX, cv::Mat gradY, cv::OutputArray &dst);
/*
 * This function applies Gaussian blurring using two separable 1D Gaussian kernels
 */
void applyGaussianBlur(const cv::Mat& src, cv::Mat& dst, float sigma);
/*
 * This function applies Median Filering using a Histogram
 */
void applyMedianFilter(const cv::Mat& src, cv::Mat& dst, int kernelSize);
/*
 * This function applies Bilateral Filering
 */
void applyBilateralFilterSingleChannel(const cv::Mat& src, cv::Mat& dst, int kernelSize, float spatialSigma, float brightnessSigma);
#endif