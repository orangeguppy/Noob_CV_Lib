#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H
#include <opencv2/opencv.hpp>

/*
 * This function takes an RGB CV MAT IMage and returns its greyscaled version
 */
void convertToGreyscale(const cv::InputArray &src, cv::OutputArray &dst);
/*
 * This function applies the Sobel filter to an RGB image
 */
void applySobel(const cv::InputArray &src, cv::OutputArray &dst);
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