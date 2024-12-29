#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/image_utils.h"
#include "utils/core_utils.h"

#define PI 3.141592653589793

/*
 * @brief Converts an RGB image to greyscale
 * @param src InputArray containing the input RGB image
 * @param dst OutputArray containing the output greyscale image
 * @note If the input image is from the GPU, it will be moved to CPU
*/
void applyFourierTransformSingleChannel(const cv::InputArray &src, cv::OutputArray &dst) {
    // Raise an error if the input is empty
    if (src.empty()) {
        throw std::invalid_argument("Error: Empty image matrix!");
    }

    // Check if the InputArray contains a cv::Mat or cv::UMat
    // If it is neither, raise an error
    cv::Mat inputImage; // This variable stores the cv::Mat stored in src for a valid input

    if (src.isUMat()) {
        inputImage = src.getUMat(cv::ACCESS_READ).getMat(cv::ACCESS_READ);
    } else if (src.isMat()) {
        inputImage = src.getMat();
    } else {
        throw std::invalid_argument("Error: Input is not cv::Mat or cv::UMat!");
    }

    // Create an empty array to store the result
    dst.create(inputImage.rows, inputImage.cols, CV_8UC1);

    // Reference Counting: OpenCV uses smart pointers with reference counting, 
    // so outputImage and dst.getMat() point to the same memory unless a deep copy
    // is triggered.
    cv::Mat outputImage = dst.getMat();

    float exponentPowerConstant = -1.0 * std::sqrt(-1) * 2 * PI;

    // Loop through all pixels in the image
    for (int i = 0 ; i < src.rows ; i++) {
        for (int j = 0 ; j < src.cols ; j++) {
            int currentValue = src.at<uchar>(i, j);
            float exponentTerm = std::exp(exponentPowerConstant * ())
        }
    }
    cv::imwrite("greyscaleout.jpg", outputImage);
}