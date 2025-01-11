#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/image_utils.h"
#include "utils/core_utils.h"
#include "core/constants.h"

/*
 * @brief Converts an RGB image to greyscale
 * @param src InputArray containing the input RGB image
 * @param dst OutputArray containing the output greyscale image
 * @note If the input image is from the GPU, it will be moved to CPU
*/
void convertToGreyscale(const cv::InputArray &src, cv::OutputArray &dst) {
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

    // Get pointers to the input and output
    cv::Vec3b* inputData = inputImage.ptr<cv::Vec3b>();
    uchar* outputData = outputImage.ptr<uchar>();
    int totalPixels = inputImage.rows * inputImage.cols;

    // Loop through the input image with OpenMp for parallelism
    #pragma omp parallel for
    for (int index = 0; index < totalPixels; index++) {
        const cv::Vec3b pixel = inputData[index];
        outputData[index] = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
    }
}

/*
 * @brief Read an image from a file
 * @param filepath File path to read the image from
 * @param color Whether to read the image in greyscale or color
*/
cv::Mat readImage(const std::string filepath, int color = cv::IMREAD_COLOR) {
    cv::Mat image = cv::imread(filepath, color);

    if(image.empty()) {
        throw std::runtime_error("Could not read image: " + filepath);
    }

    return image;
}

/*
 * @brief Write an image to a file
 * @param filepath File path to write the image to
 * @param image Image to write to a
*/
void writeImage(const std::string filepath, cv::Mat& image) {
    if (!cv::imwrite(filepath, image)) {
        throw std::runtime_error("Could not write image: " + filepath);
    }
}
