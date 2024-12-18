#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/image_utils.h"
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

void applySobel(const cv::InputArray &src, cv::OutputArray &dst) {
    // Raise an error if the input is empty
    if (src.empty()) {
        throw std::invalid_argument("Error: Empty image matrix!");
    }

    // Check if the InputArray is from the GPU or CPU
    // If it is neither, raise an error
    // This variable stores the image cv::Mat
    cv::Mat inputImage;
    if (src.isUMat()) {
        inputImage = src.getUMat(cv::ACCESS_READ).getMat(cv::ACCESS_READ);
    } else if (src.isMat()) {
        inputImage = src.getMat();
    } else {
        throw std::invalid_argument("Error: Input is not cv::Mat or cv::UMat!");
    }

    // Convert the RGB to greyscale
    cv::Mat greyImg;
    convertToGreyscale(src, greyImg);

    // Create an empty array to store the result
    dst.create(greyImg.rows, greyImg.cols, CV_8UC1);

    // Reference Counting: OpenCV uses smart pointers with reference counting, 
    // so outputImage and dst.getMat() point to the same memory unless a deep copy
    // is triggered.
    cv::Mat outputImage = dst.getMat();

    // Initialise the Gx and Gy finite difference operators
    cv::Mat gradX, gradY;
    cv::Mat Gx = (cv::Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat Gy = (cv::Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    
    Gx.convertTo(Gx, CV_32F);
    Gy.convertTo(Gy, CV_32F);

    // Convolve the original image with each Sobel filterZ    
    manualFilter2D(greyImg, gradX, Gx, 1, CV_32F, "SAME");
    manualFilter2D(greyImg, gradY, Gy, 1, CV_32F, "SAME");
    // std::cout << gradX;

    cv::Mat gradMag;
    cv::magnitude(gradX, gradY, gradMag);
    gradMag.convertTo(dst, CV_8U);

    cv::imwrite("sobelout.jpg", dst.getMat());
}