#include <iostream>
#include "core/image_processing.h"
#include <opencv2/opencv.hpp>

int main() {
    // Call a function from your library
    cv::Mat inputImage = cv::imread("C:/cvlib_proj/examples/subhome-ai.jpg");
    cv::Mat outputImage;
    convertToGreyscale(inputImage, outputImage);
    // Save the result
    if (cv::imwrite("examples/greyscale.jpg", outputImage)) {
        std::cout << "Greyscale image saved!" << std::endl;
    } else {
        std::cerr << "Error: Failed to save greyscale image!" << std::endl;
    }

    cv::Mat inputImage2 = cv::imread("C:/cvlib_proj/examples/subhome-ai.jpg");
    cv::Mat outputImage2;
    applySobel(inputImage2, outputImage2);
}