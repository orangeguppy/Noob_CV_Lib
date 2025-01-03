#include <iostream>
#include "core/image_processing.h"
#include "core/image_processing_ii.h"
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

    cv::Mat inputImage3 = cv::imread("C:/cvlib_proj/examples/dood.png");
    cv::Mat outputImage3;
    applyMedianFilter(inputImage3, outputImage3, 3);
    std::cout <<"Done!!\n";

    cv::Mat inputImage4 = cv::imread("C:/cvlib_proj/examples/doge.png", cv::IMREAD_GRAYSCALE);
    cv::Mat outputImage4;
    applyBilateralFilterSingleChannel(inputImage4, outputImage4, 5, 3.0, 25.0);
    std::cout <<"Done!!\n";

    cv::Mat inputImage5 = cv::imread("C:/cvlib_proj/examples/rubiks.png", cv::IMREAD_GRAYSCALE);
    cv::Mat outputImage5;
    cv::Mat finalOutput;
    applyFourierTransformSingleChannel(inputImage5, outputImage5);
    applyFrequencyFilter(outputImage5, 10.0, false);
    invertFourierTransformSingleChannel(outputImage5, finalOutput);
    cv::imwrite("invertedFourier.png", finalOutput);
    plotMagnitudeSpectrum(outputImage5);
    std::cout <<"Done!!\n";
}