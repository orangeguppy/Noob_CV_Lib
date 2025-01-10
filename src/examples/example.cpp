#include <iostream>
#include <opencv2/opencv.hpp>
#include "core/simple_filters.h"
#include "core/preprocessing.h"

int main() {
    cv::Mat greyscaleImage;
    convertToGreyscale(readImage("images/subhome-ai.jpg", cv::IMREAD_COLOR), greyscaleImage);
    writeImage("output/greyscale.jpg", greyscaleImage);

    cv::Mat sobelImage;
    applySobel(greyscaleImage, sobelImage);
    writeImage("output/sobel.jpg", sobelImage);

    cv::Mat medFiltImage;
    applyMedianFilter(readImage("images/dood.png", cv::IMREAD_COLOR), medFiltImage, 3);
    writeImage("output/medianfilteringout.jpg", medFiltImage);

    cv::Mat bilatFiltImage;
    applyBilateralFilterSingleChannel(readImage("images/doge.png", cv::IMREAD_GRAYSCALE), bilatFiltImage, 5, 3.0, 25.0);
    writeImage("output/bilatfilteroutput.jpg", bilatFiltImage);

    std::cout <<"Done!!\n";

    // cv::Mat inputImage5 = cv::imread("images/rubiks.jpg", cv::IMREAD_GRAYSCALE);
    // cv::Mat outputImage5;
    // cv::Mat finalOutput;
    // applyFourierTransformSingleChannel(inputImage5, outputImage5);
    // applyFrequencyFilter(outputImage5, 10.0, false);
    // invertFourierTransformSingleChannel(outputImage5, finalOutput);
    // cv::imwrite("invertedFourier.png", finalOutput);
    // plotMagnitudeSpectrum(outputImage5);
    // std::cout <<"Done!!\n";
}