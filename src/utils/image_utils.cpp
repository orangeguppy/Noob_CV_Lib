#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <tuple>

cv::Mat extractMat(const cv::InputArray &src) {
    // Extract cv::Mat from the input
    if (src.isUMat()) {
        return src.getUMat(cv::ACCESS_READ).getMat(cv::ACCESS_READ);
    } else if (src.isMat()) {
        return src.getMat();
    } else {
        return cv::Mat();;
    }
}

int calculateOutputSize(int inputSize, int filterSize, int pStart, int pEnd, int stride) {
    int intermediate = (inputSize - filterSize + pStart + pEnd) / stride;
    return intermediate + 1;
}

std::tuple<cv::Mat, int, int> applyPaddingAndCalculateOutputSize(const cv::Mat &src, const cv::Mat &kernel, int stride, const std::string& paddingMode) {
    int filterSize = kernel.rows;
    int inputSize = src.rows;

    cv::Mat paddedImage; // Stores the output
    int outputWidthSize, outputHeightSize;

    // Determine padding size based on paddingMode
    if (paddingMode == "VALID") {
        return std::make_tuple(src.clone(), src.cols-filterSize+1, src.rows-filterSize+1);
    } else if (paddingMode == "SAME") {
        int pStart = static_cast<int>(std::floor((1.0*stride * std::ceil(1.0*inputSize/stride) - inputSize + filterSize - stride))/2);
        int pEnd = static_cast<int>(std::ceil((1.0*stride * std::ceil(1.0*inputSize/stride) - inputSize + filterSize - stride))/2);
        if (pStart < 0 || pEnd < 0) {
            throw std::runtime_error("Negative padding detected!");
        }
        cv::copyMakeBorder(
            src, paddedImage, 
            pStart, pEnd,   // Top and Bottom padding
            pStart, pEnd,   // Left and Right padding
            cv::BORDER_CONSTANT, cv::Scalar(0)  // Zero padding, BORDER_CONSTANT means all the padded areas are filled with the same padding
                                                // and cv::Scalar(0) means this padding value is 0
        );
        outputWidthSize = calculateOutputSize(static_cast<int>(src.cols), static_cast<int>(filterSize), pStart, pEnd, stride);
        outputHeightSize = calculateOutputSize(static_cast<int>(src.rows), static_cast<int>(filterSize), pStart, pEnd, stride);

    } else if (paddingMode == "FULL") {
        int pStart = static_cast<int>(1);
        int pEnd = static_cast<int>(filterSize - 1);
        if (pStart < 0 || pEnd < 0) {
            throw std::runtime_error("Negative padding detected!");
        }
        cv::copyMakeBorder(
            src, paddedImage, 
            pStart, pEnd,   // Top and Bottom padding
            pStart, pEnd,   // Left and Right padding
            cv::BORDER_CONSTANT, cv::Scalar(0)  // Zero padding, BORDER_CONSTANT means all the padded areas are filled with the same padding
                                                // and cv::Scalar(0) means this padding value is 0
        );
        outputWidthSize = calculateOutputSize(static_cast<int>(src.cols), static_cast<int>(filterSize), pStart, pEnd, stride);
        outputHeightSize = calculateOutputSize(static_cast<int>(src.rows), static_cast<int>(filterSize), pStart, pEnd, stride);
    } else {
        throw std::invalid_argument("Invalid padding mode! Use SAME, VALID, or FULL.");
    }
    if (paddedImage.empty()) {
        throw std::runtime_error("Padded image is empty after padding!");
    }
    // cv::imwrite("output/padded_image.jpg", paddedImage);
    return std::make_tuple(paddedImage, outputWidthSize, outputHeightSize);
}

void manualFilter2D(const cv::Mat &src, cv::Mat &dst, const cv::Mat &kernel, int stride, int depthChannelsCode, const std::string& paddingMode) {
    // If empty, throw an error
    if (src.empty() || kernel.empty()) {
        throw std::invalid_argument("Error: Empty image!");
    }

    // Get cv::Mat
    cv::Mat inputImage = extractMat(src);

    // Check the kernel is valid
    if (kernel.rows % 2 == 0 || kernel.cols % 2 == 0) {
        throw std::invalid_argument("Error: Kernel is empty or invalid kernel size!");
    }

    // Pad the image
    std::tuple<cv::Mat, int, int> paddingAndOutputSize = applyPaddingAndCalculateOutputSize(src, kernel, stride, paddingMode);
    cv::Mat paddedImage;
    std::get<0>(paddingAndOutputSize).convertTo(paddedImage, CV_32F);
    if (paddedImage.type() != CV_32F) {
        throw std::runtime_error("Padded image type is not CV_32F.");
    }
    int outputWidth = std::get<1>(paddingAndOutputSize);
    int outputHeight = std::get<2>(paddingAndOutputSize);

    // Initialise the output matrix
    dst.create(outputHeight, outputWidth, depthChannelsCode);
    dst.setTo(cv::Scalar(0));

    // Get the centre of the kernel
    int kRows = kernel.rows;
    int kCols = kernel.cols;
    int kCentreY = kRows / 2;
    int kCentreX = kCols / 2;

    std::cout <<"We got here!\n";
    
    // Perform the convolution
    #pragma omp parallel for collapse(2)
    for (int y = kCentreY; y < paddedImage.rows - kCentreY - 1 ; y += stride) {
        for (int x = kCentreX; x < paddedImage.cols - kCentreX - 1 ; x += stride) {
            double sum = 0.0;  // Use double for precision

            // Apply the kernel
            for (int ky = -kCentreY; ky <= kCentreY; ky++) {
                for (int kx = -kCentreX; kx <= kCentreX; kx++) {
                    int pixelY = y + ky;  // Row in the padded source
                    int pixelX = x + kx;  // Column in the padded source

                    if (pixelY < 0 || pixelY >= paddedImage.rows || 
                        pixelX < 0 || pixelX >= paddedImage.cols) {
                        throw std::runtime_error("Out-of-bounds access in padded image.");
                    }

                    // Access pixel and kernel values
                    float pixel = paddedImage.at<float>(pixelY, pixelX);
                    // std::cout <<pixel;
                    float kernelVal = kernel.at<float>(ky + kCentreY, kx + kCentreX);

                    if (std::isnan(pixel) || std::isnan(kernelVal)) {
                        throw std::runtime_error("NaN detected in convolution computation.");
                    }

                    // Sum will accumulate the result of a single convolution and reset when the convolution is done
                    // std::cout<<kernelVal;
                    sum += pixel * kernelVal;
                }
            }

            // Store result based on precision
            int outputY = (y - kCentreY) / stride;
            int outputX = (x - kCentreX) / stride;

            if (outputY < 0 || outputY >= dst.rows || outputX < 0 || outputX >= dst.cols) {
                throw std::runtime_error("Out-of-bounds access in output matrix.");
            }

            if (depthChannelsCode == CV_32F) {
                dst.at<float>(outputY, outputX) = static_cast<float>(sum);
            } else if (depthChannelsCode == CV_8U) {
                dst.at<uchar>(outputY, outputX) = cv::saturate_cast<uchar>(sum);
            } else if (depthChannelsCode == CV_16S) {
                dst.at<short>(outputY, outputX) = cv::saturate_cast<short>(sum);
            }
        }
    }
}