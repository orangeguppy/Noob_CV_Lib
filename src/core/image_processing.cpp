#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/image_utils.h"
#include "utils/core_utils.h"
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
    cv::imwrite("greyscaleout.jpg", outputImage);
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

/*
* We use kernelSize = 3 * sigma + 1 because about 99.7% of data from a Gaussian distribution fall within 3 standard deviations, 
* +1 so that the kernel size is an odd number and there is a clearly defined centre value
*/
cv::Mat generateGaussianKernel1D(float sigma) {
    // Calculate the kernel size
    int kernelSize = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
    int halfKernelSize = kernelSize / 2;

    // Initialise kernel
    cv::Mat kernel(1, kernelSize, CV_32F);

    // Compute kernel values
    // We divide by the total sum of all values in the kernel so that it becomes like a probability distribution
    float sum = 0.0f;
    for (int i = -halfKernelSize; i <= halfKernelSize; ++i) {
        float value = std::exp(-(i * i) / (2 * sigma * sigma));
        kernel.at<float>(0, i + halfKernelSize) = value;
        sum += value;
    }

    // Normalisation
    kernel /= sum;
    return kernel;
}


/*
* Apply Gaussian blurring using two separable 1D Gaussian filters
*/
void applyGaussianBlur(const cv::Mat& src, cv::Mat& dst, float sigma) {
    // Generate the 1D kernel
    cv::Mat kernel1D = generateGaussianKernel1D(sigma);

    if (src.channels() == 1) {
        // If its a Greyscale image, there is only 1 channel so we can
        // apply the filter to the single channel directly
        cv::sepFilter2D(src, dst, CV_32F, kernel1D, kernel1D);
        dst.convertTo(dst, CV_8U);
    } else if (src.channels() == 3) {
        // If its an RGB image image, there are 3 channels so we need to split the image into the 3
        // constituent channels and apply the filter separately to each channel
        std::vector<cv::Mat> channels(3);
        cv::split(src, channels);

        for (auto& channel : channels) {
            // For each channel, apply the 1D kernel
            cv::sepFilter2D(channel, channel, CV_32F, kernel1D, kernel1D);
            channel.convertTo(channel, CV_8U);
        }

        // After applying the filter to each channel separately, we can combine the 3 channels back and
        // return the final output RGB image
        cv::merge(channels, dst);
        cv::imwrite("gaussianout.jpg", dst);
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}

/*
* Apply Median Filtering with a predefined kernel size for a single-channel image
// */
void applyMedianFilterSingleChannel(const cv::Mat& src, cv::Mat& dst, int kernelSize) {
    // Don't accept even-sized kernels
    if (kernelSize % 2 == 0 || kernelSize <= 1) {
        throw std::invalid_argument("Kernel size must be an odd number greater than 1.");
    }
    int halfKernel = kernelSize / 2;
    // Window size for calculating median
    int windowSize = kernelSize * kernelSize;
    cv::Mat paddedImg;

    // Initialise the histogram
    int histogram[256] = {0};

    if (src.channels() == 1) {
        std::cout <<"Happening\n";
        // Initialise the output image to have the same dimensions and channels as the input
        dst.create(src.rows, src.cols, CV_8UC1);

        // If its a Greyscale image, there is only 1 channel so we can
        // apply the filter to the single channel directly
        // Add a border so that the corners can be filtered as well
        cv::copyMakeBorder(src, paddedImg, halfKernel, halfKernel, halfKernel, halfKernel, cv::BORDER_REFLECT); // mirror along the edges

        // Initialise the histogram
        for (int i = 0 ; i < kernelSize ; i++) {
            for (int j = 0 ; j < kernelSize ; j++) {
                int currentValue = paddedImg.at<uchar>(i, j);
                histogram[currentValue]++;
            }
        }

        // Add the first median
        dst.at<uchar>(0,0) = findMedian(histogram, windowSize, 256);

        // Iterate each pixel in the input image
        for (int i = halfKernel ; i < src.rows ; i++) {
            for (int j = halfKernel ; j < src.cols ; j++) {
                if (i == halfKernel && j == halfKernel) {
                    continue;
                }
                // std::cout <<i<<" "<<j<<"\n";
                if (j == halfKernel) { // Case 1: Starting a new row
                    std::fill(histogram, histogram + 256, 0);

                    // Refill the kernel
                    for (int m = -halfKernel ; m <= halfKernel ; m++) {
                        for (int n = -halfKernel ; n <= halfKernel ; n++) {
                            int currentValue = paddedImg.at<uchar>(m+i, n+j);
                            histogram[currentValue]++;
                        }
                    }
                } else { // Case 2: Normal sliding across the row
                    for (int k = -halfKernel ; k <= halfKernel ; k++) {
                        int newValue = paddedImg.at<uchar>(i + k, j + halfKernel);
                        histogram[newValue]++;
                        // Remove corresponding old element
                        int oldValue = paddedImg.at<uchar>(i + k, j - halfKernel - 1);
                        histogram[oldValue]--;
                    }
                }
                int median = findMedian(histogram, windowSize, 256);
                // std::cout << "Median at (" << i << ", " << j << ") is: " << median << std::endl;
                dst.at<uchar>(i - halfKernel, j - halfKernel) = median;
            }
        }
        cv::imwrite("medianfilteringout.jpg", dst);

    } else if (src.channels() == 3) {
        // If its a Greyscale image, there is only 1 channel so we can
        // apply the filter to the single channel directly
        // Add a border so that the corners can be filtered as well
        cv::copyMakeBorder(src, paddedImg, halfKernel, halfKernel, halfKernel, halfKernel, cv::BORDER_REFLECT); // mirror along the edges
        return;
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}

void applyMedianFilter(const cv::Mat& src, cv::Mat& dst, int kernelSize) {
    if (src.channels() == 1) {
        // Can directly apply the median filter to a greyscale image
        applyMedianFilterSingleChannel(src, dst, kernelSize);
    } else if (src.channels() == 3) {
        // Split the 3 channels in RGB
        std::vector<cv::Mat> channels(3);
        std::vector<cv::Mat> filteredChannels(3);
        cv::split(src, channels);

        // Apply the median filter to each channel
        for (int c = 0; c < 3; c++) {
            applyMedianFilterSingleChannel(channels[c], filteredChannels[c], kernelSize);
        }

        // Merge the filtered channels back into one image
        cv::merge(filteredChannels, dst);
        cv::imwrite("rgb_median_filter.jpg", dst);
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}