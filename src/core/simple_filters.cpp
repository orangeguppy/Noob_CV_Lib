#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/image_utils.h"
#include "utils/core_utils.h"
#include "core/constants.h"

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
        cv::imwrite("output/gaussianout.jpg", dst);
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
        cv::imwrite("output/medianfilteringout.jpg", dst);
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}

/*
* Wrapper function for applyMedianFilterSingleChannel so that it can be extended to RGB images
// */
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
        cv::imwrite("output/rgb_median_filter.jpg", dst);
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}

/*
* Apply Bilateral Filtering with a predefined kernel size for a single-channel image
// */
void applyBilateralFilterSingleChannel(const cv::Mat& src, cv::Mat& dst, int kernelSize, float spatialSigma, float brightnessSigma) {
    // Don't accept even-sized kernels
    if (kernelSize % 2 == 0 || kernelSize <= 1) {
        throw std::invalid_argument("Kernel size must be an odd number greater than 1.");
    }
    int halfKernel = kernelSize / 2;

    cv::Mat paddedImg;

    // Normalising constants for the gaussian distributions
    float spatialGaussNormalisingConstant = 1.0 / (2 * PI * spatialSigma * spatialSigma);
    float brightnessGaussNormalisingConstant = 1.0 / (std::sqrt(2 * PI) * brightnessSigma);

    if (src.channels() == 1) {
        std::cout <<"Happening\n";
        // Initialise the output image to have the same dimensions and channels as the input
        dst.create(src.rows, src.cols, CV_8UC1);

        // Add a border so that the corners can be filtered as well
        cv::copyMakeBorder(src, paddedImg, halfKernel, halfKernel, halfKernel, halfKernel, cv::BORDER_REFLECT); // mirror along the edges

        // Iterate the sliding of the kernel centre across the whole image
        for (int i = halfKernel ; i < src.rows ; i++) {
            for (int j = halfKernel ; j < src.cols ; j++) {

                // Accumulate weights for the spatial and brightness gaussians
                float accumWeights = 0.0f;
                float unnormalisedOutput = 0.0f;

                // Loop through all neighbours of the central pixel
                for (int m = -halfKernel ; m <= halfKernel ; m++) {
                    for (int n = -halfKernel ; n <= halfKernel ; n++) {
                        // x and y distances from the centre of the kernel
                        int x_dist = std::abs(m);
                        int y_dist = std::abs(n);

                        // Calculate the difference in intensity between the current pixel and the centre pixel
                        int intensityDiff = paddedImg.at<uchar>(i,j) - paddedImg.at<uchar>(i+m,j+n);

                        // Calculate the weights for the spatial and brightness gaussians
                        float spatialExponent = -0.5 * (x_dist*x_dist + y_dist*y_dist) / (spatialSigma * spatialSigma);
                        float spatialWeight = spatialGaussNormalisingConstant * std::exp(spatialExponent);

                        float brightnessExponent = -0.5 * (intensityDiff*intensityDiff) / (brightnessSigma * brightnessSigma);
                        float brightnessWeight = brightnessGaussNormalisingConstant * std::exp(brightnessExponent);
                        accumWeights += spatialWeight * brightnessWeight;
                        unnormalisedOutput += paddedImg.at<uchar>(i+m,j+n) * spatialWeight * brightnessWeight;
                    }
                }
                int outputValue =  unnormalisedOutput / accumWeights;
                // std::cout << "Median at (" << i << ", " << j << ") is: " << median << std::endl;
                dst.at<uchar>(i - halfKernel, j - halfKernel) = outputValue;
            }
        }
        cv::imwrite("output/bilateralfilteringout.jpg", dst);
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}

void applyBilateralFiltering(const cv::Mat& src, cv::Mat& dst, int kernelSize, float spatialSigma, float brightnessSigma) {
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
            applyBilateralFilterSingleChannel(channels[c], filteredChannels[c], kernelSize, spatialSigma, brightnessSigma);
        }

        // Merge the filtered channels back into one image
        cv::merge(filteredChannels, dst);
        cv::imwrite("output/rgb_median_filter.jpg", dst);
    } else {
        throw std::invalid_argument("Unsupported number of channels in input image.");
    }
}