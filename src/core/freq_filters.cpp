#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils/image_utils.h"
#include "utils/core_utils.h"
#include <omp.h>

#define PI 3.141592653589793

// Need to flip the quadrants for plotting thr frequencies after calculating them in applyFourierTransformSingleChannel
void fftShift(cv::Mat& magnitudeImage) {
    // Calculate the center of the image
    int centreX = magnitudeImage.cols / 2;
    int centreY = magnitudeImage.rows / 2;

    // Split the image into four quadrants
    cv::Mat quad0(magnitudeImage, cv::Rect(0, 0, centreX, centreY)); // Top-left, low frequencies
    cv::Mat quad1(magnitudeImage, cv::Rect(centreX, 0, centreX, centreY)); // Top-right, high frequencies in vertical direction
    cv::Mat quad2(magnitudeImage, cv::Rect(0, centreY, centreX, centreY)); // Bottom-left, high frequencies in horizontal directions
    cv::Mat quad3(magnitudeImage, cv::Rect(centreX, centreY, centreX, centreY)); // Bottom-right, high frequencies in both directions

    // Create temporary storage for swapping quadrants
    cv::Mat tmp;

    // Swap quadrants diagonally so that the lower frequencies are at the centre
    quad0.copyTo(tmp);
    quad3.copyTo(quad0);
    tmp.copyTo(quad3);

    quad1.copyTo(tmp);
    quad2.copyTo(quad1);
    tmp.copyTo(quad2);
}

// Plot the frequencies after calculating the Fourier Transform
void plotMagnitudeSpectrum(const cv::Mat& complexImage) {
    // Obtain the real and imaginary parts from the input
    std::vector<cv::Mat> channels(2);
    cv::split(complexImage, channels);

    // Calculate the magnitude using both parts
    cv::Mat magnitudeImage;
    cv::magnitude(channels[0], channels[1], magnitudeImage);

    // Use logarithmic scaling for better visibility
    magnitudeImage += cv::Scalar::all(1); // Avoid log(0)
    cv::log(magnitudeImage, magnitudeImage);

    // Shift the quadrants to ensure lower frequencies are in the middle
    fftShift(magnitudeImage);

    // Normalise to [0,255]
    cv::normalize(magnitudeImage, magnitudeImage, 0, 255, cv::NORM_MINMAX);
    magnitudeImage.convertTo(magnitudeImage, CV_8U);

    cv::applyColorMap(magnitudeImage, magnitudeImage, cv::COLORMAP_JET);

    // Save the result as an image
    cv::imwrite("magnitude_spectrum.jpg", magnitudeImage);
}

// Apply a Low-Pass or High-Pass Filter to the complex image
// Takes in 2-channel cv::Mat, the first channel is for the real component and the second is for the imaginary component
// It modifies the frequency components in-place
void applyFrequencyFilter(cv::Mat& complexImage, float radius, bool isLow) {
    // Obtain the real and imaginary parts from the input
    std::vector<cv::Mat> channels(2);
    cv::split(complexImage, channels);

    // Shift the quadrants to ensure lower frequencies are in the middle
    fftShift(complexImage);

    // Save the centre coordinates
    float centreX = complexImage.cols / 2;
    float centreY = complexImage.rows / 2;

    // Set parameters for the frequency filter
    float backgroundValue;
    float circleValue;

    if (isLow) {
        backgroundValue = 0.0f;
        circleValue = 1.0f;
    } else {
        backgroundValue = 1.0f;
        circleValue = 0.0f;
    }

    // Create a mask to convolve with the frequency components. Expand it to 2 channels so that it can be used to filter
    // the real and imaginary channels
    cv::Mat mask = generateCircularFilter(complexImage.rows, complexImage.cols, backgroundValue, circleValue, radius);
    // This is a vector containing two cv::Mat objects
    std::vector<cv::Mat> maskChannels(2, mask);

    // Create a new cv::Mat mask by merging the channels in maskChannels
    cv::Mat finalMask;
    cv::merge(maskChannels, finalMask);

    // Filter the image with the mask from the previous step
    cv::multiply(complexImage, finalMask, complexImage);

    // Shift the quadrants back
    fftShift(complexImage);
}

/*
 * @brief Performs Discrete Fourier Transform on an input image
 * @param src cv::Mat, the input image
 * @param dst cv::Mat, the output image
*/
void applyFourierTransformSingleChannel(const cv::Mat& src, cv::Mat& dst) {
    // Raise an error if the input is empty
    if (src.empty()) {
        throw std::invalid_argument("Error: Empty image matrix!");
    }

    // Convert to CV_32F for more precision
    cv::Mat copy = src.clone();
    copy.convertTo(copy, CV_32F); // Need this higher level of precision for complex numbers

    // Create an empty array to store the result
    dst.create(src.rows, src.cols, CV_32FC2);

    // Iterate all possible horizontal and vertical frequencies in the image
    // The first two for loops here are for iterating frequency components
    #pragma omp parallel for
    for (int p = 0 ; p < src.rows ; p++) {
        for (int q = 0 ; q < src.cols ; q++) {
            // Stores the value of the frequency component
            std::complex<float> freqComponent(0.0f, 0.0f);

            // These two for loops are for iterating all pixels to calculate the intensity of this frequency component
            // For all pixels
            for (int m = 0 ; m < src.rows ; m++) {
                for (int n = 0 ; n < src.cols ; n++) {
                    // Get the current value at this frequency
                    float currentValue = copy.at<float>(m, n);
                    float angle = -2.0f * PI * (static_cast<float>(p * m) / src.rows + static_cast<float>(q * n) / src.cols);

                    // 1.0f because unit circle
                    std::complex<float> expTerm = std::polar(1.0f, angle);

                    // Accumulate values for this frequency component
                    freqComponent += std::complex<float>(currentValue) * expTerm;
                }
            }

            // Save value for this frequency component
            dst.at<cv::Vec2f>(p, q)[0] = freqComponent.real(); // Real part
            dst.at<cv::Vec2f>(p, q)[1] = freqComponent.imag(); // Imaginary part
        }
    }
}

void invertFourierTransformSingleChannel(const cv::Mat& src, cv::Mat& dst) {
    // Raise an error if the input is empty
    if (src.empty()) {
        throw std::invalid_argument("Error: Empty image matrix!");
    }

    // Store the number of rows and columns
    int numRows = src.rows;
    int numCols = src.cols;

    // Allocate memory to dst
    dst.create(numRows, numCols, CV_32F);

    // Iterate each pixel in the image
    for (int m = 0 ; m < numRows ; m++) {
        for (int n = 0 ; n < numCols ; n++) {
            // Stores the value of the pixel intensity
            std::complex<float> pixelIntensity(0.0f, 0.0f);

            // Iterate each frequency component and see how much each component contributes to the original image
            for (int p = 0 ; p < numRows ; p++) {
                for (int q = 0 ; q < numCols ; q++) {
                    // Get the real and imaginary components
                    float realComponent = src.at<cv::Vec2f>(p, q)[0];
                    float imaginaryComponent = src.at<cv::Vec2f>(p, q)[1];
                    std::complex<float> freqComponent(realComponent, imaginaryComponent); // Combine them into a single variable for calculating stuff
                    
                    // Reverse the rotation to recover the original component
                    float angle = 2.0f * PI * (static_cast<float>(p * m) / src.rows + static_cast<float>(q * n) / src.cols);

                    // 1.0f because unit circle
                    std::complex<float> expTerm = std::polar(1.0f, angle);

                    // Accumulate pixel intensity from this frequncy component
                    pixelIntensity += freqComponent * expTerm;
                }
            }

            // Normalise using 1/MN
            dst.at<float>(m, n) = (1.0 / (numRows * numCols)) * pixelIntensity.real();
        }
    }
}