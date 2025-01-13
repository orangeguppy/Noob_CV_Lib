#include <opencv2/opencv.hpp>
#include "core/simple_filters.h"
#include "core/preprocessing.h"

void applyLucasKanadeSingleChannel(const cv::Mat &src1, const cv::Mat &src2, cv::OutputArray &dst, int kernelSize) {
    // Raise an error if the input is empty
    if (src1.empty() || src2.empty()) {
        throw std::invalid_argument("Error: Empty image matrix!");
    }

    // Store higher-precision source images for calculation
    cv::Mat src1Float, src2Float;

    std::cout <<"Here\n";
    // Create an empty array to store the result
    dst.create(src1.size(), CV_32FC2);
    std::cout <<"Here2\n";

    // Calculate half kernel size and rows/columns
    int halfKernel = kernelSize / 2;
    int numRows = src1.rows;
    int numCols = src1.cols;

    // Apply Sobel to calculate X and Y gradients
    // We only use the first frame due to 2 assumptions:
    // 1) Brightness of each pixel remains constant as the pixel moves from the first frame to the next
    // 2) Rate of change of brightness is constant
    // Hence we can use the first frame to calculate Ix and Iy
    cv::Mat sobelImage1;
    cv::Mat gradX1, gradY1;
    std::cout <<"Done\n";

    // Calculate It by subtracting frame 1 from frame 2
    src1.convertTo(src1Float, CV_32F);
    src2.convertTo(src2Float, CV_32F);

    // applySobel(src1, gradX1, gradY1, sobelImage1);
    cv::Sobel(src1Float, gradX1, CV_32F, 1, 0, kernelSize);
    cv::Sobel(src1Float, gradY1, CV_32F, 0, 1, kernelSize);

    cv::Mat temporalGradient = src2Float - src1Float;
    std::cout <<"Done3\n";

    // Slide a window over the sobel image
    // Calculate optical flow using the neighbourhood of each pixel
    // First two for loops are meant to slide the centre pixel across the image
    for (int i = halfKernel ; i < numRows-halfKernel ; i++) {
        for (int j = halfKernel ; j < numCols-halfKernel ; j++) {
            // Use array slicing to extract the neighbourhood around the central pixel
            int startY = i - halfKernel;
            int startX = j - halfKernel;
            
            // Define the neighbourhood rectangle  and slice it
            cv::Rect neiBoundingBox(startX, startY, kernelSize, kernelSize);
            cv::Mat neiGradX1 = gradX1(neiBoundingBox).clone().reshape(1,1);
            cv::Mat neiGradY1 = gradY1(neiBoundingBox).clone().reshape(1,1);

            // Create the coordinate matrices
            cv::Mat neiGradX1Y1;
            cv::vconcat(neiGradX1, neiGradY1, neiGradX1Y1);
            neiGradX1Y1.convertTo(neiGradX1Y1, CV_32F);

            // Get Temporal Gradients in the neighbourhood
            cv::Mat neiTempGrad = temporalGradient(neiBoundingBox).clone().reshape(1,1);
            neiTempGrad.convertTo(neiTempGrad, CV_32F);
            cv::transpose(neiTempGrad, neiTempGrad); // Tranpose so it is 9x1

            // Tranpose to prepare for least squares
            cv::transpose(neiGradX1Y1, neiGradX1Y1);

            // Create terms of the equation
            cv::Mat AtA = neiGradX1Y1.t() * neiGradX1Y1;
            cv::Mat Atb = neiGradX1Y1.t() * neiTempGrad;

            // Solve for (u,v)
            cv::Mat uv;
            cv::solve(AtA, Atb, uv, cv::DECOMP_SVD);
            // std::cout << uv;
            uv.convertTo(uv, CV_32F);
            // if (uv.at<float>(0) > 0 && uv.at<float>(1) > 0) {
            //     std::cout << "Index " << i << " and " << j << "\n";
            // }
            dst.getMat().at<cv::Vec2f>(i, j) = cv::Vec2f(uv.at<float>(0), uv.at<float>(1));
        }
    }
}

void visualiseOpticalFlow(cv::Mat& flow) {
    // Step 1: Create a matrix to store the magnitude
    cv::Mat magnitude(flow.size(), CV_32F);

    // Step 2: Compute the magnitude for each pixel
    for (int i = 0; i < flow.rows; ++i) {
        for (int j = 0; j < flow.cols; ++j) {
            // Get the flow vector (u, v)
            cv::Vec2f flowVector = flow.at<cv::Vec2f>(i, j);
            float u = flowVector[0];
            float v = flowVector[1];

            // Compute magnitude
            magnitude.at<float>(i, j) = std::sqrt(u * u + v * v) * 100;

            if (magnitude.at<float>(i, j) > 0) {
                std::cout << "magnitude " << i << " and " << j <<"\n";
            }
        }
    }

    // Step 3: Normalize the magnitude to [0, 255] for visualization
    cv::Mat magnitudeNormalized;
    cv::normalize(magnitude, magnitudeNormalized, 0, 255, cv::NORM_MINMAX);
    // std::cout << magnitude;

    // Step 4: Convert to 8-bit for display
    magnitudeNormalized.convertTo(magnitudeNormalized, CV_8U);

    // Display the result
    writeImage("output/lucaskanadeout.jpg", magnitude);
}