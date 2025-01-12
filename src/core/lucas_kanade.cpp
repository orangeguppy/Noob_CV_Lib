// #include <opencv2/opencv.hpp>

// void applyLucasKanadeSingleChannel(const cv::Mat &src, cv::OutputArray &dst) {
//     // Raise an error if the input is empty
//     if (src.empty()) {
//         throw std::invalid_argument("Error: Empty image matrix!");
//     }

//     // Create an empty array to store the result
//     dst.create(src.rows, src.cols, CV_8UC1);

//     // Initialise the Gx and Gy finite difference operators
//     cv::Mat gradX, gradY;
//     cv::Mat Gx = (cv::Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
//     cv::Mat Gy = (cv::Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
    
//     Gx.convertTo(Gx, CV_32F);
//     Gy.convertTo(Gy, CV_32F);

//     // Convolve the original image with each Sobel filters to calculate the x and y gradients
//     manualFilter2D(src, gradX, Gx, 1, CV_32F, "SAME");
//     manualFilter2D(src, gradY, Gy, 1, CV_32F, "SAME");
    
//     // Now slide the window over the gradient maps


// }