#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>

// Find the median for histogram (this is for the median filtering function)
int findMedian(int* histogram, int n, int range) {
    int count = 0;
    for (int i = 0 ; i < range ; i++) {
        count += histogram[i];
        if (count > n / 2) {
            return i;
        }
    }
    return 0;
}

cv::Mat generateCircularFilter(int numRows, int numCols, float backgroundValue, float circleValue, float radius) {
    // Create the mask with the default value
    cv::Mat mask = cv::Mat(numRows, numCols, CV_32F, backgroundValue);

    // Save the centre of the image
    int centreX = numCols / 2;
    int centreY = numRows / 2;

    // We don't want to manually check every single pixel in the image
    // So instead find the smallest bounding box that encloses the circle and check the pixels inside this instead
    // This square will have width = radius
    // Here we define the bounding box for the circle
    int minX = std::max(centreX-radius, 0.0f);
    int maxX = std::min(centreX+radius, static_cast<float>(numCols-1));
    int minY = std::max(centreY-radius, 0.0f);
    int maxY = std::min(centreY+radius, static_cast<float>(numRows-1));

    // Iterate all pixels in the bounding box
    for (int i = minY ; i <= maxY ; i++) { // rows
        for (int j = minX ; j <= maxX ; j++) { // columns
            // Calculate distance from the centre
            float distFromCentre = std::sqrt(std::pow(centreY-i, 2) + std::pow(centreX-j, 2));
            // Low Pass Filter: Only allow frequencies below the defined threshold
            if (distFromCentre <= radius) {
                mask.at<float>(i, j) = circleValue;
            }
        }
    }

    // Return the mask
    return mask;
}