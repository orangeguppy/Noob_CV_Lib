#ifndef CORE_UTILS_H
#define CORE_UTILS_H
#include <opencv2/opencv.hpp>

int findMedian(int* histogram, int n, int range);
cv::Mat generateCircularFilter(int numRows, int numCols, float backgroundValue, float circleValue, float radius);
#endif