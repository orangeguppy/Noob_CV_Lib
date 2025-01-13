#ifndef LUCAS_KANADE_H
#define LUCAS_KANADE_H

#include <opencv2/opencv.hpp>

void applyLucasKanadeSingleChannel(const cv::Mat &src1, const cv::Mat &src2, cv::OutputArray &dst, int kernelSize);
void visualiseOpticalFlow(cv::Mat& flow);
#endif