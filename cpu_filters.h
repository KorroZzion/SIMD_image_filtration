#ifndef CPU_FILTERS_H
#define CPU_FILTERS_H
#pragma once
#include <opencv2/opencv.hpp>

void gaussianBlurCPU(const cv::Mat& input, cv::Mat& output, int kernelSize = 5, double sigma = 1.0);

void gaussianBlurManual(const cv::Mat& input, cv::Mat& output, int kernelSize = 5, double sigma = 1.0);

void gaussianBlurSIMD(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma);


void sobelFilterManual(const cv::Mat& input, cv::Mat& output);

void sobelFilterSIMD(const cv::Mat& input, cv::Mat& output);

void sobelFilterOpenCV(const cv::Mat& input, cv::Mat& output);



void cannyEdgeDetectorOpenCV(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold);

void cannyEdgeDetectorSIMD(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold);

void cannyEdgeDetectorManual(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold);



void medianFilterManual(const cv::Mat& input, cv::Mat& output, int kernelSize);

void medianFilterOpenCV(const cv::Mat& input, cv::Mat& output, int kernelSize);

void medianFilterSIMD(const cv::Mat& input, cv::Mat& output, int kernelSize);



void adjustBrightnessOpenCV(const cv::Mat& input, cv::Mat& output, int beta);

void adjustBrightnessManual(const cv::Mat& input, cv::Mat& output, int beta);

void adjustBrightnessSIMD(const cv::Mat& input, cv::Mat& output, int beta);



void adjustSaturationOpenCV(const cv::Mat& input, cv::Mat& output, double saturationFactor);

void adjustSaturationManual(const cv::Mat& input, cv::Mat& output, double saturationFactor);

void adjustSaturationSIMD(const cv::Mat& input, cv::Mat& output, float saturationFactor);



void pixelateOpenCV(const cv::Mat& input, cv::Mat& output, int pixelSize);

void pixelateSIMD(const cv::Mat& input, cv::Mat& output, int pixelSize);

void pixelateManual(const cv::Mat& input, cv::Mat& output, int pixelSize);



void sharpenOpenCV(const cv::Mat& input, cv::Mat& output);

void sharpenManual(const cv::Mat& input, cv::Mat& output);

void sharpenSIMD(const cv::Mat& input, cv::Mat& output);

#endif
