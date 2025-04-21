#include "cpu_filters.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <immintrin.h>
#include <algorithm>
#include <vector>

void gaussianBlurCPU(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    cv::GaussianBlur(input, output, cv::Size(kernelSize, kernelSize), sigma);
}
cv::Mat createGaussianKernel(int size, double sigma) {
    int halfSize = size / 2;
    cv::Mat kernel(size, size, CV_64F);
    double sum = 0.0;

    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            kernel.at<double>(i + halfSize, j + halfSize) =
                exp(-(i * i + j * j) / (2 * sigma * sigma)) / (8 * atan(1) * sigma * sigma);
            sum += kernel.at<double>(i + halfSize, j + halfSize);
        }
    }

    kernel /= sum;
    return kernel;
}

void gaussianBlurManual(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    // ������� ���� ������
    cv::Mat kernel = createGaussianKernel(kernelSize, sigma);

    int halfSize = kernelSize / 2;
    output = input.clone();

    // �������� �� ������� �������
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            double sumB = 0.0, sumG = 0.0, sumR = 0.0;

            // ��� ������� ������� ���������, ��� ���� �� ������� �� �������
            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    // ������� ��������, � �������� �������� ����
                    int x = i + ki;
                    int y = j + kj;

                    // �������� �� ����� �� ������� �����������, ���� �� �������� - ���������� �������� �������
                    if (x < 0) x = 0; // ������� ����
                    if (x >= input.rows) x = input.rows - 1; // ������� ���
                    if (y < 0) y = 0; // ������� �����
                    if (y >= input.cols) y = input.cols - 1; // ������� ������

                    sumB += input.at<cv::Vec3b>(x, y)[0] * kernel.at<double>(ki + halfSize, kj + halfSize);
                    sumG += input.at<cv::Vec3b>(x, y)[1] * kernel.at<double>(ki + halfSize, kj + halfSize);
                    sumR += input.at<cv::Vec3b>(x, y)[2] * kernel.at<double>(ki + halfSize, kj + halfSize);
                }
            }

            // ���������� ��������� � �������� �����������
            output.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(sumB);
            output.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(sumG);
            output.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(sumR);
        }
    }
}

void gaussianBlurSIMD(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    cv::Mat kernel = createGaussianKernel(kernelSize, sigma);

    int halfSize = kernelSize / 2;
    output = input.clone();

    // ��� ������� �������
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; j += 8) {  // ��� �� 8 ��������

            __m256 sumB = _mm256_setzero_ps();
            __m256 sumG = _mm256_setzero_ps();
            __m256 sumR = _mm256_setzero_ps();

            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int x = i + ki;
                    int y = j + kj;

                    // ���������� ��������� �����
                    x = std::max(0, std::min(x, input.rows - 1));
                    y = y < 0 ? -y : (y >= input.cols ? 2 * input.cols - y - 2 : y);

                    // ��������� ������ ��� 8 ��������
                    __m256 weights = _mm256_set1_ps(kernel.at<double>(ki + halfSize, kj + halfSize));

                    __m256 pixelB = _mm256_set_ps(input.at<cv::Vec3b>(x, std::min(y + 7, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 6, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 5, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 4, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 3, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 2, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 1, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, y)[0]);

                    __m256 pixelG = _mm256_set_ps(input.at<cv::Vec3b>(x, std::min(y + 7, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 6, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 5, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 4, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 3, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 2, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 1, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, y)[1]);

                    __m256 pixelR = _mm256_set_ps(input.at<cv::Vec3b>(x, std::min(y + 7, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 6, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 5, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 4, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 3, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 2, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 1, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, y)[2]);

                    sumB = _mm256_fmadd_ps(pixelB, weights, sumB);
                    sumG = _mm256_fmadd_ps(pixelG, weights, sumG);
                    sumR = _mm256_fmadd_ps(pixelR, weights, sumR);
                }
            }

            // ������� ��������� ������� ��� �������� ��������
            float tmpB[8], tmpG[8], tmpR[8];

            // ��������� �������� �� SIMD-��������� � �������
            _mm256_storeu_ps(tmpB, sumB);
            _mm256_storeu_ps(tmpG, sumG);
            _mm256_storeu_ps(tmpR, sumR);

            // ��������� ��������� � �����������
            for (int k = 0; k < 8; ++k) {
                if (j + k < input.cols) {  // ������ �� ������ �� �������
                    output.at<cv::Vec3b>(i, j + k)[0] = static_cast<uchar>(tmpB[k]);
                    output.at<cv::Vec3b>(i, j + k)[1] = static_cast<uchar>(tmpG[k]);
                    output.at<cv::Vec3b>(i, j + k)[2] = static_cast<uchar>(tmpR[k]);
                }
            }
        }
    }
}


void sobelFilterManual(const cv::Mat& input, cv::Mat& output) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    // ���� ������
    int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    output = cv::Mat::zeros(gray.size(), CV_8UC1);

    // ��������� ����
    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
            int gradX = 0, gradY = 0;

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int pixel = gray.at<uchar>(i + ki, j + kj);
                    gradX += sobelX[ki + 1][kj + 1] * pixel;
                    gradY += sobelY[ki + 1][kj + 1] * pixel;
                }
            }

            // ������������ �������� � �����������
            int magnitude = static_cast<int>(sqrt(gradX * gradX + gradY * gradY));
            output.at<uchar>(i, j) = static_cast<uchar>(std::min(magnitude, 255));
        }
    }
}


void sobelFilterSIMD(const cv::Mat& input, cv::Mat& output) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    // ���� ������
    const int sobelX[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    const int sobelY[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    output = cv::Mat::zeros(gray.size(), CV_8UC1);

    int width = gray.cols;
    int height = gray.rows;

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j <= width - 8; j += 8) {  // ��� 8 ��������
            __m256 gradX = _mm256_setzero_ps();
            __m256 gradY = _mm256_setzero_ps();

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int x = j + kj;

                    // ��������� 8 ��������
                    __m256 pixels = _mm256_cvtepi32_ps(
                        _mm256_cvtepu8_epi32(
                            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(&gray.at<uchar>(i + ki, x)))
                        )
                    );

                    // ��������� ���� ������ �� X � Y
                    __m256 weightX = _mm256_set1_ps(sobelX[ki + 1][kj + 1]);
                    __m256 weightY = _mm256_set1_ps(sobelY[ki + 1][kj + 1]);

                    gradX = _mm256_fmadd_ps(pixels, weightX, gradX);
                    gradY = _mm256_fmadd_ps(pixels, weightY, gradY);
                }
            }

            // ��������� �������� ������� (��������)
            __m256 gradMagnitude = _mm256_sqrt_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(gradX, gradX),
                    _mm256_mul_ps(gradY, gradY)
                )
            );

            __m256i result = _mm256_cvtps_epi32(gradMagnitude); // ���������� ������� � ����� ������
            __m256i clamped = _mm256_min_epi32(_mm256_max_epi32(result, _mm256_setzero_si256()), _mm256_set1_epi32(255));

            // ������������ � 8-������ �������� � ���������
            __m128i packed = _mm_packus_epi16(
                _mm_packs_epi32(_mm256_castsi256_si128(clamped), _mm256_extracti128_si256(clamped, 1)),
                _mm_setzero_si128()
            );

            _mm_storel_epi64(reinterpret_cast<__m128i*>(&output.at<uchar>(i, j)), packed);
        }
    }
}


void sobelFilterOpenCV(const cv::Mat& input, cv::Mat& output) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    cv::Mat gradX, gradY;
    cv::Mat absGradX, absGradY;

    // ��������� ������ ������ �� X � Y
    cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);

    // ����������� ��������� � ���������� ��������
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);

    // ��������� ��� ���������
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, output);
}

void cannyEdgeDetectorOpenCV(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    // ��������� �������� �����
    cv::Canny(gray, output, lowThreshold, highThreshold);
}


void cannyEdgeDetectorSIMD(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);

    cv::Mat gradX, gradY;
    cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3);


    cv::Mat magnitude, angle;
    cv::cartToPolar(gradX, gradY, magnitude, angle, true);

    // ������������ �������� ���������
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);

    // ���������� ������������ (non-maximum suppression)
    cv::Mat suppressed = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
            // ���������� ���� ���������
            float gradAngle = angle.at<float>(i, j);
            float mag = magnitude.at<float>(i, j);

            // ���������� ���� ��������� � ��������� ���� (0, 45, 90, 135 ��������)
            float neighbor1, neighbor2;
            if ((gradAngle >= 0 && gradAngle < CV_PI / 8) || (gradAngle >= 7 * CV_PI / 8)) {
                neighbor1 = magnitude.at<float>(i, j + 1);
                neighbor2 = magnitude.at<float>(i, j - 1);
            }
            else if (gradAngle >= CV_PI / 8 && gradAngle < 3 * CV_PI / 8) {
                neighbor1 = magnitude.at<float>(i + 1, j - 1);
                neighbor2 = magnitude.at<float>(i - 1, j + 1);
            }
            else if (gradAngle >= 3 * CV_PI / 8 && gradAngle < 5 * CV_PI / 8) {
                neighbor1 = magnitude.at<float>(i + 1, j);
                neighbor2 = magnitude.at<float>(i - 1, j);
            }
            else {
                neighbor1 = magnitude.at<float>(i - 1, j - 1);
                neighbor2 = magnitude.at<float>(i + 1, j + 1);
            }

            // ���� �������� � ������� ������� ������ �������, ��������� ���, ����� ������ 0
            if (mag >= neighbor1 && mag >= neighbor2) {
                suppressed.at<uchar>(i, j) = static_cast<uchar>(mag);
            }
            else {
                suppressed.at<uchar>(i, j) = 0;
            }
        }
    }


    output = cv::Mat::zeros(suppressed.size(), CV_8UC1);
    for (int i = 1; i < suppressed.rows - 1; ++i) {
        for (int j = 1; j < suppressed.cols - 1; ++j) {
            if (suppressed.at<uchar>(i, j) > highThreshold) {
                output.at<uchar>(i, j) = 255;  // �������� ��������
            }
            else if (suppressed.at<uchar>(i, j) >= lowThreshold) {
                // ���� �������� ����� ������ � ������� �������, ��������� �������
                if (suppressed.at<uchar>(i + 1, j) > highThreshold ||
                    suppressed.at<uchar>(i - 1, j) > highThreshold ||
                    suppressed.at<uchar>(i, j + 1) > highThreshold ||
                    suppressed.at<uchar>(i, j - 1) > highThreshold) {
                    output.at<uchar>(i, j) = 255;  // ������� ��� ����� �������
                }
                else {
                    output.at<uchar>(i, j) = 0;  // �� �������� ��������
                }
            }
        }
    }
}

void cannyEdgeDetectorManual(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);

    cv::Mat gradX, gradY;
    cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3);

    cv::Mat magnitude, angle;
    cv::cartToPolar(gradX, gradY, magnitude, angle, true);

    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);

    // ���������� ������������ (Non-Maximum Suppression)
    cv::Mat suppressed = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
            // ���������� ���� ���������
            float gradAngle = angle.at<float>(i, j);
            float mag = magnitude.at<float>(i, j);

            // ���������� ���� ��������� � ��������� ���� (0, 45, 90, 135 ��������)
            float neighbor1, neighbor2;
            if ((gradAngle >= 0 && gradAngle < 22.5) || (gradAngle >= 157.5 && gradAngle < 180)) {
                neighbor1 = magnitude.at<float>(i, j + 1);
                neighbor2 = magnitude.at<float>(i, j - 1);
            }
            else if (gradAngle >= 22.5 && gradAngle < 67.5) {
                neighbor1 = magnitude.at<float>(i + 1, j - 1);
                neighbor2 = magnitude.at<float>(i - 1, j + 1);
            }
            else if (gradAngle >= 67.5 && gradAngle < 112.5) {
                neighbor1 = magnitude.at<float>(i + 1, j);
                neighbor2 = magnitude.at<float>(i - 1, j);
            }
            else {
                neighbor1 = magnitude.at<float>(i - 1, j - 1);
                neighbor2 = magnitude.at<float>(i + 1, j + 1);
            }

            // ���� �������� � ������� ������� ������ �������, ��������� ���, ����� ������ 0
            if (mag >= neighbor1 && mag >= neighbor2) {
                suppressed.at<uchar>(i, j) = static_cast<uchar>(mag);
            }
            else {
                suppressed.at<uchar>(i, j) = 0;
            }
        }
    }


    output = cv::Mat::zeros(suppressed.size(), CV_8UC1);
    for (int i = 1; i < suppressed.rows - 1; ++i) {
        for (int j = 1; j < suppressed.cols - 1; ++j) {
            if (suppressed.at<uchar>(i, j) > highThreshold) {
                output.at<uchar>(i, j) = 255;  // �������� ��������
            }
            else if (suppressed.at<uchar>(i, j) >= lowThreshold) {
                // ���� �������� ����� ������ � ������� �������, ��������� �������
                if (suppressed.at<uchar>(i + 1, j) > highThreshold ||
                    suppressed.at<uchar>(i - 1, j) > highThreshold ||
                    suppressed.at<uchar>(i, j + 1) > highThreshold ||
                    suppressed.at<uchar>(i, j - 1) > highThreshold) {
                    output.at<uchar>(i, j) = 255;  // ������� ��� ����� �������
                }
                else {
                    output.at<uchar>(i, j) = 0;  // �� �������� ��������
                }
            }
        }
    }
}


void medianFilterManual(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    output = cv::Mat::zeros(gray.size(), gray.type());

    int halfSize = kernelSize / 2;
    int width = gray.cols;
    int height = gray.rows;

    // ��� ������� ������� �����������
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {

            std::vector<uchar> window;

            // ��� ������� ������� � ����
            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int x = i + ki;
                    int y = j + kj;

                    // ������������ ������� �� ����
                    if (x < 0) x = 0;
                    if (x >= height) x = height - 1;
                    if (y < 0) y = 0;
                    if (y >= width) y = width - 1;

                    // ��������� �������� ������� � ����
                    window.push_back(gray.at<uchar>(x, y));
                }
            }

            // ��������� ���� � �������� �������
            std::sort(window.begin(), window.end());
            output.at<uchar>(i, j) = window[window.size() / 2];
        }
    }
}


void medianFilterOpenCV(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    // ��������� ��������� ������
    cv::medianBlur(gray, output, kernelSize);
}


void medianFilterSIMD(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    cv::Mat gray;
    if (input.channels() == 3) {
        // ����������� ����������� � �������� ������
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    output = cv::Mat::zeros(gray.size(), gray.type());

    int halfSize = kernelSize / 2;
    int width = gray.cols;
    int height = gray.rows;

    // ��� ������� ������� �����������
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::vector<uchar> window;

            // ��� ������� ������� � ����
            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int x = i + ki;
                    int y = j + kj;

                    // ������������ ������� �� ����
                    if (x < 0) x = 0;
                    if (x >= height) x = height - 1;
                    if (y < 0) y = 0;
                    if (y >= width) y = width - 1;

                    // ��������� �������� ������� � ����
                    window.push_back(gray.at<uchar>(x, y));
                }
            }

            // ���������� SIMD ��� ��������� �������� �������� (��������� �� 8 ��������)
            __m256i pixel_values = _mm256_setzero_si256();
            for (size_t k = 0; k < window.size(); k += 8) {
                // ��������� SIMD-������� ���������
                pixel_values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&window[k]));
                // �������� ��������� ��������� ��������
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output.at<uchar>(i, j)), pixel_values);
            }

            // ���������� ����������� ���������� ��� ���������� �������
            std::sort(window.begin(), window.end());
            output.at<uchar>(i, j) = window[window.size() / 2];
        }
    }
}


void adjustBrightnessOpenCV(const cv::Mat& input, cv::Mat& output, int beta) {
    input.convertTo(output, -1, 1, beta);
}


void adjustBrightnessManual(const cv::Mat& input, cv::Mat& output, int beta) {
    output = input.clone();
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            for (int c = 0; c < input.channels(); ++c) {
                int pixelValue = input.at<cv::Vec3b>(i, j)[c];
                int newValue = std::min(std::max(static_cast<int>(pixelValue + beta), 0), 255);
                output.at<cv::Vec3b>(i, j)[c] = static_cast<uchar>(newValue);
            }
        }
    }
}


void adjustBrightnessSIMD(const cv::Mat& input, cv::Mat& output, int beta) {
    output = input.clone();

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; j += 8) {  // ������������ 8 �������� �� ���
            // ��������� 8 �������� (�� ������� ������ ��� RGB)
            __m256i pixelB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input.at<cv::Vec3b>(i, j)[0]));
            __m256i pixelG = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input.at<cv::Vec3b>(i, j)[1]));
            __m256i pixelR = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input.at<cv::Vec3b>(i, j)[2]));

            // ��������� beta � ������� ����������
            __m256i betaVec = _mm256_set1_epi8(static_cast<char>(beta));

            // ��������� ��������� ������� (����������� beta)
            __m256i newB = _mm256_adds_epu8(pixelB, betaVec);
            __m256i newG = _mm256_adds_epu8(pixelG, betaVec);
            __m256i newR = _mm256_adds_epu8(pixelR, betaVec);

            // ������������ �������� �� 0 �� 255
            newB = _mm256_min_epu8(newB, _mm256_set1_epi8(255));
            newB = _mm256_max_epu8(newB, _mm256_set1_epi8(0));

            newG = _mm256_min_epu8(newG, _mm256_set1_epi8(255));
            newG = _mm256_max_epu8(newG, _mm256_set1_epi8(0));

            newR = _mm256_min_epu8(newR, _mm256_set1_epi8(255));
            newR = _mm256_max_epu8(newR, _mm256_set1_epi8(0));

            // ��������� ���������� �������� � �������� �����������
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output.at<cv::Vec3b>(i, j)[0]), newB);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output.at<cv::Vec3b>(i, j)[1]), newG);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output.at<cv::Vec3b>(i, j)[2]), newR);
        }
    }
}


void adjustSaturationOpenCV(const cv::Mat& input, cv::Mat& output, double saturationFactor) {
    // ����������� ����������� � ������ HSV
    cv::Mat hsvImage;
    cv::cvtColor(input, hsvImage, cv::COLOR_BGR2HSV);

    // ��������� ����������� �� ������ (H, S, V)
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // �������� ������������ (����� S)
    hsvChannels[1] *= saturationFactor; // �������� �� ����������� ������������

    // ������������ �������� ������������ � ��������� [0, 255]
    cv::threshold(hsvChannels[1], hsvChannels[1], 255, 255, cv::THRESH_TRUNC);
    cv::threshold(hsvChannels[1], hsvChannels[1], 0, 0, cv::THRESH_TOZERO);

    // �������� ������ ������� � �����������
    cv::merge(hsvChannels, hsvImage);

    // ����������� ����������� ������� � ������ BGR
    cv::cvtColor(hsvImage, output, cv::COLOR_HSV2BGR);
}



void adjustSaturationManual(const cv::Mat& input, cv::Mat& output, double saturationFactor) {
    output = input.clone();

    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            int r = pixel[2], g = pixel[1], b = pixel[0];

            // ����������� �� RGB � HSV
            double rNorm = r / 255.0;
            double gNorm = g / 255.0;
            double bNorm = b / 255.0;

            double cmax = std::max({ rNorm, gNorm, bNorm });
            double cmin = std::min({ rNorm, gNorm, bNorm });
            double delta = cmax - cmin;

            double h = 0, s = 0, v = cmax;

            if (delta != 0) {
                if (cmax == rNorm) {
                    h = 60 * fmod(((gNorm - bNorm) / delta), 6);
                }
                else if (cmax == gNorm) {
                    h = 60 * (((bNorm - rNorm) / delta) + 2);
                }
                else if (cmax == bNorm) {
                    h = 60 * (((rNorm - gNorm) / delta) + 4);
                }

                if (h < 0) {
                    h += 360;
                }

                s = delta / cmax;
            }

            s *= saturationFactor;
            s = std::min(std::max(s, 0.0), 1.0);

            // ����������� ������� � RGB
            double c = v * s;
            double x = c * (1 - fabs(fmod(h / 60.0, 2) - 1));
            double m = v - c;

            double rp, gp, bp;
            if (h >= 0 && h < 60) {
                rp = c;
                gp = x;
                bp = 0;
            }
            else if (h >= 60 && h < 120) {
                rp = x;
                gp = c;
                bp = 0;
            }
            else if (h >= 120 && h < 180) {
                rp = 0;
                gp = c;
                bp = x;
            }
            else if (h >= 180 && h < 240) {
                rp = 0;
                gp = x;
                bp = c;
            }
            else if (h >= 240 && h < 300) {
                rp = x;
                gp = 0;
                bp = c;
            }
            else {
                rp = c;
                gp = 0;
                bp = x;
            }

            r = static_cast<int>((rp + m) * 255);
            g = static_cast<int>((gp + m) * 255);
            b = static_cast<int>((bp + m) * 255);

            output.at<cv::Vec3b>(i, j) = cv::Vec3b(b, g, r);
        }
    }
}



void adjustSaturationSIMD(const cv::Mat& input, cv::Mat& output, float saturationScale) {
    // ���������, ��� ������� ����������� ����� ������ CV_8UC3 (8-������ RGB)
    if (input.type() != CV_8UC3) {
        throw std::invalid_argument("Input image must be of type CV_8UC3 (8-bit RGB).");
    }

    // ��������� ������� ����������� � ��������
    output = input.clone();
    int width = input.cols;
    int height = input.rows;

    for (int i = 0; i < height; ++i) {
        const uchar* inputRow = input.ptr<uchar>(i); // ��������� �� ������ ������ �������� �����������
        uchar* outputRow = output.ptr<uchar>(i);     // ��������� �� ������ ������ ��������� �����������

        int j = 0;
        for (; j <= width - 8; j += 8) { // ������������ �� 8 �������� �� ���
            // ��������� 8 �������� (24 �����) ��� 8 ����� ����� (�� 3 ����� �� �������)
            __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&inputRow[j * 3]));

            // ��������� ������ B, G, R
            __m256i blue = _mm256_and_si256(pixels, _mm256_set1_epi32(0xFF)); // ������� ����
            __m256i green = _mm256_and_si256(_mm256_srli_epi32(pixels, 8), _mm256_set1_epi32(0xFF));
            __m256i red = _mm256_and_si256(_mm256_srli_epi32(pixels, 16), _mm256_set1_epi32(0xFF));

            // ����������� � float ��� ����������
            __m256 blue_f = _mm256_cvtepi32_ps(blue);
            __m256 green_f = _mm256_cvtepi32_ps(green);
            __m256 red_f = _mm256_cvtepi32_ps(red);

            // ��������� ������� ��� ������� �� �������
            __m256 intensity = _mm256_div_ps(
                _mm256_add_ps(_mm256_add_ps(red_f, green_f), blue_f), _mm256_set1_ps(3.0f));

            // ����������� ������������
            __m256 new_blue = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(blue_f, intensity), _mm256_set1_ps(saturationScale)), intensity);
            __m256 new_green = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(green_f, intensity), _mm256_set1_ps(saturationScale)), intensity);
            __m256 new_red = _mm256_add_ps(
                _mm256_mul_ps(_mm256_sub_ps(red_f, intensity), _mm256_set1_ps(saturationScale)), intensity);

            // ������������ �������� � ��������� [0, 255]
            new_blue = _mm256_min_ps(_mm256_max_ps(new_blue, _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));
            new_green = _mm256_min_ps(_mm256_max_ps(new_green, _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));
            new_red = _mm256_min_ps(_mm256_max_ps(new_red, _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));

            // ����������� ������� � ����� �����
            __m256i new_blue_i = _mm256_cvtps_epi32(new_blue);
            __m256i new_green_i = _mm256_cvtps_epi32(new_green);
            __m256i new_red_i = _mm256_cvtps_epi32(new_red);

            // �������� ������ ������� � ���� __m256i
            __m256i result = _mm256_or_si256(
                _mm256_or_si256(new_blue_i, _mm256_slli_epi32(new_green_i, 8)),
                _mm256_slli_epi32(new_red_i, 16));

            // ��������� ���������
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&outputRow[j * 3]), result);
        }

        // ��������� ���������� �������� � ������ (���� ��� ����)
        for (; j < width; ++j) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(i, j);
            float r = static_cast<float>(pixel[2]);
            float g = static_cast<float>(pixel[1]);
            float b = static_cast<float>(pixel[0]);

            float intensity = (r + g + b) / 3.0f;
            r = intensity + (r - intensity) * saturationScale;
            g = intensity + (g - intensity) * saturationScale;
            b = intensity + (b - intensity) * saturationScale;

            r = std::min(255.0f, std::max(0.0f, r));
            g = std::min(255.0f, std::max(0.0f, g));
            b = std::min(255.0f, std::max(0.0f, b));

            output.at<cv::Vec3b>(i, j) = cv::Vec3b(static_cast<uchar>(b), static_cast<uchar>(g), static_cast<uchar>(r));
        }
    }
}




void pixelateOpenCV(const cv::Mat& input, cv::Mat& output, int pixelSize) {
    output = input.clone();

    int width = input.cols;
    int height = input.rows;

    // �������� �� ����������� ������� ��������
    for (int i = 0; i < height; i += pixelSize) {
        for (int j = 0; j < width; j += pixelSize) {
            // ������������ ������� �������� ��� ����� ��������
            cv::Rect block(j, i, pixelSize, pixelSize);
            cv::Mat blockRegion = input(block);

            // ������� ������� �������� ���� �������� � �����
            cv::Scalar meanColor = cv::mean(blockRegion);

            // ����������� ���� ���� ���� ������� ������
            for (int y = i; y < std::min(i + pixelSize, height); ++y) {
                for (int x = j; x < std::min(j + pixelSize, width); ++x) {
                    output.at<cv::Vec3b>(y, x) = cv::Vec3b(meanColor[0], meanColor[1], meanColor[2]);
                }
            }
        }
    }
}




void pixelateSIMD(const cv::Mat& input, cv::Mat& output, int pixelSize) {
    output = input.clone();

    int width = input.cols;
    int height = input.rows;

    // ������� ��������� ����������� ��� ������� B, G � R
    cv::Mat channelB = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat channelG = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Mat channelR = cv::Mat::zeros(height, width, CV_8UC1);

    // ������������ ������ ����� ��������
    for (int channel = 0; channel < 3; ++channel) {
        cv::Mat currentChannel;

        if (channel == 0) currentChannel = channelB;
        else if (channel == 1) currentChannel = channelG;
        else currentChannel = channelR;

        // ������������ ����������� ������� ��������
        for (int i = 0; i < height; i += pixelSize) {
            for (int j = 0; j < width; j += pixelSize) {
                int blockEndX = std::min(j + pixelSize, width);
                int blockEndY = std::min(i + pixelSize, height);

                // ����� �������� ������ � �����
                int sum = 0;
                int pixelCount = 0;

                // ������������ ����� �������� � �����
                for (int y = i; y < blockEndY; ++y) {
                    for (int x = j; x < blockEndX; ++x) {
                        sum += input.at<cv::Vec3b>(y, x)[channel]; // ��������� ������ �����
                        pixelCount++;
                    }
                }

                // ��������� ������� �������� ����� ��� �����
                int meanValue = sum / pixelCount;

                // ����������� ���� ������� ���������
                for (int y = i; y < blockEndY; ++y) {
                    for (int x = j; x < blockEndX; ++x) {
                        currentChannel.at<uchar>(y, x) = static_cast<uchar>(meanValue);
                    }
                }
            }
        }
    }

    // ���������� ������ � ���� �����������
    std::vector<cv::Mat> channels = { channelB, channelG, channelR };
    cv::merge(channels, output);
}


void pixelateManual(const cv::Mat& input, cv::Mat& output, int pixelSize) {
    output = input.clone();

    int width = input.cols;
    int height = input.rows;

    // ������� ����������� ������� ��������
    for (int i = 0; i < height; i += pixelSize) {
        for (int j = 0; j < width; j += pixelSize) {
            int blockEndX = std::min(j + pixelSize, width);
            int blockEndY = std::min(i + pixelSize, height);

            // ����� ������ � �����
            int sumB = 0, sumG = 0, sumR = 0;
            int pixelCount = 0;

            // ������������ ������� �������� ����� � �����
            for (int y = i; y < blockEndY; ++y) {
                for (int x = j; x < blockEndX; ++x) {
                    cv::Vec3b pixel = input.at<cv::Vec3b>(y, x);
                    sumB += pixel[0]; // ����� �����
                    sumG += pixel[1]; // ������� �����
                    sumR += pixel[2]; // ������� �����
                    pixelCount++;
                }
            }

            // ��������� ������� ���� �����
            int meanB = sumB / pixelCount;
            int meanG = sumG / pixelCount;
            int meanR = sumR / pixelCount;

            // ����������� ���� ������� ������
            for (int y = i; y < blockEndY; ++y) {
                for (int x = j; x < blockEndX; ++x) {
                    output.at<cv::Vec3b>(y, x) = cv::Vec3b(meanB, meanG, meanR);
                }
            }
        }
    }
}


void sharpenOpenCV(const cv::Mat& input, cv::Mat& output) {
    // ������� ���� ��� ��������� ��������
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0);

    // ��������� ������ �������
    cv::filter2D(input, output, -1, kernel);
}



void sharpenManual(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();

    int width = input.cols;
    int height = input.rows;

    // ���� ��������� ��������
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    // ������������ ������ �������, ������� � 1 ����� �������� ������ �� �������
    for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            int sumB = 0, sumG = 0, sumR = 0;

            // �������� �� ���� 3x3
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    cv::Vec3b pixel = input.at<cv::Vec3b>(i + k, j + l);
                    int weight = kernel[k + 1][l + 1];

                    // ��������� �������� ������� � ������ ���� ����
                    sumB += pixel[0] * weight;
                    sumG += pixel[1] * weight;
                    sumR += pixel[2] * weight;
                }
            }

            // ������������ �������� � ��������� [0, 255]
            sumB = std::min(255, std::max(0, sumB));
            sumG = std::min(255, std::max(0, sumG));
            sumR = std::min(255, std::max(0, sumR));

            // ���������� ��������� � �������� �����������
            output.at<cv::Vec3b>(i, j) = cv::Vec3b(sumB, sumG, sumR);
        }
    }
}


void sharpenSIMD(const cv::Mat& input, cv::Mat& output) {
    output = input.clone();
    int width = input.cols;
    int height = input.rows;

    // ���� ��� ��������� ��������
    int kernel[3][3] = {
        { 0, -1,  0},
        {-1,  5, -1},
        { 0, -1,  0}
    };

    for (int i = 1; i < height - 1; ++i) {
        for (int j = 0; j <= width - 8; j += 8) {  // ������������ 8 �������� �� ���
            // ��������� 8 �������� ��� �������� ����
            __m256i pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input.at<cv::Vec3b>(i, j)));

            // ��������� ���������� ������� (�����, �������, �������)
            __m256i blue = _mm256_and_si256(pixels, _mm256_set1_epi32(0xFF)); // ����� �����
            __m256i green = _mm256_and_si256(_mm256_srli_epi32(pixels, 8), _mm256_set1_epi32(0xFF)); // ������� �����
            __m256i red = _mm256_and_si256(_mm256_srli_epi32(pixels, 16), _mm256_set1_epi32(0xFF)); // ������� �����

            // ����������� ���������� � ������ float
            __m256 blue_f = _mm256_cvtepi32_ps(blue);
            __m256 green_f = _mm256_cvtepi32_ps(green);
            __m256 red_f = _mm256_cvtepi32_ps(red);

            // ��� ������� ������� ������� ����� � ����������� ����
            __m256 sumB = _mm256_setzero_ps();
            __m256 sumG = _mm256_setzero_ps();
            __m256 sumR = _mm256_setzero_ps();

            // ������ �� ���� 3x3
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    int weight = kernel[k + 1][l + 1];
                    __m256 weight_val = _mm256_set1_ps(static_cast<float>(weight));

                    // ������� ������ ��� �������� ��������
                    int offsetX = j + l;
                    int offsetY = i + k;

                    // �������� �� ����� �� ������� �����������
                    if (offsetX >= 0 && offsetX < width && offsetY >= 0 && offsetY < height) {
                        __m256i neighbor_pixels = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input.at<cv::Vec3b>(offsetY, offsetX)));

                        // ��������� ���������� ������� ��� �������� ��������
                        __m256i neighbor_blue = _mm256_and_si256(neighbor_pixels, _mm256_set1_epi32(0xFF));
                        __m256i neighbor_green = _mm256_and_si256(_mm256_srli_epi32(neighbor_pixels, 8), _mm256_set1_epi32(0xFF));
                        __m256i neighbor_red = _mm256_and_si256(_mm256_srli_epi32(neighbor_pixels, 16), _mm256_set1_epi32(0xFF));

                        // ����������� ���������� �������� �������� � ������ float
                        __m256 neighbor_blue_f = _mm256_cvtepi32_ps(neighbor_blue);
                        __m256 neighbor_green_f = _mm256_cvtepi32_ps(neighbor_green);
                        __m256 neighbor_red_f = _mm256_cvtepi32_ps(neighbor_red);

                        // ��������� � ����� �������� � ������ ����
                        sumB = _mm256_fmadd_ps(neighbor_blue_f, weight_val, sumB);
                        sumG = _mm256_fmadd_ps(neighbor_green_f, weight_val, sumG);
                        sumR = _mm256_fmadd_ps(neighbor_red_f, weight_val, sumR);
                    }
                }
            }

            // ������������ �������� � ��������� [0, 255]
            sumB = _mm256_min_ps(_mm256_max_ps(sumB, _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));
            sumG = _mm256_min_ps(_mm256_max_ps(sumG, _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));
            sumR = _mm256_min_ps(_mm256_max_ps(sumR, _mm256_set1_ps(0.0f)), _mm256_set1_ps(255.0f));

            // ����������� ������� � ����� �����
            __m256i final_blue = _mm256_cvtps_epi32(sumB);
            __m256i final_green = _mm256_cvtps_epi32(sumG);
            __m256i final_red = _mm256_cvtps_epi32(sumR);

            // �������� ��������� ������� � �������
            __m256i result = _mm256_or_si256(
                _mm256_or_si256(final_blue, _mm256_slli_epi32(final_green, 8)),
                _mm256_slli_epi32(final_red, 16)
            );

            // ��������� ��������� � �������� �����������
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output.at<cv::Vec3b>(i, j)), result);
        }
    }
}
