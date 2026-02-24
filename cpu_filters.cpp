#include "cpu_filters.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <immintrin.h>
#include <algorithm>
#include <vector>
#include <thread>


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


void gaussianBlurOpenCV(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    cv::GaussianBlur(input, output, cv::Size(kernelSize, kernelSize), sigma);
}


void gaussianBlurSIMDRange(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, int kernelSize, int startRow, int endRow) {
    int halfSize = kernelSize / 2;

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < input.cols; j += 8) {

            __m256 sumB = _mm256_setzero_ps();
            __m256 sumG = _mm256_setzero_ps();
            __m256 sumR = _mm256_setzero_ps();

            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {

                    int x = i + ki;
                    int y = j + kj;

                    x = std::max(0, std::min(x, input.rows - 1));
                    y = y < 0 ? -y : (y >= input.cols ? 2 * input.cols - y - 2 : y);

                    __m256 weight = _mm256_set1_ps(
                        static_cast<float>(kernel.at<double>(ki + halfSize, kj + halfSize))
                        );

                    __m256 pixelB = _mm256_set_ps(
                        input.at<cv::Vec3b>(x, std::min(y + 7, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 6, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 5, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 4, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 3, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 2, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, std::min(y + 1, input.cols - 1))[0],
                        input.at<cv::Vec3b>(x, y)[0]
                        );

                    __m256 pixelG = _mm256_set_ps(
                        input.at<cv::Vec3b>(x, std::min(y + 7, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 6, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 5, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 4, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 3, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 2, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, std::min(y + 1, input.cols - 1))[1],
                        input.at<cv::Vec3b>(x, y)[1]
                        );

                    __m256 pixelR = _mm256_set_ps(
                        input.at<cv::Vec3b>(x, std::min(y + 7, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 6, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 5, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 4, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 3, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 2, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, std::min(y + 1, input.cols - 1))[2],
                        input.at<cv::Vec3b>(x, y)[2]
                        );

                    sumB = _mm256_fmadd_ps(pixelB, weight, sumB);
                    sumG = _mm256_fmadd_ps(pixelG, weight, sumG);
                    sumR = _mm256_fmadd_ps(pixelR, weight, sumR);
                }
            }

            float tmpB[8], tmpG[8], tmpR[8];
            _mm256_storeu_ps(tmpB, sumB);
            _mm256_storeu_ps(tmpG, sumG);
            _mm256_storeu_ps(tmpR, sumR);

            for (int k = 0; k < 8 && j + k < input.cols; ++k) {
                output.at<cv::Vec3b>(i, j + k) = cv::Vec3b(
                    static_cast<uchar>(std::clamp(tmpB[k], 0.0f, 255.0f)),
                    static_cast<uchar>(std::clamp(tmpG[k], 0.0f, 255.0f)),
                    static_cast<uchar>(std::clamp(tmpR[k], 0.0f, 255.0f))
                    );
            }
        }
    }
}


void gaussianBlurSIMD(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    CV_Assert(input.type() == CV_8UC3);

    output.create(input.size(), input.type());

    cv::Mat kernel = createGaussianKernel(kernelSize, sigma);

    unsigned int threadsCount = std::thread::hardware_concurrency();
    if (threadsCount == 0)
        threadsCount = 4;

    int rowsPerThread = input.rows / threadsCount;

    std::vector<std::thread> threads;
    int startRow = 0;

    for (unsigned int t = 0; t < threadsCount; ++t) {
        int endRow = (t == threadsCount - 1)
        ? input.rows
        : startRow + rowsPerThread;

        threads.emplace_back(
            gaussianBlurSIMDRange,
            std::cref(input),
            std::ref(output),
            std::cref(kernel),
            kernelSize,
            startRow,
            endRow
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
}


void gaussianBlurManualRange(const cv::Mat& input, cv::Mat& output, const cv::Mat& kernel, int startRow, int endRow) {
    int kernelSize = kernel.rows;
    int halfSize = kernelSize / 2;

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 0; j < input.cols; ++j) {

            double sumB = 0.0, sumG = 0.0, sumR = 0.0;

            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {

                    int x = i + ki;
                    int y = j + kj;

                    if (x < 0) x = 0;
                    if (x >= input.rows) x = input.rows - 1;
                    if (y < 0) y = 0;
                    if (y >= input.cols) y = input.cols - 1;

                    double k = kernel.at<double>(ki + halfSize, kj + halfSize);

                    const cv::Vec3b& pix = input.at<cv::Vec3b>(x, y);

                    sumB += pix[0] * k;
                    sumG += pix[1] * k;
                    sumR += pix[2] * k;
                }
            }

            output.at<cv::Vec3b>(i, j) = cv::Vec3b(
                static_cast<uchar>(sumB),
                static_cast<uchar>(sumG),
                static_cast<uchar>(sumR)
                );
        }
    }
}


void gaussianBlurManual( const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    CV_Assert(input.type() == CV_8UC3);

    output.create(input.size(), input.type());

    cv::Mat kernel = createGaussianKernel(kernelSize, sigma);

    unsigned int threadsCount = std::thread::hardware_concurrency();
    if (threadsCount == 0)
        threadsCount = 4; // запасной вариант

    int rowsPerThread = input.rows / threadsCount;

    std::vector<std::thread> threads;

    int startRow = 0;
    for (unsigned int t = 0; t < threadsCount; ++t) {
        int endRow = (t == threadsCount - 1)
        ? input.rows
        : startRow + rowsPerThread;

        threads.emplace_back(
            gaussianBlurManualRange,
            std::cref(input),
            std::ref(output),
            std::cref(kernel),
            startRow,
            endRow
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
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

    // Применяем фильтр Собеля по X и Y
    cv::Sobel(gray, gradX, CV_16S, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_16S, 0, 1, 3);

    // Преобразуем градиенты в абсолютные значения
    cv::convertScaleAbs(gradX, absGradX);
    cv::convertScaleAbs(gradY, absGradY);

    // Суммируем оба градиента
    cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, output);
}


void sobelFilterSIMDRange(const cv::Mat& gray, cv::Mat& output, int startRow, int endRow) {
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

    int width = gray.cols;

    startRow = std::max(startRow, 1);
    endRow   = std::min(endRow, gray.rows - 1);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 1; j <= width - 8; j += 8) {

            __m256 gradX = _mm256_setzero_ps();
            __m256 gradY = _mm256_setzero_ps();

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {

                    const uchar* src =
                        &gray.at<uchar>(i + ki, j + kj);

                    __m256 pixels = _mm256_cvtepi32_ps(
                        _mm256_cvtepu8_epi32(
                            _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src))
                            )
                        );

                    __m256 wX = _mm256_set1_ps(sobelX[ki + 1][kj + 1]);
                    __m256 wY = _mm256_set1_ps(sobelY[ki + 1][kj + 1]);

                    gradX = _mm256_fmadd_ps(pixels, wX, gradX);
                    gradY = _mm256_fmadd_ps(pixels, wY, gradY);
                }
            }

            __m256 magnitude = _mm256_sqrt_ps(
                _mm256_add_ps(
                    _mm256_mul_ps(gradX, gradX),
                    _mm256_mul_ps(gradY, gradY)
                    )
                );

            __m256i result = _mm256_cvtps_epi32(magnitude);
            __m256i clamped = _mm256_min_epi32(
                _mm256_max_epi32(result, _mm256_setzero_si256()),
                _mm256_set1_epi32(255)
                );

            __m128i packed = _mm_packus_epi16(
                _mm_packs_epi32(
                    _mm256_castsi256_si128(clamped),
                    _mm256_extracti128_si256(clamped, 1)
                    ),
                _mm_setzero_si128()
                );

            _mm_storel_epi64(
                reinterpret_cast<__m128i*>(&output.at<uchar>(i, j)),
                packed
                );
        }
    }
}


void sobelFilterSIMD(const cv::Mat& input, cv::Mat& output) {
    cv::Mat gray;

    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    output = cv::Mat::zeros(gray.size(), CV_8UC1);

    unsigned int threadsCount = std::thread::hardware_concurrency();
    if (threadsCount == 0)
        threadsCount = 4;

    int rowsPerThread = gray.rows / threadsCount;

    std::vector<std::thread> threads;
    int startRow = 0;

    for (unsigned int t = 0; t < threadsCount; ++t) {
        int endRow = (t == threadsCount - 1)
        ? gray.rows
        : startRow + rowsPerThread;

        threads.emplace_back(
            sobelFilterSIMDRange,
            std::cref(gray),
            std::ref(output),
            startRow,
            endRow
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
}


void sobelFilterManualRange(const cv::Mat& gray, cv::Mat& output, int startRow, int endRow) {
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

    // Не трогаем границы
    startRow = std::max(startRow, 1);
    endRow   = std::min(endRow, gray.rows - 1);

    for (int i = startRow; i < endRow; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {

            int gradX = 0;
            int gradY = 0;

            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    int pixel = gray.at<uchar>(i + ki, j + kj);
                    gradX += sobelX[ki + 1][kj + 1] * pixel;
                    gradY += sobelY[ki + 1][kj + 1] * pixel;
                }
            }

            int magnitude = static_cast<int>(
                std::sqrt(gradX * gradX + gradY * gradY)
                );

            output.at<uchar>(i, j) =
                static_cast<uchar>(std::min(magnitude, 255));
        }
    }
}


void sobelFilterManual(const cv::Mat& input, cv::Mat& output) {
    cv::Mat gray;

    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    output = cv::Mat::zeros(gray.size(), CV_8UC1);

    unsigned int threadsCount = std::thread::hardware_concurrency();
    if (threadsCount == 0)
        threadsCount = 4;

    int rowsPerThread = gray.rows / threadsCount;

    std::vector<std::thread> threads;
    int startRow = 0;

    for (unsigned int t = 0; t < threadsCount; ++t) {
        int endRow = (t == threadsCount - 1)
        ? gray.rows
        : startRow + rowsPerThread;

        threads.emplace_back(
            sobelFilterManualRange,
            std::cref(gray),
            std::ref(output),
            startRow,
            endRow
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
}




void cannyEdgeDetectorOpenCV(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold) {
    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    // Применяем оператор Канни
    cv::Canny(gray, output, lowThreshold, highThreshold);
}


static inline uchar quantizeAngle(float angle) {
    if ((angle >= 0 && angle < 22.5f) || (angle >= 157.5f))
        return 0;
    else if (angle < 67.5f)
        return 45;
    else if (angle < 112.5f)
        return 90;
    else
        return 135;
}


void cannyEdgeDetectorSIMD(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.4);

    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);

    cv::Mat magnitude, angle;
    cv::cartToPolar(gradX, gradY, magnitude, angle, true);
    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);

    cv::Mat direction(angle.size(), CV_8UC1);
    for (int i = 0; i < angle.rows; ++i)
        for (int j = 0; j < angle.cols; ++j)
            direction.at<uchar>(i, j) = quantizeAngle(angle.at<float>(i, j));

    cv::Mat suppressed = cv::Mat::zeros(gray.size(), CV_8UC1);

    auto workerSIMD = [&](int y0, int y1) {
        for (int i = y0; i < y1; ++i) {
            for (int j = 1; j <= gray.cols - 9; j += 8) {

                __m256 mag = _mm256_loadu_ps(&magnitude.at<float>(i, j));
                __m256 left  = _mm256_loadu_ps(&magnitude.at<float>(i, j - 1));
                __m256 right = _mm256_loadu_ps(&magnitude.at<float>(i, j + 1));

                __m256 m1 = _mm256_cmp_ps(mag, left, _CMP_GE_OQ);
                __m256 m2 = _mm256_cmp_ps(mag, right, _CMP_GE_OQ);
                __m256 mask = _mm256_and_ps(m1, m2);

                __m256 res = _mm256_blendv_ps(
                    _mm256_setzero_ps(), mag, mask
                    );

                __m256i ri = _mm256_cvtps_epi32(res);
                ri = _mm256_min_epi32(ri, _mm256_set1_epi32(255));
                ri = _mm256_max_epi32(ri, _mm256_setzero_si256());

                __m128i lo = _mm256_castsi256_si128(ri);
                __m128i hi = _mm256_extracti128_si256(ri, 1);
                __m128i pack = _mm_packus_epi16(
                    _mm_packs_epi32(lo, hi),
                    _mm_setzero_si128()
                    );

                _mm_storel_epi64(
                    reinterpret_cast<__m128i*>(&suppressed.at<uchar>(i, j)),
                    pack
                    );
            }
        }
    };

    int threads = std::thread::hardware_concurrency();
    int step = gray.rows / threads;
    std::vector<std::thread> pool;

    for (int t = 0; t < threads; ++t) {
        int y0 = t * step + 1;
        int y1 = (t == threads - 1) ? gray.rows - 1 : y0 + step;
        pool.emplace_back(workerSIMD, y0, y1);
    }
    for (auto& th : pool) th.join();

    output = cv::Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
            uchar v = suppressed.at<uchar>(i, j);
            if (v > highThreshold)
                output.at<uchar>(i, j) = 255;
            else if (v >= lowThreshold) {
                if (suppressed.at<uchar>(i + 1, j) > highThreshold ||
                    suppressed.at<uchar>(i - 1, j) > highThreshold ||
                    suppressed.at<uchar>(i, j + 1) > highThreshold ||
                    suppressed.at<uchar>(i, j - 1) > highThreshold)
                    output.at<uchar>(i, j) = 255;
            }
        }
    }
}


void cannyEdgeDetectorManual(const cv::Mat& input, cv::Mat& output, double lowThreshold, double highThreshold) {
    cv::Mat gray;
    if (input.channels() == 3)
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    else
        gray = input.clone();

    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 1.4);

    cv::Mat gradX, gradY;
    cv::Sobel(gray, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gradY, CV_32F, 0, 1, 3);

    cv::Mat magnitude, angle;
    cv::cartToPolar(gradX, gradY, magnitude, angle, true);

    cv::normalize(magnitude, magnitude, 0, 255, cv::NORM_MINMAX);

    cv::Mat direction(angle.size(), CV_8UC1);
    for (int i = 0; i < angle.rows; ++i)
        for (int j = 0; j < angle.cols; ++j)
            direction.at<uchar>(i, j) = quantizeAngle(angle.at<float>(i, j));

    cv::Mat suppressed = cv::Mat::zeros(gray.size(), CV_8UC1);

    auto worker = [&](int y0, int y1) {
        for (int i = y0; i < y1; ++i) {
            for (int j = 1; j < gray.cols - 1; ++j) {
                float mag = magnitude.at<float>(i, j);
                uchar dir = direction.at<uchar>(i, j);

                float n1 = 0, n2 = 0;

                if (dir == 0) {
                    n1 = magnitude.at<float>(i, j - 1);
                    n2 = magnitude.at<float>(i, j + 1);
                }
                else if (dir == 45) {
                    n1 = magnitude.at<float>(i - 1, j + 1);
                    n2 = magnitude.at<float>(i + 1, j - 1);
                }
                else if (dir == 90) {
                    n1 = magnitude.at<float>(i - 1, j);
                    n2 = magnitude.at<float>(i + 1, j);
                }
                else {
                    n1 = magnitude.at<float>(i - 1, j - 1);
                    n2 = magnitude.at<float>(i + 1, j + 1);
                }

                if (mag >= n1 && mag >= n2)
                    suppressed.at<uchar>(i, j) = static_cast<uchar>(mag);
            }
        }
    };

    int threads = std::thread::hardware_concurrency();
    int step = gray.rows / threads;
    std::vector<std::thread> pool;

    for (int t = 0; t < threads; ++t) {
        int y0 = t * step + 1;
        int y1 = (t == threads - 1) ? gray.rows - 1 : y0 + step;
        pool.emplace_back(worker, y0, y1);
    }
    for (auto& th : pool) th.join();

    output = cv::Mat::zeros(gray.size(), CV_8UC1);

    for (int i = 1; i < gray.rows - 1; ++i) {
        for (int j = 1; j < gray.cols - 1; ++j) {
            uchar v = suppressed.at<uchar>(i, j);
            if (v > highThreshold)
                output.at<uchar>(i, j) = 255;
            else if (v >= lowThreshold) {
                if (suppressed.at<uchar>(i + 1, j) > highThreshold ||
                    suppressed.at<uchar>(i - 1, j) > highThreshold ||
                    suppressed.at<uchar>(i, j + 1) > highThreshold ||
                    suppressed.at<uchar>(i, j - 1) > highThreshold)
                    output.at<uchar>(i, j) = 255;
            }
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

    // Применяем медианный фильтр
    cv::medianBlur(gray, output, kernelSize);
}


void medianFilterSIMD(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    cv::Mat gray;
    if (input.channels() == 3) {
        // Преобразуем изображение в градации серого
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    else {
        gray = input.clone();
    }

    output = cv::Mat::zeros(gray.size(), gray.type());

    int halfSize = kernelSize / 2;
    int width = gray.cols;
    int height = gray.rows;

    // Для каждого пикселя изображения
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            std::vector<uchar> window;

            // Для каждого пикселя в ядре
            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int x = i + ki;
                    int y = j + kj;

                    // Ограничиваем пиксели на краю
                    if (x < 0) x = 0;
                    if (x >= height) x = height - 1;
                    if (y < 0) y = 0;
                    if (y >= width) y = width - 1;

                    // Добавляем значение пикселя в окно
                    window.push_back(gray.at<uchar>(x, y));
                }
            }

            // Используем SIMD для ускорения загрузки значений (обработка по 8 пикселей)
            __m256i pixel_values = _mm256_setzero_si256();
            for (size_t k = 0; k < window.size(); k += 8) {
                // Заполняем SIMD-регистр пикселями
                pixel_values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&window[k]));
                // Проводим обработку векторных значений
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output.at<uchar>(i, j)), pixel_values);
            }

            // Используем стандартную сортировку для нахождения медианы
            std::sort(window.begin(), window.end());
            output.at<uchar>(i, j) = window[window.size() / 2];
        }
    }
}


static void medianFilterManualRange(const cv::Mat& gray, cv::Mat& output, int kernelSize, int yBegin, int yEnd) {
    int halfSize = kernelSize / 2;
    int width = gray.cols;
    int height = gray.rows;

    std::vector<uchar> window;
    window.reserve(kernelSize * kernelSize);

    for (int i = yBegin; i < yEnd; ++i) {
        for (int j = 0; j < width; ++j) {

            window.clear();

            for (int ki = -halfSize; ki <= halfSize; ++ki) {
                for (int kj = -halfSize; kj <= halfSize; ++kj) {
                    int x = i + ki;
                    int y = j + kj;

                    if (x < 0) x = 0;
                    if (x >= height) x = height - 1;
                    if (y < 0) y = 0;
                    if (y >= width) y = width - 1;

                    window.push_back(gray.at<uchar>(x, y));
                }
            }

            std::nth_element(
                window.begin(),
                window.begin() + window.size() / 2,
                window.end()
                );

            output.at<uchar>(i, j) = window[window.size() / 2];
        }
    }
}


void medianFilterManual(const cv::Mat& input, cv::Mat& output, int kernelSize) {
    CV_Assert(kernelSize % 2 == 1);

    cv::Mat gray;
    if (input.channels() == 3) {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }

    output = cv::Mat::zeros(gray.size(), gray.type());

    int height = gray.rows;
    int threads = std::thread::hardware_concurrency();
    if (threads == 0) threads = 4;

    int step = height / threads;
    std::vector<std::thread> workers;

    for (int t = 0; t < threads; ++t) {
        int yBegin = t * step;
        int yEnd = (t == threads - 1) ? height : yBegin + step;

        workers.emplace_back(
            medianFilterManualRange,
            std::cref(gray),
            std::ref(output),
            kernelSize,
            yBegin,
            yEnd
            );
    }

    for (auto& th : workers) {
        th.join();
    }
}




void adjustBrightnessOpenCV(const cv::Mat& input, cv::Mat& output, int beta) {
    input.convertTo(output, -1, 1, beta);
}


static void adjustBrightnessSIMDRange(const cv::Mat& input, cv::Mat& output, int beta, int rowBegin, int rowEnd) {
    const int colsBytes = input.cols * input.channels();

    __m256i betaVec = _mm256_set1_epi8(static_cast<char>(beta));

    for (int i = rowBegin; i < rowEnd; ++i) {
        const uchar* src = input.ptr<uchar>(i);
        uchar* dst = output.ptr<uchar>(i);

        int j = 0;

        // SIMD-блоки по 32 байта
        for (; j <= colsBytes - 32; j += 32) {
            __m256i pixels = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(src + j)
                );

            __m256i result;

            if (beta >= 0) {
                result = _mm256_adds_epu8(pixels, betaVec);
            } else {
                result = _mm256_subs_epu8(pixels, _mm256_abs_epi8(betaVec));
            }

            _mm256_storeu_si256(
                reinterpret_cast<__m256i*>(dst + j),
                result
                );
        }

        // Хвост
        for (; j < colsBytes; ++j) {
            int val = src[j] + beta;
            dst[j] = static_cast<uchar>(std::min(std::max(val, 0), 255));
        }
    }
}


void adjustBrightnessSIMD(const cv::Mat& input, cv::Mat& output, int beta) {
    CV_Assert(input.type() == CV_8UC3);

    output.create(input.size(), input.type());

    const unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    int rowsPerThread = input.rows / threadCount;
    int currentRow = 0;

    for (unsigned int t = 0; t < threadCount; ++t) {
        int startRow = currentRow;
        int endRow = (t == threadCount - 1)
                         ? input.rows
                         : startRow + rowsPerThread;

        threads.emplace_back(
            adjustBrightnessSIMDRange,
            std::cref(input),
            std::ref(output),
            beta,
            startRow,
            endRow
            );

        currentRow = endRow;
    }

    for (auto& th : threads) {
        th.join();
    }
}


static void adjustBrightnessManualRange(const cv::Mat& input, cv::Mat& output, int beta, int rowBegin, int rowEnd) {
    const int cols = input.cols;
    const int channels = input.channels();

    for (int i = rowBegin; i < rowEnd; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int c = 0; c < channels; ++c) {
                int val = input.at<cv::Vec3b>(i, j)[c] + beta;
                output.at<cv::Vec3b>(i, j)[c] =
                    static_cast<uchar>(std::min(std::max(val, 0), 255));
            }
        }
    }
}


void adjustBrightnessManual(const cv::Mat& input, cv::Mat& output, int beta) {
    CV_Assert(input.type() == CV_8UC3);

    output.create(input.size(), input.type());

    const unsigned int threadCount =
        std::max(1u, std::thread::hardware_concurrency());

    std::vector<std::thread> threads;
    threads.reserve(threadCount);

    int rowsPerThread = input.rows / threadCount;
    int currentRow = 0;

    for (unsigned int t = 0; t < threadCount; ++t) {
        int startRow = currentRow;
        int endRow = (t == threadCount - 1)
                         ? input.rows
                         : startRow + rowsPerThread;

        threads.emplace_back(
            adjustBrightnessManualRange,
            std::cref(input),
            std::ref(output),
            beta,
            startRow,
            endRow
            );

        currentRow = endRow;
    }

    for (auto& th : threads) {
        th.join();
    }
}




void adjustSaturationOpenCV(const cv::Mat& input, cv::Mat& output, double saturationFactor) {
    // Преобразуем изображение в модель HSV
    cv::Mat hsvImage;
    cv::cvtColor(input, hsvImage, cv::COLOR_BGR2HSV);

    // Разделяем изображение на каналы (H, S, V)
    std::vector<cv::Mat> hsvChannels;
    cv::split(hsvImage, hsvChannels);

    // Изменяем насыщенность (канал S)
    hsvChannels[1] *= saturationFactor; // Умножаем на коэффициент насыщенности

    // Ограничиваем значения насыщенности в диапазоне [0, 255]
    cv::threshold(hsvChannels[1], hsvChannels[1], 255, 255, cv::THRESH_TRUNC);
    cv::threshold(hsvChannels[1], hsvChannels[1], 0, 0, cv::THRESH_TOZERO);

    // Собираем каналы обратно в изображение
    cv::merge(hsvChannels, hsvImage);

    // Преобразуем изображение обратно в модель BGR
    cv::cvtColor(hsvImage, output, cv::COLOR_HSV2BGR);
}


void adjustSaturationSIMDRange(const cv::Mat& input, cv::Mat& output, int startRow, int endRow, float saturationScale) {
    const __m256 scale = _mm256_set1_ps(saturationScale);
    const __m256 zero  = _mm256_set1_ps(0.0f);
    const __m256 maxv  = _mm256_set1_ps(255.0f);
    const __m256 inv3  = _mm256_set1_ps(1.0f / 3.0f);

    for (int y = startRow; y < endRow; ++y) {
        const uchar* src = input.ptr<uchar>(y);
        uchar* dst = output.ptr<uchar>(y);

        int x = 0;

        // SIMD-блок: 8 пикселей
        for (; x <= input.cols - 8; x += 8) {

            // Загружаем 8 BGR-пикселей вручную
            __m256 bf, gf, rf;

            {
                int b[8], g[8], r[8];
                for (int k = 0; k < 8; ++k) {
                    b[k] = src[(x + k) * 3 + 0];
                    g[k] = src[(x + k) * 3 + 1];
                    r[k] = src[(x + k) * 3 + 2];
                }

                bf = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)b));
                gf = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)g));
                rf = _mm256_cvtepi32_ps(_mm256_loadu_si256((__m256i*)r));
            }

            // intensity = (r + g + b) / 3
            __m256 intensity = _mm256_mul_ps(
                _mm256_add_ps(_mm256_add_ps(rf, gf), bf),
                inv3
                );

            bf = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(bf, intensity), scale), intensity);
            gf = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(gf, intensity), scale), intensity);
            rf = _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(rf, intensity), scale), intensity);

            bf = _mm256_min_ps(_mm256_max_ps(bf, zero), maxv);
            gf = _mm256_min_ps(_mm256_max_ps(gf, zero), maxv);
            rf = _mm256_min_ps(_mm256_max_ps(rf, zero), maxv);

            __m256i bi = _mm256_cvtps_epi32(bf);
            __m256i gi = _mm256_cvtps_epi32(gf);
            __m256i ri = _mm256_cvtps_epi32(rf);

            alignas(32) int B[8], G[8], R[8];
            _mm256_store_si256((__m256i*)B, bi);
            _mm256_store_si256((__m256i*)G, gi);
            _mm256_store_si256((__m256i*)R, ri);

            for (int k = 0; k < 8; ++k) {
                dst[(x + k) * 3 + 0] = static_cast<uchar>(B[k]);
                dst[(x + k) * 3 + 1] = static_cast<uchar>(G[k]);
                dst[(x + k) * 3 + 2] = static_cast<uchar>(R[k]);
            }
        }

        // scalar tail
        for (; x < input.cols; ++x) {
            float b = src[x * 3 + 0];
            float g = src[x * 3 + 1];
            float r = src[x * 3 + 2];

            float intensity = (r + g + b) / 3.0f;

            r = intensity + (r - intensity) * saturationScale;
            g = intensity + (g - intensity) * saturationScale;
            b = intensity + (b - intensity) * saturationScale;

            dst[x * 3 + 0] = cv::saturate_cast<uchar>(b);
            dst[x * 3 + 1] = cv::saturate_cast<uchar>(g);
            dst[x * 3 + 2] = cv::saturate_cast<uchar>(r);
        }
    }
}


void adjustSaturationSIMD(const cv::Mat& input, cv::Mat& output, float saturationScale)
{
    if (input.type() != CV_8UC3)
        throw std::invalid_argument("Input image must be CV_8UC3");

    output = input.clone();

    const int numThreads = std::max(1u, std::thread::hardware_concurrency());
    const int rowsPerThread = input.rows / numThreads;

    std::vector<std::thread> threads;
    int startRow = 0;

    for (int t = 0; t < numThreads; ++t) {
        int endRow = (t == numThreads - 1)
        ? input.rows
        : startRow + rowsPerThread;

        threads.emplace_back(
            adjustSaturationSIMDRange,
            std::cref(input),
            std::ref(output),
            startRow,
            endRow,
            saturationScale
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
}


void adjustSaturationManualRange(const cv::Mat& input, cv::Mat& output, int startRow, int endRow, double saturationFactor) {
    for (int i = startRow; i < endRow; ++i) {
        const uchar* inRow  = input.ptr<uchar>(i);
        uchar* outRow = output.ptr<uchar>(i);

        for (int j = 0; j < input.cols; ++j) {
            int idx = j * 3;

            int b = inRow[idx + 0];
            int g = inRow[idx + 1];
            int r = inRow[idx + 2];

            double rN = r / 255.0;
            double gN = g / 255.0;
            double bN = b / 255.0;

            double cmax = std::max({ rN, gN, bN });
            double cmin = std::min({ rN, gN, bN });
            double delta = cmax - cmin;

            double h = 0.0, s = 0.0, v = cmax;

            if (delta > 1e-6) {
                if (cmax == rN)
                    h = 60.0 * fmod((gN - bN) / delta, 6.0);
                else if (cmax == gN)
                    h = 60.0 * ((bN - rN) / delta + 2.0);
                else
                    h = 60.0 * ((rN - gN) / delta + 4.0);

                if (h < 0.0)
                    h += 360.0;

                s = delta / cmax;
            }

            s *= saturationFactor;
            s = std::min(std::max(s, 0.0), 1.0);

            double c = v * s;
            double x = c * (1.0 - fabs(fmod(h / 60.0, 2.0) - 1.0));
            double m = v - c;

            double rp, gp, bp;
            if (h < 60) { rp = c; gp = x; bp = 0; }
            else if (h < 120) { rp = x; gp = c; bp = 0; }
            else if (h < 180) { rp = 0; gp = c; bp = x; }
            else if (h < 240) { rp = 0; gp = x; bp = c; }
            else if (h < 300) { rp = x; gp = 0; bp = c; }
            else { rp = c; gp = 0; bp = x; }

            outRow[idx + 2] = static_cast<uchar>(std::min(255.0, (rp + m) * 255.0));
            outRow[idx + 1] = static_cast<uchar>(std::min(255.0, (gp + m) * 255.0));
            outRow[idx + 0] = static_cast<uchar>(std::min(255.0, (bp + m) * 255.0));
        }
    }
}


void adjustSaturationManual(const cv::Mat& input, cv::Mat& output, double saturationFactor) {
    output = input.clone();

    const int numThreads = std::thread::hardware_concurrency();
    const int rowsPerThread = input.rows / numThreads;

    std::vector<std::thread> threads;

    int startRow = 0;
    for (int t = 0; t < numThreads; ++t) {
        int endRow = (t == numThreads - 1)
        ? input.rows
        : startRow + rowsPerThread;

        threads.emplace_back(
            adjustSaturationManualRange,
            std::cref(input),
            std::ref(output),
            startRow,
            endRow,
            saturationFactor
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
}




void pixelateOpenCV(const cv::Mat& input, cv::Mat& output, int pixelSize) {
    output = input.clone();

    int width = input.cols;
    int height = input.rows;

    // Проходим по изображению блоками пикселей
    for (int i = 0; i < height; i += pixelSize) {
        for (int j = 0; j < width; j += pixelSize) {
            // Рассчитываем среднее значение для блока пикселей
            cv::Rect block(j, i, pixelSize, pixelSize);
            cv::Mat blockRegion = input(block);

            // Находим среднее значение всех пикселей в блоке
            cv::Scalar meanColor = cv::mean(blockRegion);

            // Закрашиваем весь блок этим средним цветом
            for (int y = i; y < std::min(i + pixelSize, height); ++y) {
                for (int x = j; x < std::min(j + pixelSize, width); ++x) {
                    output.at<cv::Vec3b>(y, x) = cv::Vec3b(meanColor[0], meanColor[1], meanColor[2]);
                }
            }
        }
    }
}


void pixelateSIMDRange(const cv::Mat& input, cv::Mat& output, int startBlockY, int endBlockY, int pixelSize) {
    const int width  = input.cols;
    const int height = input.rows;

    for (int blockY = startBlockY; blockY < endBlockY; ++blockY) {
        int y0 = blockY * pixelSize;
        int y1 = std::min(y0 + pixelSize, height);

        for (int x0 = 0; x0 < width; x0 += pixelSize) {
            int x1 = std::min(x0 + pixelSize, width);

            int sumB = 0, sumG = 0, sumR = 0;
            int count = 0;

            // считаем среднее
            for (int y = y0; y < y1; ++y) {
                const uchar* row = input.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x) {
                    sumB += row[x * 3 + 0];
                    sumG += row[x * 3 + 1];
                    sumR += row[x * 3 + 2];
                    ++count;
                }
            }

            const uchar meanB = sumB / count;
            const uchar meanG = sumG / count;
            const uchar meanR = sumR / count;

            // SIMD-закрашивание блока
            __m256i color = _mm256_set1_epi32(
                meanB | (meanG << 8) | (meanR << 16)
                );

            for (int y = y0; y < y1; ++y) {
                uchar* row = output.ptr<uchar>(y);

                int x = x0;
                for (; x <= x1 - 8; x += 8) {
                    _mm256_storeu_si256(
                        reinterpret_cast<__m256i*>(row + x * 3),
                        color
                        );
                }

                // хвост
                for (; x < x1; ++x) {
                    row[x * 3 + 0] = meanB;
                    row[x * 3 + 1] = meanG;
                    row[x * 3 + 2] = meanR;
                }
            }
        }
    }
}


void pixelateSIMD(const cv::Mat& input, cv::Mat& output, int pixelSize)
{
    output = input.clone();

    const int numThreads = std::max(1u, std::thread::hardware_concurrency());
    const int blocksY = (input.rows + pixelSize - 1) / pixelSize;
    const int blocksPerThread = blocksY / numThreads;

    std::vector<std::thread> threads;
    int startBlock = 0;

    for (int t = 0; t < numThreads; ++t) {
        int endBlock = (t == numThreads - 1)
        ? blocksY
        : startBlock + blocksPerThread;

        threads.emplace_back(
            pixelateSIMDRange,
            std::cref(input),
            std::ref(output),
            startBlock,
            endBlock,
            pixelSize
            );

        startBlock = endBlock;
    }

    for (auto& th : threads)
        th.join();
}


void pixelateManualRange(const cv::Mat& input, cv::Mat& output, int startBlockY, int endBlockY, int pixelSize) {
    const int width  = input.cols;
    const int height = input.rows;

    for (int blockY = startBlockY; blockY < endBlockY; ++blockY) {
        int y0 = blockY * pixelSize;
        int y1 = std::min(y0 + pixelSize, height);

        for (int x0 = 0; x0 < width; x0 += pixelSize) {
            int x1 = std::min(x0 + pixelSize, width);

            int sumB = 0, sumG = 0, sumR = 0;
            int count = 0;

            // считаем средний цвет
            for (int y = y0; y < y1; ++y) {
                const uchar* row = input.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x) {
                    sumB += row[x * 3 + 0];
                    sumG += row[x * 3 + 1];
                    sumR += row[x * 3 + 2];
                    ++count;
                }
            }

            uchar meanB = static_cast<uchar>(sumB / count);
            uchar meanG = static_cast<uchar>(sumG / count);
            uchar meanR = static_cast<uchar>(sumR / count);

            // закрашиваем блок
            for (int y = y0; y < y1; ++y) {
                uchar* row = output.ptr<uchar>(y);
                for (int x = x0; x < x1; ++x) {
                    row[x * 3 + 0] = meanB;
                    row[x * 3 + 1] = meanG;
                    row[x * 3 + 2] = meanR;
                }
            }
        }
    }
}


void pixelateManual(const cv::Mat& input, cv::Mat& output, int pixelSize)
{
    output = input.clone();

    const int numThreads = std::max(1u, std::thread::hardware_concurrency());
    const int blocksY = (input.rows + pixelSize - 1) / pixelSize;
    const int blocksPerThread = blocksY / numThreads;

    std::vector<std::thread> threads;
    int startBlock = 0;

    for (int t = 0; t < numThreads; ++t) {
        int endBlock = (t == numThreads - 1)
        ? blocksY
        : startBlock + blocksPerThread;

        threads.emplace_back(
            pixelateManualRange,
            std::cref(input),
            std::ref(output),
            startBlock,
            endBlock,
            pixelSize
            );

        startBlock = endBlock;
    }

    for (auto& th : threads)
        th.join();
}




void sharpenOpenCV(const cv::Mat& input, cv::Mat& output) {
    // Создаём ядро для повышения резкости
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
                          0, -1, 0,
                      -1, 5, -1,
                      0, -1, 0);

    // Применяем фильтр свертки
    cv::filter2D(input, output, -1, kernel);
}


void sharpenSIMDRange(const cv::Mat& input, cv::Mat& output, int startRow, int endRow) {
    const int width = input.cols;

    const __m256 zero = _mm256_set1_ps(0.0f);
    const __m256 maxv = _mm256_set1_ps(255.0f);
    const __m256 five = _mm256_set1_ps(5.0f);

    for (int y = startRow; y < endRow; ++y) {
        const uchar* rowPrev = input.ptr<uchar>(y - 1);
        const uchar* rowCurr = input.ptr<uchar>(y);
        const uchar* rowNext = input.ptr<uchar>(y + 1);
        uchar* dst = output.ptr<uchar>(y);

        int x = 1;

        // SIMD: 8 пикселей по X
        for (; x <= width - 9; x += 8) {

            float cb[8], cg[8], cr[8];
            float tb[8], tg[8], tr[8];
            float bb[8], bg[8], br[8];
            float lb[8], lg[8], lr[8];
            float rb[8], rg[8], rr[8];

            for (int k = 0; k < 8; ++k) {
                int px = x + k;

                cb[k] = rowCurr[px * 3 + 0];
                cg[k] = rowCurr[px * 3 + 1];
                cr[k] = rowCurr[px * 3 + 2];

                tb[k] = rowPrev[px * 3 + 0];
                tg[k] = rowPrev[px * 3 + 1];
                tr[k] = rowPrev[px * 3 + 2];

                bb[k] = rowNext[px * 3 + 0];
                bg[k] = rowNext[px * 3 + 1];
                br[k] = rowNext[px * 3 + 2];

                lb[k] = rowCurr[(px - 1) * 3 + 0];
                lg[k] = rowCurr[(px - 1) * 3 + 1];
                lr[k] = rowCurr[(px - 1) * 3 + 2];

                rb[k] = rowCurr[(px + 1) * 3 + 0];
                rg[k] = rowCurr[(px + 1) * 3 + 1];
                rr[k] = rowCurr[(px + 1) * 3 + 2];
            }

            __m256 B = _mm256_sub_ps(
                _mm256_mul_ps(_mm256_loadu_ps(cb), five),
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_loadu_ps(tb), _mm256_loadu_ps(bb)),
                    _mm256_add_ps(_mm256_loadu_ps(lb), _mm256_loadu_ps(rb))
                    )
                );

            __m256 G = _mm256_sub_ps(
                _mm256_mul_ps(_mm256_loadu_ps(cg), five),
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_loadu_ps(tg), _mm256_loadu_ps(bg)),
                    _mm256_add_ps(_mm256_loadu_ps(lg), _mm256_loadu_ps(rg))
                    )
                );

            __m256 R = _mm256_sub_ps(
                _mm256_mul_ps(_mm256_loadu_ps(cr), five),
                _mm256_add_ps(
                    _mm256_add_ps(_mm256_loadu_ps(tr), _mm256_loadu_ps(br)),
                    _mm256_add_ps(_mm256_loadu_ps(lr), _mm256_loadu_ps(rr))
                    )
                );

            B = _mm256_min_ps(_mm256_max_ps(B, zero), maxv);
            G = _mm256_min_ps(_mm256_max_ps(G, zero), maxv);
            R = _mm256_min_ps(_mm256_max_ps(R, zero), maxv);

            alignas(32) float b[8], g[8], r[8];
            _mm256_store_ps(b, B);
            _mm256_store_ps(g, G);
            _mm256_store_ps(r, R);

            for (int k = 0; k < 8; ++k) {
                dst[(x + k) * 3 + 0] = static_cast<uchar>(b[k]);
                dst[(x + k) * 3 + 1] = static_cast<uchar>(g[k]);
                dst[(x + k) * 3 + 2] = static_cast<uchar>(r[k]);
            }
        }

        // scalar-хвост
        for (; x < width - 1; ++x) {
            for (int c = 0; c < 3; ++c) {
                int val =
                    5 * rowCurr[x * 3 + c]
                    - rowPrev[x * 3 + c]
                    - rowNext[x * 3 + c]
                    - rowCurr[(x - 1) * 3 + c]
                    - rowCurr[(x + 1) * 3 + c];

                dst[x * 3 + c] = cv::saturate_cast<uchar>(val);
            }
        }
    }
}


void sharpenSIMD(const cv::Mat& input, cv::Mat& output)
{
    if (input.type() != CV_8UC3)
        throw std::invalid_argument("Input image must be CV_8UC3");

    output = input.clone();

    const int numThreads = std::max(1u, std::thread::hardware_concurrency());
    const int workRows = input.rows - 2;
    const int rowsPerThread = workRows / numThreads;

    std::vector<std::thread> threads;
    int start = 1;

    for (int t = 0; t < numThreads; ++t) {
        int end = (t == numThreads - 1)
        ? input.rows - 1
        : start + rowsPerThread;

        threads.emplace_back(
            sharpenSIMDRange,
            std::cref(input),
            std::ref(output),
            start,
            end
            );

        start = end;
    }

    for (auto& th : threads)
        th.join();
}


void sharpenManualRange(const cv::Mat& input, cv::Mat& output, int startRow, int endRow) {
    const int width = input.cols;

    for (int y = startRow; y < endRow; ++y) {
        const uchar* rowPrev = input.ptr<uchar>(y - 1);
        const uchar* rowCurr = input.ptr<uchar>(y);
        const uchar* rowNext = input.ptr<uchar>(y + 1);
        uchar* dst = output.ptr<uchar>(y);

        for (int x = 1; x < width - 1; ++x) {
            for (int c = 0; c < 3; ++c) {

                int val =
                    5 * rowCurr[x * 3 + c]
                    - rowPrev[x * 3 + c]
                    - rowNext[x * 3 + c]
                    - rowCurr[(x - 1) * 3 + c]
                    - rowCurr[(x + 1) * 3 + c];

                // ручной clamp
                if (val < 0)   val = 0;
                if (val > 255) val = 255;

                dst[x * 3 + c] = static_cast<uchar>(val);
            }
        }
    }
}


void sharpenManual(const cv::Mat& input, cv::Mat& output)
{
    if (input.type() != CV_8UC3)
        throw std::invalid_argument("Input image must be CV_8UC3");

    output = input.clone();

    const int numThreads = std::max(1u, std::thread::hardware_concurrency());
    const int workRows = input.rows - 2;
    const int rowsPerThread = workRows / numThreads;

    std::vector<std::thread> threads;
    int startRow = 1;

    for (int t = 0; t < numThreads; ++t) {
        int endRow = (t == numThreads - 1)
        ? input.rows - 1
        : startRow + rowsPerThread;

        threads.emplace_back(
            sharpenManualRange,
            std::cref(input),
            std::ref(output),
            startRow,
            endRow
            );

        startRow = endRow;
    }

    for (auto& th : threads)
        th.join();
}
