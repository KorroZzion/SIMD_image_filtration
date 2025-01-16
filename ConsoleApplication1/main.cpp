#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>  // Для замера времени
#include "cpu_filters.h"
//#include "gpu_filters.cu"

// Функция для замера времени выполнения
template <typename F, typename... Args>
void measureTime(F&& func, Args&&... args) {
    setlocale(0, "");
    auto start = std::chrono::high_resolution_clock::now();  // Начало замера времени
    std::forward<F>(func)(std::forward<Args>(args)...);  // Вызов функции
    auto end = std::chrono::high_resolution_clock::now();  // Конец замера времени
    std::chrono::duration<double> duration = end - start;  // Рассчитываем длительность
    std::cout << "Время выполнения: " << duration.count() << " секунд" << std::endl;  // Выводим результат
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
    setlocale(0, "");
    // Загружаем изображение
    cv::Mat inputImage = cv::imread("image.jpg", cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cout << "Не удалось загрузить изображение!" << std::endl;
        return -1;
    }

    // Параметры для размытия
    int kernelSize;
    double sigma;
    double lowThreshold, highThreshold;
    int beta;
    float  saturationFactor;
    int pixelSize;

    // Получаем выходное изображение
    cv::Mat outputImage;

    // Фильтр Гаусса

    std::cout << "Введите размер ядра (нечётное число): ";
    std::cin >> kernelSize;
    std::cout << std::endl;
    std::cout << "Введите значение сигмы: ";
    std::cin >> sigma;
    std::cout << std::endl;

    std::cout << "Применение фильтра Гаусса с OpenCV: " << std::endl;
    measureTime(gaussianBlurCPU, inputImage, outputImage, kernelSize, sigma);
    cv::imwrite("output_gaussian_blur_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_gaussian_blur_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение фильтра Гаусса вручную: " << std::endl;
    measureTime(gaussianBlurManual, inputImage, outputImage, kernelSize, sigma);
    cv::imwrite("output_gaussian_blur_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_gaussian_blur_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение фильтра Гаусса с SIMD: " << std::endl;
    measureTime(gaussianBlurSIMD, inputImage, outputImage, kernelSize, sigma);
    cv::imwrite("output_gaussian_blur_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_gaussian_blur_SIMD.jpg" << std::endl;
    std::cout << std::endl;



    // Повышениен резкости

    std::cout << "Повышение резкости с OpenCV: " << std::endl;
    measureTime(sharpenOpenCV, inputImage, outputImage);
    cv::imwrite("output_sharpen_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_sharpen_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Повышение резкости вручную: " << std::endl;
    measureTime(sharpenManual, inputImage, outputImage);
    cv::imwrite("output_sharpen_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_sharpen_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Повышение резкости с SIMD: " << std::endl;
    measureTime(sharpenSIMD, inputImage, outputImage);
    cv::imwrite("output_sharpen_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_sharpen_SIMD.jpg" << std::endl;
    std::cout << std::endl;



    // Оператор Собеля

    std::cout << "Применение оператора Собеля с OpenCV: " << std::endl;
    measureTime(sobelFilterOpenCV, inputImage, outputImage);
    cv::imwrite("output_sobel_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_sobel_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение оператора Собеля вручную: " << std::endl;
    measureTime(sobelFilterManual, inputImage, outputImage);
    cv::imwrite("output_sobel_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_sobel_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение оператора Собеля с SIMD: " << std::endl;
    measureTime(sobelFilterSIMD, inputImage, outputImage);
    cv::imwrite("output_sobel_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_sobel_SIMD.jpg" << std::endl;
    std::cout << std::endl;



    // Оператор Канни

    std::cout << "Введите нижний порог (например, 50): ";
    std::cin >> lowThreshold;
    std::cout << std::endl;
    std::cout << "Введите верхний порог (например, 150): ";
    std::cin >> highThreshold;
    std::cout << std::endl;

    std::cout << "Применение оператора Канни с OpenCV: " << std::endl;
    measureTime(cannyEdgeDetectorOpenCV, inputImage, outputImage, lowThreshold, highThreshold);
    cv::imwrite("output_canny_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_canny_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение оператора Канни вручную: " << std::endl;
    measureTime(cannyEdgeDetectorManual, inputImage, outputImage, lowThreshold, highThreshold);
    cv::imwrite("output_canny_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_canny_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение оператора Канни с SIMD: " << std::endl;
    measureTime(cannyEdgeDetectorSIMD, inputImage, outputImage, lowThreshold, highThreshold);
    cv::imwrite("output_canny_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_canny_SIMD.jpg" << std::endl;
    std::cout << std::endl;



    // Медианный фильтр

    std::cout << "Введите размер ядра (нечётное число): ";
    std::cin >> kernelSize;
    std::cout << std::endl;

    std::cout << "Применение медианного фильтра c OpenCV: " << std::endl;
    measureTime(medianFilterOpenCV, inputImage, outputImage, kernelSize);
    cv::imwrite("output_median_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_median_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение медианного фильтра вручную: " << std::endl;
    measureTime(medianFilterManual, inputImage, outputImage, kernelSize);
    cv::imwrite("output_median_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_median_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Применение медианного фильтра c SIMD: " << std::endl;
    measureTime(medianFilterSIMD, inputImage, outputImage, kernelSize);
    cv::imwrite("output_median_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_median_SIMD.jpg" << std::endl;
    std::cout << std::endl;
    


    // Изменение яркости

    std::cout << "Введите коэффициент яркости (например, 5): ";
    std::cin >> beta;
    std::cout << std::endl;

    std::cout << "Изменение яркости с OpenCV: " << std::endl;
    measureTime(adjustBrightnessOpenCV, inputImage, outputImage, beta);
    cv::imwrite("output_brightness_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_brightness_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Изменение яркости вручную: " << std::endl;
    measureTime(adjustBrightnessManual, inputImage, outputImage, beta);
    cv::imwrite("output_brightness_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_brightness_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Изменение яркости c SIMD: " << std::endl;
    measureTime(adjustBrightnessSIMD, inputImage, outputImage, beta);
    cv::imwrite("output_brightness_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_brightness_SIMD.jpg" << std::endl;
    std::cout << std::endl;



    // Изменение насыщенности

    std::cout << "Введите коэффициент насыщенности (например, 2): ";
    std::cin >> saturationFactor;
    std::cout << std::endl;

    std::cout << "Изменение насыщенности c OpenCV: " << std::endl;
    measureTime(adjustSaturationOpenCV, inputImage, outputImage, saturationFactor);
    cv::imwrite("output_saturation_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_saturation_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Изменение насыщенности вручную: " << std::endl;
    measureTime(adjustSaturationManual, inputImage, outputImage, saturationFactor);
    cv::imwrite("output_saturation_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_saturation_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Изменение насыщенности с SIMD: " << std::endl;
    measureTime(adjustSaturationSIMD, inputImage, outputImage, saturationFactor);
    cv::imwrite("output_saturation_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_saturation_SIMD.jpg" << std::endl;
    std::cout << std::endl;



    // Пикселизация

    std::cout << "Введите размер пикселя (степень 2): ";
    std::cin >> pixelSize;
    std::cout << std::endl;

    std::cout << "Пикселизация с OpenCV: " << std::endl;
    measureTime(pixelateOpenCV, inputImage, outputImage, pixelSize);
    cv::imwrite("output_pixelate_OPENCV.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_pixelate_OPENCV.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Пикселизация вручную: " << std::endl;
    measureTime(pixelateManual, inputImage, outputImage, pixelSize);
    cv::imwrite("output_pixelate_MANUAL.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_pixelate_MANUAL.jpg" << std::endl;
    std::cout << std::endl;

    std::cout << "Пикселизация с SIMD: " << std::endl;
    measureTime(pixelateSIMD, inputImage, outputImage, pixelSize);
    cv::imwrite("output_pixelate_SIMD.jpg", outputImage);
    std::cout << "Результат сохранен в файл output_pixelate_SIMD.jpg" << std::endl;
    std::cout << std::endl;

    system("pause");
    return 0;
}
