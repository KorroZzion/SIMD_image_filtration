#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <device_launch_parameters.h>
#include <iostream>

// Функции для работы с минимальными и максимальными значениями
__device__ inline int gpu_min(int a, int b) {
    return (a < b) ? a : b;
}

__device__ inline int gpu_max(int a, int b) {
    return (a > b) ? a : b;
}

// Ядро CUDA для фильтра Гаусса
__global__ void gaussianBlurKernel(const uchar3* input, uchar3* output, int rows, int cols, const float* kernel, int kernelSize, int halfSize) {
    // Определяем текущий пиксель
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Выходим, если пиксель за границами изображения
    if (x >= cols || y >= rows) return;

    // Инициализируем суммы для каналов RGB
    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;

    // Применяем фильтр Гаусса
    for (int i = -halfSize; i <= halfSize; ++i) {
        for (int j = -halfSize; j <= halfSize; ++j) {
            // Вычисляем координаты соседнего пикселя с учетом границ
            int nx = gpu_min(gpu_max(x + j, 0), cols - 1);
            int ny = gpu_min(gpu_max(y + i, 0), rows - 1);

            // Получаем значение пикселя
            uchar3 pixel = input[ny * cols + nx];
            float weight = kernel[(i + halfSize) * kernelSize + (j + halfSize)];

            // Суммируем с учетом веса
            sumB += pixel.x * weight;
            sumG += pixel.y * weight;
            sumR += pixel.z * weight;
        }
    }

    // Записываем результат в диапазоне [0, 255]
    uchar3 result;
    result.x = gpu_min(gpu_max(static_cast<int>(sumB), 0), 255);
    result.y = gpu_min(gpu_max(static_cast<int>(sumG), 0), 255);
    result.z = gpu_min(gpu_max(static_cast<int>(sumR), 0), 255);

    output[y * cols + x] = result;
}

// Хост-функция для выполнения фильтрации на GPU
void gaussianBlurGPU(const cv::Mat& input, cv::Mat& output, int kernelSize, double sigma) {
    int rows = input.rows;
    int cols = input.cols;

    // Создание ядра Гаусса
    cv::Mat kernel = cv::getGaussianKernel(kernelSize, sigma, CV_32F);
    cv::Mat fullKernel = kernel * kernel.t();
    int halfSize = kernelSize / 2;

    // Выделение памяти на GPU
    uchar3* d_input = nullptr;
    uchar3* d_output = nullptr;
    float* d_kernel = nullptr;

    size_t imageSize = rows * cols * sizeof(uchar3);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    // Выделяем память
    cudaError_t err;
    err = cudaMalloc(&d_input, imageSize);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка cudaMalloc для d_input: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    err = cudaMalloc(&d_output, imageSize);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка cudaMalloc для d_output: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        return;
    }

    err = cudaMalloc(&d_kernel, kernelSizeBytes);
    if (err != cudaSuccess) {
        std::cerr << "Ошибка cudaMalloc для d_kernel: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        return;
    }

    // Копируем данные на GPU
    cudaMemcpy(d_input, input.ptr<uchar3>(), imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, fullKernel.ptr<float>(), kernelSizeBytes, cudaMemcpyHostToDevice);

    // Настройка сетки и блоков
    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    // Запуск CUDA-ядра
    gaussianBlurKernel << <gridSize, blockSize >> > (d_input, d_output, rows, cols, d_kernel, kernelSize, halfSize);

    // Проверка ошибок после запуска ядра
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Ошибка запуска ядра: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
        return;
    }

    // Копируем результат обратно на CPU
    output.create(rows, cols, input.type());
    cudaMemcpy(output.ptr<uchar3>(), d_output, imageSize, cudaMemcpyDeviceToHost);

    // Очистка памяти
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}
