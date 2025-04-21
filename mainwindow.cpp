#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cpu_filters.h"
#include <QFileDialog>
#include <QThread>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <chrono>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->filterComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::on_filterComboBox_currentIndexChanged);
    connect(ui->gaussKernelSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::on_gaussianKernelSizeSpinBox_valueChanged);
    connect(ui->medianKernelSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &MainWindow::on_medianKernelSizeSpinBox_valueChanged);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_gaussianKernelSizeSpinBox_valueChanged(int value)
{
    if (value % 2 == 0) {
        ui->gaussKernelSpin->setValue(value + 1);
    }
}

void MainWindow::on_medianKernelSizeSpinBox_valueChanged(int value)
{
    if (value % 2 == 0) {
        ui->medianKernelSpin->setValue(value +1);
    }
}


void MainWindow::on_filterComboBox_currentIndexChanged(int index)
{
    // Просто переключаем страницу в stackedWidget
    ui->parametersStack->setCurrentIndex(index);
}


void MainWindow::on_loadButton_clicked()
{
    QString filename = QFileDialog::getOpenFileName(this, "Открыть изображение или видео", "",
                                                    "Изображения и видео (*.png *.jpg *.bmp *.mp4 *.avi *.mov)");
    if (filename.isEmpty()) return;

    QString extension = QFileInfo(filename).suffix().toLower();

    if (extension == "png" || extension == "jpg" || extension == "bmp") {
        originalImage = cv::imread(filename.toStdString());
        if (originalImage.empty()) {
            QMessageBox::warning(this, "Ошибка", "Не удалось загрузить изображение.");
            return;
        }

        isVideo = false; // 👈 указываем, что это изображение

        cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2RGB);
        QImage qimg(originalImage.data, originalImage.cols, originalImage.rows, originalImage.step, QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));
    }
    else if (extension == "mp4" || extension == "avi" || extension == "mov") {
        currentVideoPath = filename; // сохранение пути к видео
        loadAndProcessVideo(filename);
        isVideo = true;
    }
    else {
        QMessageBox::warning(this, "Ошибка", "Неподдерживаемый формат файла.");
    }
}

void MainWindow::loadAndProcessVideo(QString filePath)
{
    cv::VideoCapture cap(filePath.toStdString());
    if (!cap.isOpened()) {
        QMessageBox::warning(this, "Ошибка", "Не удалось открыть видео.");
        return;
    }

    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        cv::Mat filteredFrame;
        originalImage = frame;  // чтобы фильтры могли работать с originalImage

        applySelectedFilter();  // твоя уже готовая функция
        filteredFrame = processedImage;

        if (filteredFrame.channels() == 1) {
            cv::cvtColor(filteredFrame, filteredFrame, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(filteredFrame, filteredFrame, cv::COLOR_BGR2RGB);
        }

        QImage qimg(filteredFrame.data, filteredFrame.cols, filteredFrame.rows, static_cast<int>(filteredFrame.step), QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));
        cv::waitKey(30); // задержка между кадрами (можно регулировать)
        QCoreApplication::processEvents();  // позволяет обновить GUI
    }

    cap.release();
}



void MainWindow::on_saveButton_clicked()
{
    if (!isVideo) {
        if (processedImage.empty()) {
            QMessageBox::information(this, "Нет данных", "Сначала обработайте изображение.");
            return;
        }

        QString filename = QFileDialog::getSaveFileName(this, "Сохранить изображение", "", "PNG (*.png);;JPEG (*.jpg)");
        if (!filename.isEmpty()) {
            cv::Mat saveImg;
            cv::cvtColor(processedImage, saveImg, cv::COLOR_RGB2BGR);
            cv::imwrite(filename.toStdString(), saveImg);
        }
    } else {
        if (processedVideoFrames.empty()) {
            QMessageBox::information(this, "Нет данных", "Сначала примените фильтр к видео.");
            return;
        }

        QString filename = QFileDialog::getSaveFileName(this, "Сохранить видео", "", "Видео (*.avi)");
        if (filename.isEmpty()) return;

        int fps = 30;
        cv::Size frameSize = processedVideoFrames[0].size();
        cv::VideoWriter writer(filename.toStdString(), cv::VideoWriter::fourcc('M','J','P','G'), fps, frameSize);

        for (const auto& frame : processedVideoFrames) {
            cv::Mat bgrFrame;
            cv::cvtColor(frame, bgrFrame, cv::COLOR_RGB2BGR);
            writer.write(bgrFrame);
        }

        writer.release();
    }
}


void MainWindow::on_applyButton_clicked()
{
    if (!isVideo) {
        // Стандартная логика для изображения
        if (originalImage.empty()) {
            QMessageBox::warning(this, "Ошибка", "Сначала загрузите изображение.");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        applySelectedFilter();
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        ui->timeLabel->setText(QString("Время обработки: %1 с").arg(duration, 0, 'f', 6));

        cv::Mat rgb;

        QString selectedFilter = ui->filterComboBox->currentText();

        if (selectedFilter == "Оператор Собеля" || selectedFilter == "Оператор Канни" || selectedFilter == "Медианный фильтр") {
            cv::Mat normalized;
            cv::normalize(processedImage, normalized, 0, 255, cv::NORM_MINMAX);
            normalized.convertTo(normalized, CV_8U);
            cv::cvtColor(normalized, rgb, cv::COLOR_GRAY2RGB);
            QImage qimg(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888);
            ui->imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
        } else {
            if (processedImage.channels() == 3) {
                cv::cvtColor(processedImage, rgb, cv::COLOR_BGR2RGB);
                QImage qimg(processedImage.data, processedImage.cols, processedImage.rows, processedImage.step, QImage::Format_RGB888);
                ui->imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));
            } else {
                QMessageBox::warning(this, "Ошибка", "Неподдерживаемый формат изображения.");
                return;
            }
        }
    } else {
        // Логика для видео
        processedVideoFrames.clear(); // очищаем старые кадры

        cv::VideoCapture cap(currentVideoPath.toStdString());
        if (!cap.isOpened()) {
            QMessageBox::warning(this, "Ошибка", "Не удалось открыть видео.");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            originalImage = frame;
            applySelectedFilter();
            processedVideoFrames.push_back(processedImage.clone()); // сохраняем обработанный кадр
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        ui->timeLabel->setText(QString("Время обработки: %1 с").arg(duration, 0, 'f', 6));

        cap.release();

        // Воспроизвести обработанное видео
        playProcessedVideo();
    }
}

void MainWindow::playProcessedVideo()
{
    for (const auto& frame : processedVideoFrames) {
        cv::Mat rgbFrame;
        if (frame.channels() == 1) {
            cv::cvtColor(frame, rgbFrame, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(frame, rgbFrame, cv::COLOR_BGR2RGB);
        }

        QImage qimg(rgbFrame.data, rgbFrame.cols, rgbFrame.rows, static_cast<int>(rgbFrame.step), QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));

        QCoreApplication::processEvents();
        QThread::msleep(30); // задержка для имитации видео
    }
}



void MainWindow::applySelectedFilter()
{
    QString filter = ui->filterComboBox->currentText();
    QString impl = ui->implementationComboBox->currentText();

    if (filter == "Размытие по Гауссу") {
        int k = ui->gaussKernelSpin->value();
        double sigma = ui->gaussSigmaSpin->value();

        if (impl == "OpenCV")
            gaussianBlurCPU(originalImage, processedImage, k, sigma);
        else if (impl == "Без библиотек")
            gaussianBlurManual(originalImage, processedImage, k, sigma);
        else if (impl == "SIMD")
            gaussianBlurSIMD(originalImage, processedImage, k, sigma);
    }
    else if (filter == "Оператор Собеля") {
        if (impl == "OpenCV")
            sobelFilterOpenCV(originalImage, processedImage);
        if (originalImage.empty()) {
            qDebug() << "Input image is empty!";
            return;
        }
        else if (impl == "Без библиотек")
            sobelFilterManual(originalImage, processedImage);
        else if (impl == "SIMD")
            sobelFilterSIMD(originalImage, processedImage);
    }
    else if (filter == "Оператор Канни") {
        double low = ui->cannyLowSpin->value();
        double high = ui->cannyHighSpin->value();

        if (impl == "OpenCV")
            cannyEdgeDetectorOpenCV(originalImage, processedImage, low, high);
        else if (impl == "Без библиотек")
            cannyEdgeDetectorManual(originalImage, processedImage, low, high);
        else if (impl == "SIMD")
            cannyEdgeDetectorSIMD(originalImage, processedImage, low, high);
    }
    else if (filter == "Медианный фильтр") {
        int kernelSize = ui->medianKernelSpin->value();

        if (impl == "OpenCV")
            medianFilterOpenCV(originalImage, processedImage, kernelSize);
        else if (impl == "Без библиотек")
            medianFilterManual(originalImage, processedImage, kernelSize);
        else if (impl == "SIMD")
            medianFilterSIMD(originalImage, processedImage, kernelSize);
    }
    else if (filter == "Изменение яркости") {
        int beta = ui->brightBetaSpin->value();

        if (impl == "OpenCV")
            adjustBrightnessOpenCV(originalImage, processedImage, beta);
        else if (impl == "Без библиотек")
            adjustBrightnessManual(originalImage, processedImage, beta);
        else if (impl == "SIMD")
            adjustBrightnessSIMD(originalImage, processedImage, beta);
    }
    else if (filter == "Изменение насыщенности") {
        double contrastAlphaSpin = ui->contrastAlphaSpin->value();

        if (impl == "OpenCV")
            adjustSaturationOpenCV(originalImage, processedImage, contrastAlphaSpin);
        else if (impl == "Без библиотек")
            adjustSaturationManual(originalImage, processedImage, contrastAlphaSpin);
        else if (impl == "SIMD")
            adjustSaturationSIMD(originalImage, processedImage, contrastAlphaSpin);
    }
    else if (filter == "Пикселизация") {
        int pixelSize = ui->pixelSizeSpin->value();

        if (impl == "OpenCV")
            pixelateOpenCV(originalImage, processedImage, pixelSize);
        else if (impl == "Без библиотек")
            pixelateManual(originalImage, processedImage, pixelSize);
        else if (impl == "SIMD")
            pixelateSIMD(originalImage, processedImage, pixelSize);
    }
    else if (filter == "Изменение резкости") {
        if (impl == "OpenCV")
            sharpenOpenCV(originalImage, processedImage);
        else if (impl == "Без библиотек")
            sharpenManual(originalImage, processedImage);
        else if (impl == "SIMD")
            sharpenSIMD(originalImage, processedImage);
    }
}
