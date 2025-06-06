#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "cpu_filters.h"
#include <QFileDialog>
#include <QThread>
#include <QMessageBox>
#include <QImage>
#include <QPixmap>
#include <chrono>
#include <QTimer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->filterComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &::MainWindow::filterComboIndexChanged);
    connect(ui->gaussKernelSpin, &QSpinBox::editingFinished,
            this, &::MainWindow::gaussianKernelSpinEditingFinished);
    connect(ui->medianKernelSpin, &QSpinBox::editingFinished,
            this, &::MainWindow::medianKernelSpinEditingFinished);

    connect(ui->loadButton,    &QPushButton::clicked,               this, &MainWindow::loadButtonClicked);
    connect(ui->saveButton,    &QPushButton::clicked,               this, &MainWindow::saveButtonClicked);
    connect(ui->applyButton,   &QPushButton::clicked,               this, &MainWindow::applyButtonClicked);
    connect(ui->filterComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::filterComboIndexChanged);
    connect(ui->videoSlider,   &QSlider::sliderPressed,             this, &MainWindow::videoSliderPressed);
    connect(ui->pauseButton,   &QPushButton::clicked,               this, &MainWindow::pauseButtonClicked);

    ui->pixelSizeSpin->setRange(1, 32768);
    connect(ui->pixelSizeSpin, &QSpinBox::editingFinished,
            this, &::MainWindow::pixelSizeSpinEditingFinished);

    playTimer = new QTimer(this);
    connect(playTimer, &QTimer::timeout, this, &MainWindow::nextFrame);
    connect(ui->videoSlider, &QSlider::sliderPressed,
            this, &::MainWindow::videoSliderPressed);
    connect(ui->videoSlider, &QSlider::valueChanged, this, [this](int value){
        currentFrameIndex = value;
        showComparisonFrame(currentFrameIndex, ui->comparisonSlider->value());

        int elapsedSec = static_cast<int>(currentFrameIndex / videoFps);
        int totalSec   = static_cast<int>(videoFrameCount / videoFps);
        ui->videoTimeLabel->setText(
            formatTime(elapsedSec) + " / " + formatTime(totalSec)
            );
    });
    connect(ui->comparisonSlider, &QSlider::valueChanged, this, [this](int){
        showComparisonFrame(currentFrameIndex, ui->comparisonSlider->value());
    });
    ui->pauseButton->setVisible(false);
    ui->videoSlider->setVisible(false);
    ui->videoTimeLabel->setVisible(false);
}


MainWindow::~MainWindow()
{
    delete ui;
}

bool isPowerOfTwo(int x) {
    return x > 0 && (x & (x - 1)) == 0;
}

int nearestLowerPowerOfTwo(int x) {
    int power = 1;
    while (power * 2 <= x) {
        power *= 2;
    }
    return power;
}

void MainWindow::pixelSizeSpinEditingFinished()
{
    int value = ui->pixelSizeSpin->value();
    if (!isPowerOfTwo(value)) {
        int corrected = nearestLowerPowerOfTwo(value);
        ui->pixelSizeSpin->setValue(corrected);
    }
}


void MainWindow::gaussianKernelSpinEditingFinished()
{
    int value = ui->gaussKernelSpin->value();
    if (value % 2 == 0) {
        ui->gaussKernelSpin->setValue(value + 1);
    }
}

void MainWindow::medianKernelSpinEditingFinished()
{
    int value = ui->medianKernelSpin->value();
    if (value % 2 == 0) {
        ui->medianKernelSpin->setValue(value + 1);
    }
}

void MainWindow::filterComboIndexChanged(int index)
{
    ui->parametersStack->setCurrentIndex(index);
}


void MainWindow::pauseButtonClicked()
{
    if (!isVideo) return;

    if (isPlaying) {
        playTimer->stop();
        isPlaying = false;
        ui->pauseButton->setText("▶");
        return;
    } else {
        playTimer->start(30);
        isPlaying = true;
        ui->pauseButton->setText("⏸");
        return;
    }
}



void MainWindow::loadButtonClicked()
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

        isVideo = false; // указываем, что это изображение

        ui->pauseButton->setVisible(false);
        ui->videoSlider->setVisible(false);
        ui->videoTimeLabel->setVisible(false);

        cv::cvtColor(originalImage, originalImage, cv::COLOR_BGR2RGB);
        QImage qimg(originalImage.data, originalImage.cols, originalImage.rows, originalImage.step, QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(ui->imageLabel->size(), Qt::KeepAspectRatio));
    }
    else if (extension == "mp4" || extension == "avi" || extension == "mov") {
        currentVideoPath = filename;
        isVideo = true;
        ui->pauseButton->setVisible(true);
        ui->videoSlider->setVisible(true);
        ui->videoTimeLabel->setVisible(true);

        // Читаем ВСЕ кадры в оригинал
        originalVideoFrames.clear();
        cv::VideoCapture cap(filename.toStdString());
        videoFps = cap.get(cv::CAP_PROP_FPS);
        videoFrameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        if (!cap.isOpened()) {
            QMessageBox::warning(this, "Ошибка", "Не удалось открыть видео.");
            return;
        }
        cv::Mat frame;
        while (cap.read(frame)) {
            originalVideoFrames.push_back(frame.clone());
        }
        ui->videoSlider->setMinimum(0);
        ui->videoSlider->setMaximum(videoFrameCount - 1);
        ui->videoSlider->setValue(0);

        int totalSec = static_cast<int>(videoFrameCount / videoFps);
        ui->videoTimeLabel->setText(
            formatTime(0) + " / " + formatTime(totalSec)
            );

        cap.release();

        // До фильтрации processed = original
        processedVideoFrames = originalVideoFrames;

        // Настраиваем слайдер
        int n = originalVideoFrames.size();
        ui->videoSlider->setMinimum(0);
        ui->videoSlider->setMaximum(n - 1);
        ui->videoSlider->setValue(0);
        currentFrameIndex = 0;

        // Показываем первый кадр:
        showComparisonFrame(0, ui->comparisonSlider->value());
    }
    else {
        QMessageBox::warning(this, "Ошибка", "Неподдерживаемый формат файла.");
    }
}


void MainWindow::saveButtonClicked()
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
            writer.write(frame);
        }

        writer.release();
    }
}


void MainWindow::applyButtonClicked()
{
    if (!isVideo) {
        // Стандартная логика для изображения
        originalVideoFrames.clear();
        processedVideoFrames.clear();
        if (originalImage.empty()) {
            QMessageBox::warning(this, "Ошибка", "Сначала загрузите изображение.");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        applySelectedFilter();
        auto end = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double>(end - start).count();
        ui->timeLabel->setText(QString("Время обработки: %1 с").arg(duration, 0, 'f', 6));

    } else {
        // Очистить предыдущие кадры
        originalVideoFrames.clear();
        processedVideoFrames.clear();

        cv::VideoCapture cap(currentVideoPath.toStdString());
        if (!cap.isOpened()) {
            QMessageBox::warning(this, "Ошибка", "Не удалось открыть видео.");
            return;
        }

        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) break;

            // сохраняем оригинал
            originalVideoFrames.push_back(frame.clone());

            // применяем фильтр
            originalImage = frame;
            applySelectedFilter();
            processedVideoFrames.push_back(processedImage.clone());
        }
        cap.release();
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double>(end - start).count();
        ui->timeLabel->setText(QString("Время обработки: %1 с").arg(duration, 0, 'f', 6));

        // Настраиваем ползунок перемотки
        int n = static_cast<int>(processedVideoFrames.size());
        ui->videoSlider->setMinimum(0);
        ui->videoSlider->setMaximum(n - 1);
        ui->videoSlider->setValue(0);
        currentFrameIndex = 0;

        // Показываем первый кадр
        showComparisonFrame(0, ui->comparisonSlider->value());

        // Запускаем воспроизведение
        isPlaying = true;
        playTimer->start(30); // ~30 мс между кадрами (примерно 33 fps)
        ui->pauseButton->setText("⏸");
    }
}

void MainWindow::showComparisonFrame(int frameIndex, int comparisonValue)
{
    if (frameIndex < 0 || frameIndex >= int(originalVideoFrames.size()))
        return;

    // Получаем оба кадра и конвертируем в RGB
    cv::Mat orig = originalVideoFrames[frameIndex];
    cv::Mat proc = processedVideoFrames[frameIndex];
    cv::Mat origRGB, procRGB;
    cv::cvtColor(orig,  origRGB, cv::COLOR_BGR2RGB);
    cv::cvtColor(proc,  procRGB, cv::COLOR_BGR2RGB);

    // Если ползунок в 0% — показываем целиком оригинал
    if (comparisonValue <= 0) {
        QImage img(procRGB.data, procRGB.cols, procRGB.rows,
                   int(procRGB.step), QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(img)
                                      .scaled(ui->imageLabel->size(),
                                              Qt::KeepAspectRatio));
        return;
    }
    // Если ползунок в 100% — показываем целиком обработанное
    if (comparisonValue >= 100) {

        QImage img(origRGB.data, origRGB.cols, origRGB.rows,
                   int(origRGB.step), QImage::Format_RGB888);
        ui->imageLabel->setPixmap(QPixmap::fromImage(img)
                                      .scaled(ui->imageLabel->size(),
                                              Qt::KeepAspectRatio));
        return;
    }

    // Для остальных значений — нормальная «смешанная» отрисовка
    int width  = origRGB.cols;
    int height = origRGB.rows;
    int splitX = width * comparisonValue / 100;

    // Создаём пустую картинку
    cv::Mat blended(height, width, CV_8UC3);

    // Копируем левую часть из original
    origRGB(cv::Rect(0, 0, splitX, height))
        .copyTo(blended(cv::Rect(0, 0, splitX, height)));

    // Копируем правую часть из processed
    procRGB(cv::Rect(splitX, 0, width - splitX, height))
        .copyTo(blended(cv::Rect(splitX, 0, width - splitX, height)));

    // Отправляем в QLabel
    QImage img(blended.data, blended.cols, blended.rows,
               int(blended.step), QImage::Format_RGB888);
    ui->imageLabel->setPixmap(QPixmap::fromImage(img)
                                  .scaled(ui->imageLabel->size(),
                                          Qt::KeepAspectRatio));
}



QString MainWindow::formatTime(int totalSeconds) const
{
    int h = totalSeconds / 3600;
    int m = (totalSeconds % 3600) / 60;
    int s = totalSeconds % 60;

    if (h > 0)
        return QString("%1:%2:%3")
            .arg(h, 2, 10, QChar('0'))
            .arg(m, 2, 10, QChar('0'))
            .arg(s, 2, 10, QChar('0'));
    else
        return QString("%1:%2")
            .arg(m, 2, 10, QChar('0'))
            .arg(s, 2, 10, QChar('0'));
}


void MainWindow::nextFrame()
{
    if (!isVideo || processedVideoFrames.empty()) return;

    currentFrameIndex++;
    // сколько прошло
    int elapsedSec = static_cast<int>(currentFrameIndex / videoFps);
    // сколько всего
    int totalSec = static_cast<int>(videoFrameCount / videoFps);

    ui->videoTimeLabel->setText(
        formatTime(elapsedSec) + " / " + formatTime(totalSec)
        );

    if (currentFrameIndex >= (int)processedVideoFrames.size()) {
        currentFrameIndex = 0; // Зациклить
    }

    ui->videoSlider->blockSignals(true);
    ui->videoSlider->setValue(currentFrameIndex);
    ui->videoSlider->blockSignals(false);

    showComparisonFrame(currentFrameIndex, ui->comparisonSlider->value());
}


void MainWindow::videoSliderPressed()
{
    if (isPlaying) {
        playTimer->stop();
        isPlaying = false;
        ui->pauseButton->setText("▶");
    }
}

void MainWindow::paintEvent(QPaintEvent* event)
{
    if (!isVideo)
    {
        QMainWindow::paintEvent(event);

        if (originalImage.empty() || processedImage.empty()) return;

        cv::Mat originalRGB, processedRGB;

        if (originalImage.channels() == 1)
            cv::cvtColor(originalImage, originalRGB, cv::COLOR_GRAY2RGB);
        else
            originalRGB = originalImage.clone();

        if (processedImage.channels() == 1) {
            cv::Mat normalized;
            cv::normalize(processedImage, normalized, 0, 255, cv::NORM_MINMAX);
            normalized.convertTo(normalized, CV_8U);
            cv::cvtColor(normalized, processedRGB, cv::COLOR_GRAY2RGB);
        } else {
            processedRGB = processedImage.clone();
        }

        QPixmap before = QPixmap::fromImage(QImage(originalRGB.data, originalRGB.cols, originalRGB.rows,
                                                   static_cast<int>(originalRGB.step), QImage::Format_RGB888).copy());
        QPixmap after = QPixmap::fromImage(QImage(processedRGB.data, processedRGB.cols, processedRGB.rows,
                                                  static_cast<int>(processedRGB.step), QImage::Format_RGB888).copy());

        before = before.scaled(ui->imageLabel->size(), Qt::KeepAspectRatio);
        after = after.scaled(ui->imageLabel->size(), Qt::KeepAspectRatio);

        int sliderValue = ui->comparisonSlider->value();
        int splitPos = before.width() * sliderValue / 100;

        QPixmap composed(before.size());
        composed.fill(Qt::transparent);
        QPainter painter(&composed);
        painter.drawPixmap(0, 0, before.copy(0, 0, splitPos, before.height()));
        painter.drawPixmap(splitPos, 0, after.copy(splitPos, 0, after.width() - splitPos, after.height()));

        painter.end();

        ui->imageLabel->setPixmap(composed);
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
            gaussianBlurOpenCV(originalImage, processedImage, k, sigma);
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
