#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <vector>
#include <QPainter>



QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_loadButton_clicked();
    void on_saveButton_clicked();
    void on_applyButton_clicked();
    void on_filterComboBox_currentIndexChanged(int index);
    void on_gaussianKernelSpinEditingFinished();
    void on_medianKernelSpinEditingFinished();
    void on_pixelSizeSpinEditingFinished();
    void nextFrame();
    void on_videoSlider_sliderPressed();
    void on_pauseButton_clicked();


private:
    Ui::MainWindow *ui;
    cv::Mat originalImage;
    cv::Mat processedImage;
    bool isVideo = false;
    QString currentVideoPath;
    std::vector<cv::Mat> processedFrames;
    std::vector<cv::Mat> processedVideoFrames;

    std::vector<cv::Mat> originalVideoFrames;
    QTimer *playTimer;
    bool isPlaying = false;
    int currentFrameIndex = 0;

    void applySelectedFilter();
    void showComparisonFrame(int frameIndex, int comparisonValue);

    double videoFps = 30.0;
    int videoFrameCount = 0;
    QString formatTime(int totalSeconds) const;

protected:
    void paintEvent(QPaintEvent* event) override;

};

#endif // MAINWINDOW_H
