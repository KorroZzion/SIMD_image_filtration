#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>
#include <vector>

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
    void on_gaussianKernelSizeSpinBox_valueChanged(int value);
    void on_medianKernelSizeSpinBox_valueChanged(int value);
    void loadAndProcessVideo(QString filePath);
    void playProcessedVideo();

private:
    Ui::MainWindow *ui;
    cv::Mat originalImage;
    cv::Mat processedImage;
    bool isVideo = false;
    QString currentVideoPath;
    std::vector<cv::Mat> processedFrames;
    std::vector<cv::Mat> processedVideoFrames;

    void applySelectedFilter(); // 👈 функция, которую мы пропишем
};

#endif // MAINWINDOW_H
