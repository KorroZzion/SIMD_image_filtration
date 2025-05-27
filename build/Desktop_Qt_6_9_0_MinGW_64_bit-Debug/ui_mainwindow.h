/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 6.9.0
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFormLayout>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QHBoxLayout *horizontalLayout;
    QGroupBox *controlPanel;
    QVBoxLayout *controlLayout;
    QPushButton *loadButton;
    QPushButton *saveButton;
    QComboBox *filterComboBox;
    QComboBox *implementationComboBox;
    QStackedWidget *parametersStack;
    QWidget *pageGaussian;
    QFormLayout *formLayoutGaussian;
    QLabel *labelGaussKernel;
    QSpinBox *gaussKernelSpin;
    QLabel *labelGaussSigma;
    QDoubleSpinBox *gaussSigmaSpin;
    QWidget *pageSobel;
    QVBoxLayout *layoutSobel;
    QLabel *labelSobelParams;
    QWidget *pageCanny;
    QFormLayout *formLayoutCanny;
    QLabel *labelCannyLow;
    QSpinBox *cannyLowSpin;
    QLabel *labelCannyHigh;
    QSpinBox *cannyHighSpin;
    QWidget *pageMedian;
    QFormLayout *formLayoutMedian;
    QLabel *labelMedianKernel;
    QSpinBox *medianKernelSpin;
    QWidget *pageBrightness;
    QFormLayout *formLayoutBright;
    QLabel *labelBrightAlpha;
    QSpinBox *brightBetaSpin;
    QWidget *pageContrast;
    QFormLayout *formLayoutContrast;
    QLabel *labelContrastAlpha;
    QDoubleSpinBox *contrastAlphaSpin;
    QWidget *pagePixelate;
    QFormLayout *formLayoutPixelate;
    QLabel *labelPixelSize;
    QSpinBox *pixelSizeSpin;
    QWidget *pageSharpen;
    QVBoxLayout *layoutSharpen;
    QLabel *labelSharpenParams;
    QLabel *label;
    QSlider *comparisonSlider;
    QPushButton *applyButton;
    QLabel *timeLabel;
    QVBoxLayout *verticalLayout;
    QLabel *imageLabel;
    QHBoxLayout *horizontalLayout_3;
    QPushButton *pauseButton;
    QSlider *videoSlider;
    QLabel *videoTimeLabel;
    QMenuBar *menubar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName("MainWindow");
        MainWindow->resize(964, 554);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName("centralwidget");
        horizontalLayout = new QHBoxLayout(centralwidget);
        horizontalLayout->setObjectName("horizontalLayout");
        controlPanel = new QGroupBox(centralwidget);
        controlPanel->setObjectName("controlPanel");
        controlLayout = new QVBoxLayout(controlPanel);
        controlLayout->setObjectName("controlLayout");
        loadButton = new QPushButton(controlPanel);
        loadButton->setObjectName("loadButton");

        controlLayout->addWidget(loadButton);

        saveButton = new QPushButton(controlPanel);
        saveButton->setObjectName("saveButton");

        controlLayout->addWidget(saveButton);

        filterComboBox = new QComboBox(controlPanel);
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->addItem(QString());
        filterComboBox->setObjectName("filterComboBox");

        controlLayout->addWidget(filterComboBox);

        implementationComboBox = new QComboBox(controlPanel);
        implementationComboBox->addItem(QString());
        implementationComboBox->addItem(QString());
        implementationComboBox->addItem(QString());
        implementationComboBox->setObjectName("implementationComboBox");

        controlLayout->addWidget(implementationComboBox);

        parametersStack = new QStackedWidget(controlPanel);
        parametersStack->setObjectName("parametersStack");
        pageGaussian = new QWidget();
        pageGaussian->setObjectName("pageGaussian");
        formLayoutGaussian = new QFormLayout(pageGaussian);
        formLayoutGaussian->setObjectName("formLayoutGaussian");
        labelGaussKernel = new QLabel(pageGaussian);
        labelGaussKernel->setObjectName("labelGaussKernel");

        formLayoutGaussian->setWidget(0, QFormLayout::ItemRole::LabelRole, labelGaussKernel);

        gaussKernelSpin = new QSpinBox(pageGaussian);
        gaussKernelSpin->setObjectName("gaussKernelSpin");
        gaussKernelSpin->setMinimum(1);
        gaussKernelSpin->setMaximum(99);
        gaussKernelSpin->setSingleStep(2);
        gaussKernelSpin->setValue(5);

        formLayoutGaussian->setWidget(0, QFormLayout::ItemRole::FieldRole, gaussKernelSpin);

        labelGaussSigma = new QLabel(pageGaussian);
        labelGaussSigma->setObjectName("labelGaussSigma");

        formLayoutGaussian->setWidget(1, QFormLayout::ItemRole::LabelRole, labelGaussSigma);

        gaussSigmaSpin = new QDoubleSpinBox(pageGaussian);
        gaussSigmaSpin->setObjectName("gaussSigmaSpin");
        gaussSigmaSpin->setDecimals(1);
        gaussSigmaSpin->setMinimum(0.000000000000000);
        gaussSigmaSpin->setMaximum(100.000000000000000);
        gaussSigmaSpin->setSingleStep(1.000000000000000);
        gaussSigmaSpin->setValue(1.000000000000000);

        formLayoutGaussian->setWidget(1, QFormLayout::ItemRole::FieldRole, gaussSigmaSpin);

        parametersStack->addWidget(pageGaussian);
        pageSobel = new QWidget();
        pageSobel->setObjectName("pageSobel");
        layoutSobel = new QVBoxLayout(pageSobel);
        layoutSobel->setObjectName("layoutSobel");
        labelSobelParams = new QLabel(pageSobel);
        labelSobelParams->setObjectName("labelSobelParams");
        labelSobelParams->setAlignment(Qt::AlignmentFlag::AlignCenter);

        layoutSobel->addWidget(labelSobelParams);

        parametersStack->addWidget(pageSobel);
        pageCanny = new QWidget();
        pageCanny->setObjectName("pageCanny");
        formLayoutCanny = new QFormLayout(pageCanny);
        formLayoutCanny->setObjectName("formLayoutCanny");
        labelCannyLow = new QLabel(pageCanny);
        labelCannyLow->setObjectName("labelCannyLow");

        formLayoutCanny->setWidget(0, QFormLayout::ItemRole::LabelRole, labelCannyLow);

        cannyLowSpin = new QSpinBox(pageCanny);
        cannyLowSpin->setObjectName("cannyLowSpin");
        cannyLowSpin->setMinimum(0);
        cannyLowSpin->setMaximum(255);
        cannyLowSpin->setValue(50);

        formLayoutCanny->setWidget(0, QFormLayout::ItemRole::FieldRole, cannyLowSpin);

        labelCannyHigh = new QLabel(pageCanny);
        labelCannyHigh->setObjectName("labelCannyHigh");

        formLayoutCanny->setWidget(1, QFormLayout::ItemRole::LabelRole, labelCannyHigh);

        cannyHighSpin = new QSpinBox(pageCanny);
        cannyHighSpin->setObjectName("cannyHighSpin");
        cannyHighSpin->setMinimum(0);
        cannyHighSpin->setMaximum(255);
        cannyHighSpin->setValue(150);

        formLayoutCanny->setWidget(1, QFormLayout::ItemRole::FieldRole, cannyHighSpin);

        parametersStack->addWidget(pageCanny);
        pageMedian = new QWidget();
        pageMedian->setObjectName("pageMedian");
        formLayoutMedian = new QFormLayout(pageMedian);
        formLayoutMedian->setObjectName("formLayoutMedian");
        labelMedianKernel = new QLabel(pageMedian);
        labelMedianKernel->setObjectName("labelMedianKernel");

        formLayoutMedian->setWidget(0, QFormLayout::ItemRole::LabelRole, labelMedianKernel);

        medianKernelSpin = new QSpinBox(pageMedian);
        medianKernelSpin->setObjectName("medianKernelSpin");
        medianKernelSpin->setMinimum(1);
        medianKernelSpin->setMaximum(99);
        medianKernelSpin->setSingleStep(2);
        medianKernelSpin->setValue(3);

        formLayoutMedian->setWidget(0, QFormLayout::ItemRole::FieldRole, medianKernelSpin);

        parametersStack->addWidget(pageMedian);
        pageBrightness = new QWidget();
        pageBrightness->setObjectName("pageBrightness");
        formLayoutBright = new QFormLayout(pageBrightness);
        formLayoutBright->setObjectName("formLayoutBright");
        labelBrightAlpha = new QLabel(pageBrightness);
        labelBrightAlpha->setObjectName("labelBrightAlpha");

        formLayoutBright->setWidget(0, QFormLayout::ItemRole::LabelRole, labelBrightAlpha);

        brightBetaSpin = new QSpinBox(pageBrightness);
        brightBetaSpin->setObjectName("brightBetaSpin");
        brightBetaSpin->setMinimum(-100);
        brightBetaSpin->setMaximum(100);
        brightBetaSpin->setValue(0);

        formLayoutBright->setWidget(0, QFormLayout::ItemRole::FieldRole, brightBetaSpin);

        parametersStack->addWidget(pageBrightness);
        pageContrast = new QWidget();
        pageContrast->setObjectName("pageContrast");
        formLayoutContrast = new QFormLayout(pageContrast);
        formLayoutContrast->setObjectName("formLayoutContrast");
        labelContrastAlpha = new QLabel(pageContrast);
        labelContrastAlpha->setObjectName("labelContrastAlpha");

        formLayoutContrast->setWidget(0, QFormLayout::ItemRole::LabelRole, labelContrastAlpha);

        contrastAlphaSpin = new QDoubleSpinBox(pageContrast);
        contrastAlphaSpin->setObjectName("contrastAlphaSpin");
        contrastAlphaSpin->setDecimals(0);
        contrastAlphaSpin->setMinimum(0.000000000000000);
        contrastAlphaSpin->setMaximum(100.000000000000000);
        contrastAlphaSpin->setSingleStep(1.000000000000000);
        contrastAlphaSpin->setValue(1.000000000000000);

        formLayoutContrast->setWidget(0, QFormLayout::ItemRole::FieldRole, contrastAlphaSpin);

        parametersStack->addWidget(pageContrast);
        pagePixelate = new QWidget();
        pagePixelate->setObjectName("pagePixelate");
        formLayoutPixelate = new QFormLayout(pagePixelate);
        formLayoutPixelate->setObjectName("formLayoutPixelate");
        labelPixelSize = new QLabel(pagePixelate);
        labelPixelSize->setObjectName("labelPixelSize");

        formLayoutPixelate->setWidget(0, QFormLayout::ItemRole::LabelRole, labelPixelSize);

        pixelSizeSpin = new QSpinBox(pagePixelate);
        pixelSizeSpin->setObjectName("pixelSizeSpin");
        pixelSizeSpin->setMinimum(1);
        pixelSizeSpin->setMaximum(1000000000);
        pixelSizeSpin->setValue(1);

        formLayoutPixelate->setWidget(0, QFormLayout::ItemRole::FieldRole, pixelSizeSpin);

        parametersStack->addWidget(pagePixelate);
        pageSharpen = new QWidget();
        pageSharpen->setObjectName("pageSharpen");
        layoutSharpen = new QVBoxLayout(pageSharpen);
        layoutSharpen->setObjectName("layoutSharpen");
        labelSharpenParams = new QLabel(pageSharpen);
        labelSharpenParams->setObjectName("labelSharpenParams");
        labelSharpenParams->setAlignment(Qt::AlignmentFlag::AlignCenter);

        layoutSharpen->addWidget(labelSharpenParams);

        parametersStack->addWidget(pageSharpen);

        controlLayout->addWidget(parametersStack);

        label = new QLabel(controlPanel);
        label->setObjectName("label");

        controlLayout->addWidget(label);

        comparisonSlider = new QSlider(controlPanel);
        comparisonSlider->setObjectName("comparisonSlider");
        comparisonSlider->setMaximum(100);
        comparisonSlider->setValue(50);
        comparisonSlider->setOrientation(Qt::Orientation::Horizontal);

        controlLayout->addWidget(comparisonSlider);

        applyButton = new QPushButton(controlPanel);
        applyButton->setObjectName("applyButton");

        controlLayout->addWidget(applyButton);

        timeLabel = new QLabel(controlPanel);
        timeLabel->setObjectName("timeLabel");
        timeLabel->setAlignment(Qt::AlignmentFlag::AlignCenter);

        controlLayout->addWidget(timeLabel);


        horizontalLayout->addWidget(controlPanel);

        verticalLayout = new QVBoxLayout();
        verticalLayout->setObjectName("verticalLayout");
        imageLabel = new QLabel(centralwidget);
        imageLabel->setObjectName("imageLabel");
        imageLabel->setMinimumSize(QSize(640, 480));
        imageLabel->setFrameShape(QFrame::Shape::Box);
        imageLabel->setAlignment(Qt::AlignmentFlag::AlignCenter);

        verticalLayout->addWidget(imageLabel);

        horizontalLayout_3 = new QHBoxLayout();
        horizontalLayout_3->setObjectName("horizontalLayout_3");
        pauseButton = new QPushButton(centralwidget);
        pauseButton->setObjectName("pauseButton");

        horizontalLayout_3->addWidget(pauseButton);

        videoSlider = new QSlider(centralwidget);
        videoSlider->setObjectName("videoSlider");
        videoSlider->setOrientation(Qt::Orientation::Horizontal);

        horizontalLayout_3->addWidget(videoSlider);

        videoTimeLabel = new QLabel(centralwidget);
        videoTimeLabel->setObjectName("videoTimeLabel");

        horizontalLayout_3->addWidget(videoTimeLabel);


        verticalLayout->addLayout(horizontalLayout_3);


        horizontalLayout->addLayout(verticalLayout);

        MainWindow->setCentralWidget(centralwidget);
        menubar = new QMenuBar(MainWindow);
        menubar->setObjectName("menubar");
        menubar->setGeometry(QRect(0, 0, 964, 22));
        MainWindow->setMenuBar(menubar);

        retranslateUi(MainWindow);

        parametersStack->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QCoreApplication::translate("MainWindow", "\320\236\320\261\321\200\320\260\320\261\320\276\321\202\320\272\320\260 \320\274\320\265\320\264\320\270\320\260", nullptr));
        controlPanel->setTitle(QCoreApplication::translate("MainWindow", "\320\237\320\260\320\275\320\265\320\273\321\214 \321\203\320\277\321\200\320\260\320\262\320\273\320\265\320\275\320\270\321\217", nullptr));
        loadButton->setText(QCoreApplication::translate("MainWindow", "\320\236\321\202\320\272\321\200\321\213\321\202\321\214", nullptr));
        saveButton->setText(QCoreApplication::translate("MainWindow", "\320\241\320\276\321\205\321\200\320\260\320\275\320\270\321\202\321\214", nullptr));
        filterComboBox->setItemText(0, QCoreApplication::translate("MainWindow", "\320\240\320\260\320\267\320\274\321\213\321\202\320\270\320\265 \320\277\320\276 \320\223\320\260\321\203\321\201\321\201\321\203", nullptr));
        filterComboBox->setItemText(1, QCoreApplication::translate("MainWindow", "\320\236\320\277\320\265\321\200\320\260\321\202\320\276\321\200 \320\241\320\276\320\261\320\265\320\273\321\217", nullptr));
        filterComboBox->setItemText(2, QCoreApplication::translate("MainWindow", "\320\236\320\277\320\265\321\200\320\260\321\202\320\276\321\200 \320\232\320\260\320\275\320\275\320\270", nullptr));
        filterComboBox->setItemText(3, QCoreApplication::translate("MainWindow", "\320\234\320\265\320\264\320\270\320\260\320\275\320\275\321\213\320\271 \321\204\320\270\320\273\321\214\321\202\321\200", nullptr));
        filterComboBox->setItemText(4, QCoreApplication::translate("MainWindow", "\320\230\320\267\320\274\320\265\320\275\320\265\320\275\320\270\320\265 \321\217\321\200\320\272\320\276\321\201\321\202\320\270", nullptr));
        filterComboBox->setItemText(5, QCoreApplication::translate("MainWindow", "\320\230\320\267\320\274\320\265\320\275\320\265\320\275\320\270\320\265 \320\275\320\260\321\201\321\213\321\211\320\265\320\275\320\275\320\276\321\201\321\202\320\270", nullptr));
        filterComboBox->setItemText(6, QCoreApplication::translate("MainWindow", "\320\237\320\270\320\272\321\201\320\265\320\273\320\270\320\267\320\260\321\206\320\270\321\217", nullptr));
        filterComboBox->setItemText(7, QCoreApplication::translate("MainWindow", "\320\230\320\267\320\274\320\265\320\275\320\265\320\275\320\270\320\265 \321\200\320\265\320\267\320\272\320\276\321\201\321\202\320\270", nullptr));

#if QT_CONFIG(tooltip)
        filterComboBox->setToolTip(QCoreApplication::translate("MainWindow", "\320\222\321\213\320\261\320\265\321\200\320\270\321\202\320\265 \321\204\320\270\320\273\321\214\321\202\321\200", nullptr));
#endif // QT_CONFIG(tooltip)
        implementationComboBox->setItemText(0, QCoreApplication::translate("MainWindow", "OpenCV", nullptr));
        implementationComboBox->setItemText(1, QCoreApplication::translate("MainWindow", "SIMD", nullptr));
        implementationComboBox->setItemText(2, QCoreApplication::translate("MainWindow", "\320\221\320\265\320\267 \320\261\320\270\320\261\320\273\320\270\320\276\321\202\320\265\320\272", nullptr));

#if QT_CONFIG(tooltip)
        implementationComboBox->setToolTip(QCoreApplication::translate("MainWindow", "\320\222\321\213\320\261\320\265\321\200\320\270\321\202\320\265 \321\200\320\265\320\260\320\273\320\270\320\267\320\260\321\206\320\270\321\216", nullptr));
#endif // QT_CONFIG(tooltip)
        labelGaussKernel->setText(QCoreApplication::translate("MainWindow", "\320\240\320\260\320\267\320\274\320\265\321\200 \321\217\320\264\321\200\320\260:", nullptr));
        labelGaussSigma->setText(QCoreApplication::translate("MainWindow", "\320\241\321\202\320\260\320\275\320\264\320\260\321\200\321\202\320\275\320\276\320\265 \320\276\321\202\320\272\320\273\320\276\320\275\320\265\320\275\320\270\320\265", nullptr));
        labelSobelParams->setText(QCoreApplication::translate("MainWindow", "\320\237\320\260\321\200\320\260\320\274\320\265\321\202\321\200\321\213 \320\276\321\202\321\201\321\203\321\202\321\201\321\202\320\262\321\203\321\216\321\202", nullptr));
        labelCannyLow->setText(QCoreApplication::translate("MainWindow", "\320\235\320\270\320\266\320\275\320\270\320\271 \320\277\320\276\321\200\320\276\320\263:", nullptr));
        labelCannyHigh->setText(QCoreApplication::translate("MainWindow", "\320\222\320\265\321\200\321\205\320\275\320\270\320\271 \320\277\320\276\321\200\320\276\320\263:", nullptr));
        labelMedianKernel->setText(QCoreApplication::translate("MainWindow", "\320\240\320\260\320\267\320\274\320\265\321\200 \321\217\320\264\321\200\320\260:", nullptr));
        labelBrightAlpha->setText(QCoreApplication::translate("MainWindow", "\320\232\320\276\321\215\321\204\321\204\320\270\321\206\320\270\320\265\320\275\321\202 \321\217\321\200\320\272\320\276\321\201\321\202\320\270:", nullptr));
        labelContrastAlpha->setText(QCoreApplication::translate("MainWindow", "\320\232\320\276\321\215\321\204\321\204\320\270\321\206\320\270\320\265\320\275\321\202 \320\275\320\260\321\201\321\213\321\211\320\265\320\275\320\275\320\276\321\201\321\202\320\270", nullptr));
        labelPixelSize->setText(QCoreApplication::translate("MainWindow", "\320\240\320\260\320\267\320\274\320\265\321\200 \320\261\320\273\320\276\320\272\320\260:", nullptr));
        labelSharpenParams->setText(QCoreApplication::translate("MainWindow", "\320\237\320\260\321\200\320\260\320\274\320\265\321\202\321\200\321\213 \320\276\321\202\321\201\321\203\321\202\321\201\321\202\320\262\321\203\321\216\321\202", nullptr));
        label->setText(QCoreApplication::translate("MainWindow", "\320\224\320\276 /  \320\277\320\276\321\201\320\273\320\265 \320\276\320\261\321\200\320\260\320\261\320\276\321\202\320\272\320\270", nullptr));
        applyButton->setText(QCoreApplication::translate("MainWindow", "\320\237\321\200\320\270\320\274\320\265\320\275\320\270\321\202\321\214", nullptr));
        timeLabel->setText(QCoreApplication::translate("MainWindow", "\320\222\321\200\320\265\320\274\321\217 \320\276\320\261\321\200\320\260\320\261\320\276\321\202\320\272\320\270: 0.00 \321\201", nullptr));
        imageLabel->setText(QCoreApplication::translate("MainWindow", "\320\227\320\264\320\265\321\201\321\214 \320\261\321\203\320\264\320\265\321\202 \320\276\321\202\320\276\320\261\321\200\320\260\320\266\320\260\321\202\321\214\321\201\321\217 \321\200\320\265\320\267\321\203\320\273\321\214\321\202\320\260\321\202", nullptr));
        pauseButton->setText(QCoreApplication::translate("MainWindow", "\342\226\266", nullptr));
        videoTimeLabel->setText(QCoreApplication::translate("MainWindow", "00:00 / 00:00", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
