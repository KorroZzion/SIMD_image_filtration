#ifndef COMPARISON_SLIDER_WIDGET_H
#define COMPARISON_SLIDER_WIDGET_H

#include <QWidget>
#include <QImage>

class ComparisonSliderWidget : public QWidget {
    Q_OBJECT

public:
    explicit ComparisonSliderWidget(QWidget* parent = nullptr);

    void setBeforeImage(const QImage& img);
    void setAfterImage(const QImage& img);
    void setDividerPosition(int pos);  // положение слайдера от 0 до ширины

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QImage beforeImage;
    QImage afterImage;
    int dividerPos = 0;
};

#endif // COMPARISON_SLIDER_WIDGET_H
