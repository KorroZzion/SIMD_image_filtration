#include "comparison_slider_widget.h"
#include <QPainter>
#include <QPaintEvent>

ComparisonSliderWidget::ComparisonSliderWidget(QWidget* parent)
    : QWidget(parent)
{
    setMinimumSize(200, 200);  // можно изменить по ситуации
}

void ComparisonSliderWidget::setBeforeImage(const QImage& img) {
    beforeImage = img;
    update();
}

void ComparisonSliderWidget::setAfterImage(const QImage& img) {
    afterImage = img;
    update();
}

void ComparisonSliderWidget::setDividerPosition(int pos) {
    dividerPos = pos;
    update();
}

void ComparisonSliderWidget::paintEvent(QPaintEvent* /* event */) {
    QPainter painter(this);

    if (!beforeImage.isNull()) {
        painter.drawImage(rect(), beforeImage);
    }

    if (!afterImage.isNull()) {
        QRect targetRect(0, 0, dividerPos, height());
        QRect sourceRect(0, 0, dividerPos * afterImage.width() / width(), afterImage.height());
        painter.drawImage(targetRect, afterImage, sourceRect);
    }

    // Рисуем вертикальную разделительную линию
    painter.setPen(QPen(Qt::red, 2));
    painter.drawLine(dividerPos, 0, dividerPos, height());
}
