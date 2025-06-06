QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++17

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    cpu_filters.cpp

HEADERS += \
    mainwindow.h \
    cpu_filters.h

FORMS += \
    mainwindow.ui


INCLUDEPATH += D:\Programs\OpenCV_MinGW\include

# Путь к библиотекам OpenCV и сами библиотеки
LIBS += -LD:\Programs\OpenCV_MinGW\x64\mingw\bin \
    -lopencv_core455 \
    -lopencv_imgproc455 \
    -lopencv_highgui455 \
    -lopencv_imgcodecs455 \
    -lopencv_videoio455

QMAKE_CXXFLAGS += -mavx -mavx2 -mfma
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resources.qrc
