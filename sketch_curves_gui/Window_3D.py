import sys

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)

import OpenGL.GL as GL

import numpy as np


class Window_3D(QWidget):
    def __init__(self, class_type):
        super(Window_3D, self).__init__()

        self.glWidget = class_type(self)

        self.up_down = self.create_slider()
        self.turntable = self.create_slider()

        self.up_down.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.setValue)
        self.turntable.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.setValue)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.glWidget)
        main_layout.addWidget(self.turntable)
        main_layout.addWidget(self.up_down)
        self.setLayout(main_layout)

        self.up_down.setValue(15 * 16)
        self.turntable.setValue(345 * 16)

        self.setWindowTitle("Hello GL")

    @staticmethod
    def create_slider():
        slider = QSlider(Qt.Vertical)

        slider.setRange(0, 360 * 16)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setTickPosition(QSlider.TicksRight)

        return slider
