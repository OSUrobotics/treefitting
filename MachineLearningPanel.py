#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog
from PyQt5.QtGui import QPainter, QPixmap, QPen

from MyPointCloud import MyPointCloud
from Cylinder import Cylinder
from CylinderCover import CylinderCover
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from functools import partial
import imageio
import hashlib
import time

import random
import os
import sys
import pickle
import numpy as np
from tree_model import Superpoint
import hashlib
from functools import partial
from collections import defaultdict
from exp_joint_detector import convert_pc_to_grid
import imageio

class LabelAndText(QWidget):
    def __init__(self, label, starting_text=''):
        super(LabelAndText, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel(label)
        self.textbox = QLineEdit(starting_text)

        layout.addWidget(self.label)
        layout.addWidget(self.textbox)

    def __getattr__(self, attrib):
        # For easy binding to the textbox values
        return self.textbox.__getattr__(attrib)

    def text(self):
        return self.textbox.text()

    def value(self):
        return float(self.text())

class CanvasWithOverlay(QWidget):
    def __init__(self, image_size=512, hold_to_draw=False, pen_size=None, show_diagnostic_labels=True):
        """
        Initializes a canvas which can display a base image underneath it.
        :param image_size: Either an integer or a 2-tuple of integers
        :param hold_to_draw: If True, drawing will be done by holding down the mouse button.
                             Otherwise it draws every time you click
        :param pen_size: If not specified, either a pe
        """
        super(CanvasWithOverlay, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.hold_to_draw = hold_to_draw
        if isinstance(image_size, int):
            self.x = self.y = image_size
        else:
            self.x, self.y = image_size
        self.pen_size = 1
        self.pen_range = (1, 1)
        self.pen_ticks = 0
        if pen_size is not None:
            if isinstance(pen_size, int):
                self.pen_size = pen_size
            else:   # 4-tuple
                self.pen_size, low, high, self.pen_ticks = pen_size
                self.pen_range = (low, high)

        # Image label setup
        self.image_label = QLabel()
        self.image_label.setStyleSheet('padding:15px')
        layout.addWidget(self.image_label)
        self.padding = 15
        self.base_pixmap = QPixmap(self.x, self.y)
        self.base_pixmap.fill(QtCore.Qt.black)
        self.overlay_pixmap = QPixmap(self.x, self.y)
        self.overlay_pixmap.fill(QtCore.Qt.transparent)
        self.image_label.setPixmap(self.base_pixmap)
        self.image_label.mousePressEvent = self.handle_mouse_press
        self.image_label.mouseMoveEvent = self.handle_mouse_move
        self.image_label.mouseReleaseEvent = self.handle_mouse_move

        # Diagnostic label setup
        diagnostic_layout = QHBoxLayout()
        self.pen_label = QLabel('Pen size: {}px'.format(self.pen_size))
        self.mode_label = QLabel('Drawing mode')
        diagnostic_layout.addWidget(self.pen_label)
        diagnostic_layout.addWidget(self.mode_label)
        if show_diagnostic_labels:
            layout.addLayout(diagnostic_layout)

        # State tracking
        self.drawing_mode = True
        self.last_x = None
        self.last_y = None

    def draw(self, x, y):
        x -= self.padding
        y -= self.padding
        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return

        painter = QPainter(self.overlay_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = painter.pen()
        pen.setWidth(self.pen_size)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        if self.drawing_mode:
            pen.setColor(QtCore.Qt.white)
        else:
            pen.setColor(QtCore.Qt.black)
        painter.setPen(pen)
        painter.drawLine(self.last_x, self.last_y, x, y)
        painter.end()

        self.last_x, self.last_y = x, y

        combined = overlay_pixmap(self.base_pixmap, self.overlay_pixmap)
        self.image_label.setPixmap(combined)
        self.update()

    def handle_mouse_press(self, e):
        x, y = e.x(), e.y()
        if not self.hold_to_draw:
            self.draw(x, y)
        else:
            self.reset_state()

    def handle_mouse_move(self, e):
        x, y = e.x(), e.y()
        if self.hold_to_draw:
            self.draw(x, y)

    def handle_mouse_release(self, _):
        if self.hold_to_draw:
            self.reset_state()

    def reset_state(self):
        self.last_x = None
        self.last_y = None

    def reset_overlay(self):
        self.overlay_pixmap = QPixmap(self.x, self.y)
        self.overlay_pixmap.fill(QtCore.Qt.transparent)
        self.image_label.setPixmap(self.base_pixmap)
        self.reset_state()

    def update_base(self, img):
        self.base_pixmap = QPixmap(img).scaled(self.x, self.y)
        self.reset_overlay()

    def enable_erase_mode(self):
        self.mode_label.setText('Erasing mode')
        self.drawing_mode = False

    def disable_erase_mode(self):
        self.mode_label.setText('Drawing mode')
        self.drawing_mode = True

    def toggle_erase(self):
        if self.drawing_mode:
            self.enable_erase_mode()
        else:
            self.disable_erase_mode()

    def wheelEvent(self, e):
        change = e.angleDelta().y()
        if change > 0:
            new_size = self.pen_size + self.pen_ticks
        else:
            new_size = self.pen_size - self.pen_ticks
        if self.pen_range[0] <= new_size <= self.pen_range[1]:
            self.pen_size = new_size
            self.pen_label.setText('Pen size: {}px'.format(new_size))


class ML_Panel(QWidget):
    def __init__(self, callbacks=None, save_folder=None):
        super(ML_Panel, self).__init__()

        self.callbacks = callbacks
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Layout for canvas
        displays_layout = QHBoxLayout()
        self.canvas = CanvasWithOverlay(512)
        displays_layout.addWidget(self.canvas)
        layout.addLayout(displays_layout)

        # Layout for controls
        controls = QHBoxLayout()
        commit_button = QPushButton('Commit')
        skip_button = QPushButton('Skip')
        reset_button = QPushButton('Reset')
        commit_button.clicked.connect(partial(self.refresh, True))
        skip_button.clicked.connect(partial(self.refresh, False))
        reset_button.clicked.connect(self.canvas.reset_overlay)
        controls.addWidget(commit_button)
        controls.addWidget(skip_button)
        controls.addWidget(reset_button)
        layout.addLayout(controls)

        # Layout for params
        options = QHBoxLayout()
        self.box_min = LabelAndText('Min Radius', '0.05')
        self.box_max = LabelAndText('Max Radius', '0.20')
        options.addWidget(self.box_min)
        options.addWidget(self.box_max)
        layout.addLayout(options)

        # Layout for PC resampling
        resampler = QHBoxLayout()
        self.cover_radius = LabelAndText('Cover Radius', '0.10')
        self.neighbor_radius = LabelAndText('Neighbor Radius', '0.20')
        resample_button = QPushButton('Resample')
        resample_button.clicked.connect(self.resample_tree)
        resampler.addWidget(self.cover_radius)
        resampler.addWidget(self.neighbor_radius)
        resampler.addWidget(resample_button)
        layout.addLayout(resampler)


        # State variables
        self.points = np.zeros((0,3))
        self.current_grid = None
        self.save_folder = save_folder
        self.count = 0
        if save_folder is not None:
            self.count = len(os.listdir(save_folder)) // 2
        else:
            print('No save folder passed in, will not save data')


    def resample_tree(self):
        cover_radius = self.cover_radius.value()
        neighbor_radius = self.neighbor_radius.value()
        self.callbacks['resample'](cover_radius, neighbor_radius)

    def refresh(self, commit=False):
        if commit:
            self.commit()

        radius_to_pick = np.random.uniform(self.box_min.value(), self.box_max.value())
        self.points, ref = self.callbacks['random_point'](radius_to_pick)
        grid = convert_pc_to_grid(self.points, ref, grid_size=32)
        grid_max = grid.max()
        if grid_max:
            grid = grid / grid.max()
        self.current_grid = grid
        imageio.imwrite('raster.png', grid)
        self.canvas.update_base('raster.png')


    def commit(self):
        if self.current_grid is None:
            print('No grid to save!')
            return
        if self.save_folder is None:
            print('No folder to save to!')
            return

        self.canvas.overlay_pixmap.scaled(32, 32, QtCore.Qt.KeepAspectRatio,
                                          QtCore.Qt.SmoothTransformation).save('gt.png')
        image = imageio.imread('gt.png')
        if len(image.shape) > 2:
            image = image.mean(axis=2)
        im_max = image.max()
        if im_max:
            image = image / image.max()
        imageio.imsave(os.path.join(self.save_folder, '{}_t.png'.format(self.count)), image)
        imageio.imsave(os.path.join(self.save_folder, '{}.png'.format(self.count)), self.current_grid)
        self.count += 1


    def keyPressEvent(self, event):
        pressed = event.key()
        if pressed == QtCore.Qt.Key_F:
            self.canvas.reset_state()

        elif pressed == QtCore.Qt.Key_R:
            self.canvas.reset_overlay()

        elif pressed == QtCore.Qt.Key_A:
            self.refresh(commit=True)

        elif pressed == QtCore.Qt.Key_S:
            self.refresh(commit=False)




# https://stackoverflow.com/questions/53420826/overlay-two-pixmaps-with-alpha-value-using-qpainter
def overlay_pixmap(base, overlay):
    # Assumes both have same size
    rez = QPixmap(base.size())
    rez.fill(QtCore.Qt.transparent)
    painter = QPainter(rez)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.drawPixmap(QtCore.QPoint(), base)
    painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
    painter.drawPixmap(rez.rect(), overlay, overlay.rect())
    painter.end()

    return rez
