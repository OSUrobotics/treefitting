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

from MachineLearningPanel import LabelAndText

#
# self.figure = Figure()
#         self.display = FigureCanvas(self.figure)

class DataLabelingPanel(QWidget):

    ALL_CLASSES = ['False', 'Trunk', 'Support', 'Leader', 'Side', 'Other']

    def __init__(self, callbacks=None, save_folder=None):
        super(DataLabelingPanel, self).__init__()

        self.callbacks = callbacks
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Layout for canvas
        displays_layout = QHBoxLayout()

        self.figure = Figure()
        self.display = FigureCanvas(self.figure)
        self.figure.clear()
        displays_layout.addWidget(self.display)

        # Layout for data labels
        labels = QHBoxLayout()
        for val, label in enumerate(self.ALL_CLASSES):
            button = QPushButton(label)
            button.clicked.connect(partial(self.commit, val))
            labels.addWidget(button)

        layout.addLayout(labels)

        # Layout for controls
        controls = QGridLayout()
        self.left = LabelAndText('Left')
        self.right = LabelAndText('Right')
        self.up = LabelAndText('Up')
        self.down = LabelAndText('Down')
        resample = QPushButton('Resample')
        resample.clicked.connect(self.refresh)
        regen_graph = QPushButton('Regen Graph')
        regen_graph.clicked.connect(self.regen_graph)

        controls.addWidget(self.left, 0, 0)
        controls.addWidget(self.right, 0, 1)
        controls.addWidget(self.up, 1, 0)
        controls.addWidget(self.down, 1, 1)
        controls.addWidget(resample, 0, 2)
        controls.addWidget(regen_graph, 1, 2)


        # State variables
        self.start = None
        self.end = None
        self.current_raster = None
        self.current_tree_raster = None


        self.points = np.zeros((0,3))
        self.current_grid = None
        self.save_folder = save_folder
        self.count = 0
        if save_folder is not None:
            self.count = len(os.listdir(save_folder))
        else:
            print('No save folder passed in, will not save data')

    def refresh(self):
        return
        radius_to_pick = np.random.uniform(self.box_min.value(), self.box_max.value())
        self.points, ref = self.callbacks['random_point'](radius_to_pick)
        grid = convert_pc_to_grid(self.points, ref, grid_size=32)
        grid_max = grid.max()
        if grid_max:
            grid = grid / grid.max()
        self.current_grid = grid
        imageio.imwrite('raster.png', grid)
        self.canvas.update_base('raster.png')


    def commit(self, val):
        print(val)
        # if self.current_grid is None:
        #     print('No grid to save!')
        #     return
        # if self.save_folder is None:
        #     print('No folder to save to!')
        #     return
        #
        # self.canvas.overlay_pixmap.scaled(32, 32, QtCore.Qt.KeepAspectRatio,
        #                                   QtCore.Qt.SmoothTransformation).save('gt.png')
        # image = imageio.imread('gt.png')
        # if len(image.shape) > 2:
        #     image = image.mean(axis=2)
        # im_max = image.max()
        # if im_max:
        #     image = image / image.max()
        # imageio.imsave(os.path.join(self.save_folder, '{}_t.png'.format(self.count)), image)
        # imageio.imsave(os.path.join(self.save_folder, '{}.png'.format(self.count)), self.current_grid)
        # self.count += 1

    def regen_graph(self):
        pass

    def keyPressEvent(self, event):
        pressed = event.key()
        if QtCore.Qt.Key_0 <= pressed <= QtCore.Qt.Key_9:
            val = int(chr(pressed))
            if val < len(self.ALL_CLASSES):
                self.commit(val)



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
