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
from exp_joint_detector import project_point_onto_plane, project_points_onto_normal
import imageio
from scipy.linalg import svd

from MachineLearningPanel import LabelAndText

#
# self.figure = Figure()
#         self.display = FigureCanvas(self.figure)

class DataLabelingPanel(QWidget):

    ALL_CLASSES = ['Trunk', 'Support', 'Leader', 'Side', 'Other', 'False']

    def __init__(self, callbacks=None, save_folder=None):
        super(DataLabelingPanel, self).__init__()

        self.callbacks = callbacks
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Layout for canvas
        displays_layout = QHBoxLayout()

        self.figure = Figure((6.4, 4.8))
        self.display = FigureCanvas(self.figure)
        self.figure.clear()
        displays_layout.addWidget(self.display)
        layout.addLayout(displays_layout)

        # Layout for data labels
        labels = QHBoxLayout()
        for val, label in enumerate(self.ALL_CLASSES):
            button = QPushButton('{} ({})'.format(label, val))
            button.clicked.connect(partial(self.commit, val))
            labels.addWidget(button)

        layout.addLayout(labels)

        # Layout for controls
        controls = QGridLayout()
        self.left = LabelAndText('Left')
        self.right = LabelAndText('Right')
        self.up = LabelAndText('Up')
        self.down = LabelAndText('Down')
        resample = QPushButton('Next')
        resample.clicked.connect(self.refresh)
        regen_graph = QPushButton('Regen Graph')
        regen_graph.clicked.connect(self.regen_graph)

        controls.addWidget(self.left, 0, 0)
        controls.addWidget(self.right, 0, 1)
        controls.addWidget(self.up, 1, 0)
        controls.addWidget(self.down, 1, 1)
        controls.addWidget(resample, 0, 2)
        controls.addWidget(regen_graph, 1, 2)

        layout.addLayout(controls)

        # State variables
        self.current_data = None
        self.start = None
        self.end = None
        self.remaining_edges = []
        self.current_raster = None
        self.current_tree_raster = None
        self.current_branch_raster = None
        self.current_tree_raster_bounds = None

        self.save_folder = save_folder
        self.count = 0
        if save_folder is not None:
            self.count = len(os.listdir(save_folder))
        else:
            print('No save folder passed in, will not save data')

    def refresh(self):
        if not self.remaining_edges:
            self.regen_graph()
        edge = self.remaining_edges.pop()
        pts, start, end = self.callbacks['highlight_edge'](edge)

        main_axis = start - end
        main_axis = main_axis / np.linalg.norm(main_axis)
        projected = project_points_onto_normal(start, main_axis, pts)
        secondary_axis = svd(projected - projected.mean())[2][0]

        all_pts = project_point_onto_plane((start + end) / 2, main_axis, secondary_axis, pts)
        all_pts = all_pts / (np.linalg.norm(start - end) / 2)   # Makes endpoints at (-1, 0), (1, 0)
        bounds_x = np.linspace(-1.5, 1.5, 32+1)
        bounds_y = np.linspace(-0.75, 0.75, 16+1)

        grid = np.histogram2d(all_pts[:,0], all_pts[:,1], bins=[bounds_x, bounds_y])[0]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(grid)
        self.display.draw()

        self.start = start
        self.end = end
        self.current_raster = grid

        self.current_branch_raster = np.histogram2d(pts[:,0], pts[:,1], self.current_tree_raster_bounds)[0]
        self.current_branch_raster = self.current_branch_raster / self.current_branch_raster.max()

    def commit(self, val):

        is_connected = np.array([True, False])
        classification = np.zeros(len(self.ALL_CLASSES) - 1, dtype=np.float)
        if val != len(self.ALL_CLASSES) - 1:
            is_connected = np.array([False, True])
            classification[val] = 1.0

        rez = {
            'classification': classification,
            'connected': is_connected,
            'global_image': np.stack([self.current_tree_raster, self.current_branch_raster], axis=2),
            'local_image': self.current_raster,
            'start': self.start,
            'end': self.end,
            'source_file': self.current_data['source_file'],
            'radius': self.current_data['radius'],
        }



        while True:
            file_name = '{:06d}.tree'.format(self.count)
            file_path = os.path.join(self.save_folder, file_name)
            if os.path.exists(file_path):
                self.count += 1
                continue
            with open(file_path, 'wb') as fh:
                pickle.dump(rez, fh)

            print('Saved to {}'.format(file_path))
            for k in sorted(rez):
                if not isinstance(rez[k], np.ndarray):
                    print('{}: {}'.format(k, rez[k]))

            break

        self.refresh()

    def regen_graph(self):
        data = self.callbacks['refresh_superpoints']()
        graph = data['graph']
        pts = data['points']

        all_edges = list(graph.edges)
        random.shuffle(all_edges)
        self.remaining_edges = all_edges

        # Get raster info for tree
        zplane_pts = pts[:,:2]
        x_max, y_max = zplane_pts.max(axis=0)
        x_min, y_min = zplane_pts.min(axis=0)
        scale = max(x_max - x_min, y_max - y_min)
        x_cen, y_cen = (x_max + x_min) / 2, (y_max + y_min) / 2

        bounds_x = np.linspace(x_cen - scale/2, x_cen + scale/2, 129)
        bounds_y = np.linspace(y_cen - scale/2, y_cen + scale/2, 129)
        self.current_tree_raster_bounds = [bounds_x, bounds_y]

        self.current_tree_raster = np.histogram2d(zplane_pts[:,0], zplane_pts[:,1], self.current_tree_raster_bounds)[0]
        self.current_tree_raster = self.current_tree_raster / self.current_tree_raster.max()
        self.current_data = data




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
