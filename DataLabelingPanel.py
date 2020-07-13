#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog
from PyQt5.QtGui import QPainter, QPixmap, QPen
from copy import deepcopy
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
from utils import points_to_grid_svd, rasterize_3d_points

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
        grid = points_to_grid_svd(pts, start, end)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.imshow(grid)
        self.display.draw()

        self.start = start
        self.end = end
        self.current_raster = grid
        self.current_branch_raster, _ = rasterize_3d_points(pts, bounds=self.current_tree_raster_bounds)

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

        # Temp
        def get_horizontalness(k):
            p1 = graph.nodes[k[0]]['point']
            p2 = graph.nodes[k[1]]['point']
            diff = p1 - p2
            planar = np.linalg.norm(diff[[0,2]])
            return -np.abs(diff[1] / planar)

        all_edges = sorted(all_edges, key=get_horizontalness)


        # random.shuffle(all_edges)
        self.remaining_edges = all_edges

        # Get raster info for tree
        self.current_tree_raster, self.current_tree_raster_bounds = rasterize_3d_points(pts)

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






if __name__ == '__main__':

    import pymesh
    import tree_model

    # Resample all of the existing data, rebalancing categories if necessary
    TOTAL_DESIRED = 15000
    RESAMPLING_RATE = 5
    RESAMPLING_BOUNDS = [20000, 100000]

    desired_weights = np.array([2, 2, 2, 1, 2, 2.5]) # Last index is for false connections

    base_folder = '/home/main/data/tree_edge_data'
    new_folder = '/home/main/data/tree_edge_data/auxiliary'

    base_files = [f for f in os.listdir(base_folder) if f.endswith('.tree')]

    to_add = defaultdict(lambda: dict)


    # Step 1: Figure out how many of each classification we have
    current_metadata = defaultdict(list)
    for file in base_files:
        with open(os.path.join(base_folder, file), 'rb') as fh:
            data = pickle.load(fh)
        data['file'] = file
        if data['connected'][1]:
            classification = np.argmax(data['classification'])
        else:
            classification = 5

        current_metadata[classification].append(data)

    # Step 2: Count how many classifications we have and how many more we need.
    # Put them into a file-based queue for processing.

    to_process = defaultdict(list)

    total_files = (TOTAL_DESIRED * desired_weights / desired_weights.sum()).astype(np.int)
    for classification, desired in enumerate(total_files):
        existing = len(current_metadata[classification])
        remaining = max(0, desired - existing)
        to_choose = np.random.choice(existing, remaining)
        for i in to_choose:
            data = current_metadata[classification][i]
            source = data['source_file']
            to_process[source].append(deepcopy(data))

    # Step 3: For each tree, load the non-subsampled tree. For each data point, create a new dict that updates the
    # local image render and the global image render.
    count = 0
    counts_by_file = defaultdict(lambda: 0)

    for source_file, queue in to_process.items():

        random.shuffle(queue)

        base_points = pymesh.load_mesh(source_file).vertices
        base_points = tree_model.preprocess_point_cloud(base_points, downsample=False)
        base_n = len(base_points)
        points = None
        current_render = None
        bounds = None

        for iter, data in enumerate(queue):
            if not (count + 1) % 50:
                print(count + 1)

            if not iter % RESAMPLING_RATE:
                num_points = np.random.randint(RESAMPLING_BOUNDS[0], min(RESAMPLING_BOUNDS[1], base_n))
                points = base_points[np.random.choice(base_n, num_points, replace=False)]

                # Redo the raster stuff - copy and pasted
                current_render, bounds = rasterize_3d_points(points)

            near_start = np.linalg.norm(points - data['start'], axis=1) < data['radius']
            near_end = np.linalg.norm(points - data['end'], axis=1) < data['radius']
            subpoints = points[near_start | near_end]

            if len(subpoints) < 8:
                print('Dropped')
                continue

            grid = points_to_grid_svd(subpoints, data['start'], data['end'])
            global_branch_render, _ = rasterize_3d_points(subpoints, bounds)

            data['global_image'] = np.stack([current_render, global_branch_render], axis=2)
            data['local_image'] = grid

            file = data['file']
            with open(os.path.join(new_folder, '{}_{:03d}.tree'.format(file, counts_by_file[file])), 'wb') as fh:
                pickle.dump(data, fh)
            count += 1
            counts_by_file[file] += 1
