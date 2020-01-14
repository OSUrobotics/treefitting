#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QOpenGLWidget, QSlider, QWidget

from MyPointCloud import MyPointCloud
from Cylinder import Cylinder
from CylinderCover import CylinderCover

from DrawPointCloud import DrawPointCloud

import numpy as np

from MySliders import SliderIntDisplay, SliderFloatDisplay

class PointCloudViewerGUI(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowTitle('Point cloud Viewer')

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        middle_layout = self._init_middle_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        # Two side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)

        self.connected_neighborhood_radius.set_value( self.glWidget.my_pcd.radius_neighbor )

        SliderFloatDisplay.gui = self
        SliderIntDisplay.gui = self

    # Set up the left set of sliders/buttons (parameters)
    def _init_left_layout_(self):
        # Two reset buttons one for probabilities, one for doors
        recalc_neighbors_button = QPushButton('Recalculate neighbors')
        recalc_neighbors_button.clicked.connect(self.recalc_neighbors)

        recalc_cylinder_button = QPushButton('Recalculate cylinder')
        recalc_cylinder_button.clicked.connect(self.recalc_cylinder)

        new_id_button = QPushButton('Random new id')
        new_id_button.clicked.connect(self.new_random_id)

        resets = QGroupBox('Resets')
        resets_layout = QVBoxLayout()
        resets_layout.addWidget(recalc_neighbors_button)
        resets_layout.addWidget(recalc_cylinder_button)
        resets_layout.addWidget(new_id_button)
        resets.setLayout(resets_layout)

        # For setting the bin size, based on width of narrowest branch
        self.smallest_branch_width = SliderFloatDisplay('Width small branch', 0.01, 0.1, 0.015)
        self.largest_branch_width = SliderFloatDisplay('Width big branch', 0.05, 0.2, 0.1)
        self.branch_height = SliderFloatDisplay('Height cyl fit', 0.1, 0.3, 0.15)

        params_neighbors = QGroupBox('Neighbor parameters')
        params_neighbors_layout = QVBoxLayout()
        params_neighbors_layout.addWidget(self.smallest_branch_width)
        params_neighbors_layout.addWidget(self.largest_branch_width)
        params_neighbors_layout.addWidget(self.branch_height)
        params_neighbors.setLayout(params_neighbors_layout)

        # The parameters of the cylinder fit
        params_labels = QGroupBox('Connected parameters                  ')
        params_labels_layout = QVBoxLayout()
        self.connected_neighborhood_radius = SliderFloatDisplay('K Neigh Radius', 0.01, 0.2, 0.05)
        self.eigen_threshold = SliderFloatDisplay('Eigen threshold big name big name', 0.01, 0.99, 0.8)

        params_labels_layout.addWidget(self.connected_neighborhood_radius)
        params_labels_layout.addWidget(self.eigen_threshold)

        params_labels.setLayout(params_labels_layout)

        # Sliders for Camera
        self.turntable = SliderFloatDisplay('Rotate turntable', 0.0, 360, 0, 361)
        self.up_down = SliderFloatDisplay('Up down', 0, 360, 0, 361)
        self.zoom = SliderFloatDisplay('Zoom', 0.6, 2.0, 1.0)

        show_buttons = QGroupBox('Show buttons')
        show_buttons_layout = QGridLayout()

        show_closeup_button = QCheckBox('Show closeup')
        show_closeup_button.clicked.connect(self.show_closeup)

        show_one_button = QCheckBox('Show one')
        show_one_button.clicked.connect(self.show_one)

        show_pca_cyl_button = QCheckBox('Show PCA cylinders')
        show_pca_cyl_button.clicked.connect(self.show_pca_cylinders)

        show_fitted_cyl_button = QCheckBox('Show fitted cylinders')
        show_fitted_cyl_button.clicked.connect(self.show_fitted_cylinders)

        show_bins_button = QCheckBox('Show bins')
        show_bins_button.clicked.connect(self.show_bins)

        show_isolated_button = QCheckBox('Show isolated')
        show_isolated_button.clicked.connect(self.show_isolated)

        show_buttons_layout.addWidget(show_closeup_button)
        show_buttons_layout.addWidget(show_one_button)
        show_buttons_layout.addWidget(show_pca_cyl_button)
        show_buttons_layout.addWidget(show_fitted_cyl_button)
        show_buttons_layout.addWidget(show_bins_button)
        show_buttons_layout.addWidget(show_isolated_button)
        show_buttons.setLayout(show_buttons_layout)

        self.show_closeup_slider = SliderIntDisplay("Sel", 0, 10, 0)

        params_camera = QGroupBox('Camera parameters')
        params_camera_layout = QVBoxLayout()
        params_camera_layout.addWidget(show_buttons)
        params_camera_layout.addWidget(self.show_closeup_slider)
        params_camera_layout.addWidget(show_bins_button)
        params_camera_layout.addWidget(show_pca_cyl_button)
        params_camera_layout.addWidget(show_fitted_cyl_button)
        params_camera_layout.addWidget(self.turntable)
        params_camera_layout.addWidget(self.up_down)
        params_camera_layout.addWidget(self.zoom)
        params_camera.setLayout(params_camera_layout)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()

        left_side_layout.addWidget(resets)
        left_side_layout.addStretch()
        left_side_layout.addWidget(params_neighbors)
        left_side_layout.addWidget(params_labels)
        left_side_layout.addWidget(params_camera)

        return left_side_layout

    # Drawing screen and quit button
    def _init_middle_layout_(self):
        # The display for the robot drawing
        self.glWidget = DrawPointCloud( self )

        self.up_down.slider.valueChanged.connect(self.glWidget.setUpDownRotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.slider.setValue)
        self.turntable.slider.valueChanged.connect(self.glWidget.setTurntableRotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.slider.setValue)
        self.zoom.slider.valueChanged.connect(self.redraw_self)

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget( self.glWidget )
        mid_layout.addWidget(quit_button)

        return mid_layout

    # set robot back in middle
    def recalc_neighbors(self):
        self.glWidget.my_pcd.create_bins(self.smallest_branch_width.value())
        self.glWidget.my_pcd.find_neighbors()
        self.glWidget.make_bin_gl_list()
        self.repaint()

    def recalc_cylinder(self):
        if not hasattr( self.glWidget, "selected_point"):
            self.new_random_id()
        pt_ids = self.glWidget.my_pcd.find_connected( self.glWidget.selected_point, self.connected_neighborhood_radius.value() )
        self.glWidget.cyl = self.glWidget.my_pcd.fit_cylinder(self.glWidget.selected_point, pt_ids.keys())
        print( self.glWidget.cyl )
        self.glWidget.update()
        self.repaint()

    def new_random_id(self):
        id = np.random.uniform(0, len(self.glWidget.my_pcd.pc_data) )
        self.glWidget.selected_point = int( np.floor( id ) )
        self.glWidget.update()
        self.repaint()

    def set_closeup_slider(self):
        if self.glWidget.show_bins:
            self.show_closeup_slider.setMaximum(len(self.glWidget.bin_mapping))
        if self.glWidget.show_pca_cylinders:
            self.show_closeup_slider.setMaximum(len(self.glWidget.cyl_cover.cyls_pca))
        if self.glWidget.show_fitted_cylinders:
            self.show_closeup_slider.setMaximum(len(self.glWidget.cyl_cover.cyls_fitted))

    def show_closeup(self):
        self.glWidget.show_closeup = not self.glWidget.show_closeup
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_one(self):
        self.glWidget.show_one = not self.glWidget.show_one
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_pca_cylinders(self):
        self.glWidget.show_pca_cylinders = not self.glWidget.show_pca_cylinders
        self.glWidget.show_fitted_cylinders = False
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_fitted_cylinders(self):
        self.glWidget.show_fitted_cylinders = not self.glWidget.show_fitted_cylinders
        self.glWidget.show_pca_cylinders = False
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_bins(self):
        self.glWidget.show_bins = not self.glWidget.show_bins
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def show_isolated(self):
        self.glWidget.show_isolated = not self.glWidget.show_isolated
        self.glWidget.update()
        self.repaint()

    def redraw_self(self):
        self.glWidget.update()
        self.repaint()


if __name__ == '__main__':
    app = QApplication([])

    gui = PointCloudViewerGUI()

    gui.show()

    app.exec_()
