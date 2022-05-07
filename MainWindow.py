#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QComboBox
from PyQt5.QtCore import QRect, QTimer
from MyPointCloud import MyPointCloud
from Cylinder import Cylinder
from CylinderCover import CylinderCover
from PyQt5.QtGui import QColor

from functools import partial
import imageio
import hashlib
import time

import random
import os
from DrawPointCloud import DrawPointCloud
import sys
import pickle
import numpy as np
from MySliders import SliderIntDisplay, SliderFloatDisplay
import hashlib
from functools import partial
from collections import defaultdict
from DataLabelingPanel import DataLabelingPanel
from PointCloudManagementPanel import PointCloudManagementPanel
from PointCloudAnnotator import PointCloudAnnotator


ROOT = os.path.dirname(os.path.realpath(__file__))
CONFIG = os.path.join(ROOT, 'config')
if not os.path.exists(CONFIG):
    os.mkdir(CONFIG)





class PointCloudViewerGUI(QMainWindow):
    def __init__(self, **kwargs):
        QMainWindow.__init__(self)
        self.setWindowTitle('Point cloud Viewer')
        self.settings = kwargs
        self.config_dir = ''
        self.current_file = None
        self.current_superpoint = None
        self.net = None
        self.temp_process = None

        # Control buttons for the interface
        left_side_layout = self._init_left_layout_()
        middle_layout = self._init_middle_layout_()
        right_side_layout = self._init_right_layout_()

        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)

        # Two side-by-side panes
        top_level_layout = QHBoxLayout()
        widget.setLayout(top_level_layout)

        top_level_layout.addLayout(left_side_layout)
        top_level_layout.addLayout(middle_layout)
        top_level_layout.addLayout(right_side_layout)

        SliderFloatDisplay.gui = self
        SliderIntDisplay.gui = self

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_left_layout_(self):
        # For reading and writing

        try:
            with open('last_config', 'rb') as fh:
                last_config = pickle.load(fh)
        except FileNotFoundError:
            last_config = {}

        path_names = QGroupBox('File names')
        path_names_layout = QGridLayout()
        path_names_layout.setColumnMinimumWidth(0, 40)
        path_names_layout.setColumnMinimumWidth(1, 200)
        self.path_name = QLineEdit(last_config.get('fname', ''))
        # self.pcd_name = QLineEdit("cloud_final")
        self.version_name = QLineEdit(last_config.get('ver', ''))
        path_names_layout.addWidget(QLabel("Path dir:"))
        path_names_layout.addWidget(self.path_name)
        # path_names_layout.addWidget(QLabel("PCD name:"))
        # path_names_layout.addWidget(self.pcd_name)
        path_names_layout.addWidget(QLabel("Version:"))
        path_names_layout.addWidget(self.version_name)

        self.tf_widget = PointCloudOrientationWidget()
        path_names_layout.addWidget(self.tf_widget)

        path_names.setLayout(path_names_layout)

        read_point_cloud_button = QPushButton('Read point cloud')
        read_point_cloud_button.clicked.connect(self.read_point_cloud)

        pc_management_callbacks = {
            'update_pc': self.update_pc,
            'create_new_tree': self.create_new_tree,
            'save_config': self.save_config,
            'skeletonize': self.skeletonize,
            'save_skeleton': self.save_active_skeleton,
            'update_repair_mode': self.update_repair_mode,
            'apply_polygons': self.apply_polygons,
            'enable_polygon_mode': self.enable_polygon_mode,
            'replay_history': self.replay_history,
            'load_results_dict': self.load_results_dict,
            'get_current_graph': self.get_current_graph,
            'set_trunk_node': self.set_trunk_node,
        }
        self.pc_management_panel = PointCloudManagementPanel(pc_management_callbacks)
        self.pc_management_panel.hide()
        pc_management_button = QPushButton('PC Management Panel')
        pc_management_button.clicked.connect(partial(self.toggle_window, self.pc_management_panel))

        self.replay_timer = QTimer()
        self.replay_counter = 0
        self.replay_timer.timeout.connect(self.replay_history_update)

        #
        # read_cylinders_pca_button = QPushButton('Read pca cylinders')
        # read_cylinders_pca_button.clicked.connect(self.read_pca_cylinders)
        #
        # read_cylinders_fit_button = QPushButton('Read fit cylinders')
        # read_cylinders_fit_button.clicked.connect(self.read_fit_cylinders)

        file_io = QGroupBox('File io')
        file_io_layout = QVBoxLayout()
        file_io_layout.addWidget(path_names)
        file_io_layout.addWidget(read_point_cloud_button)
        file_io_layout.addWidget(pc_management_button)
        # file_io_layout.addWidget(read_cylinders_pca_button)
        # file_io_layout.addWidget(read_cylinders_fit_button)
        file_io.setLayout(file_io_layout)

        diagnostic_box = QGroupBox('diagnostics')
        diagnostic_box_layout = QVBoxLayout()
        self.loaded_hash_label = QLabel("File hash loaded: None")
        self.pca_loaded_label = QLabel("PCA Cylinders Loaded: False")
        self.fit_loaded_label = QLabel("Fit Cylinders Loaded: False")
        diagnostic_box_layout.addWidget(self.loaded_hash_label)
        diagnostic_box_layout.addWidget(self.pca_loaded_label)
        diagnostic_box_layout.addWidget(self.fit_loaded_label)
        diagnostic_box.setLayout(diagnostic_box_layout)


        # Sliders for Camera
        self.turntable = SliderFloatDisplay('Rotate turntable', 0.0, 360, 180, 361)
        self.up_down = SliderFloatDisplay('Up down', 0, 360, 180, 361)
        self.zoom = SliderFloatDisplay('Zoom', 0.6, 2.0, 1.0)

        show_buttons = QGroupBox('Show buttons')
        show_buttons_layout = QGridLayout()

        show_closeup_button = QCheckBox('Show closeup')
        show_closeup_button.clicked.connect(self.show_closeup)

        show_one_button = QCheckBox('Show one')
        show_one_button.clicked.connect(self.show_one)

        self.show_points_button = QCheckBox('Show points')
        self.show_points_button.setChecked(True)
        self.show_points_button.clicked.connect(self.show_points)

        show_pca_cyl_button = QCheckBox('Show PCA cylinders')
        show_pca_cyl_button.clicked.connect(self.show_pca_cylinders)

        show_fitted_cyl_button = QCheckBox('Show fitted cylinders')
        show_fitted_cyl_button.clicked.connect(self.show_fitted_cylinders)

        show_skeleton_button = QCheckBox('Show Skeleton')
        show_skeleton_button.clicked.connect(self.show_skeleton)

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

        self.show_min_val_slider = SliderFloatDisplay("Min", 1, 15, 5)
        self.show_min_val_slider.b_recalc_ids = True
        self.show_max_val_slider = SliderFloatDisplay("Max", 1, 15, 5)
        self.show_max_val_slider.b_recalc_ids = True

        params_camera = QGroupBox('Camera parameters')
        params_camera_layout = QVBoxLayout()
        params_camera_layout.addWidget(show_buttons)
        params_camera_layout.addWidget(self.show_closeup_slider)
        params_camera_layout.addWidget(self.show_min_val_slider)
        params_camera_layout.addWidget(self.show_max_val_slider)
        params_camera_layout.addWidget(show_bins_button)
        params_camera_layout.addWidget(show_pca_cyl_button)
        params_camera_layout.addWidget(show_fitted_cyl_button)
        params_camera_layout.addWidget(show_skeleton_button)
        params_camera_layout.addWidget(self.show_points_button)
        params_camera_layout.addWidget(self.zoom)
        params_camera_layout.addWidget(self.turntable)
        params_camera_layout.addWidget(self.up_down)

        params_camera.setLayout(params_camera_layout)

        # Put all the pieces in one box
        left_side_layout = QVBoxLayout()

        left_side_layout.addWidget(file_io)
        left_side_layout.addWidget(diagnostic_box)
        left_side_layout.addStretch()
        left_side_layout.addWidget(params_camera)

        return left_side_layout

    # Drawing screen and quit button
    def _init_middle_layout_(self):
        # The display for the robot drawing
        self.glWidget = DrawPointCloud(self, pcd_file=self.settings.get('pcd_file', None),
                                       cover_file=self.settings.get('cover_file', None))

        self.glWidget.polygon_complete_callback = self.pc_management_panel.update_polygons
        self.glWidget.set_trunk_node_callback = self.pc_management_panel.update_trunk

        self.up_down.slider.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.slider.setValue)
        self.turntable.slider.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.slider.setValue)
        self.zoom.slider.valueChanged.connect(self.redraw_self)

        self.glWidget.set_up_down_rotation(180)
        self.glWidget.set_turntable_rotation(180)

        quit_button = QPushButton('Quit')
        quit_button.clicked.connect(app.exit)

        # Put them together, quit button on the bottom
        mid_layout = QVBoxLayout()

        mid_layout.addWidget( self.glWidget )
        mid_layout.addWidget(quit_button)

        return mid_layout

    # Set up the left set of sliders/buttons (read/write, camera)
    def _init_right_layout_(self):
        # Recalculate the bins (MyPointCloud) then recalculae
        # cylinders using PCA criteria then fit those
        recalc_neighbors_button = QPushButton('Recalculate bins')
        recalc_neighbors_button.clicked.connect(self.recalc_bins)

        recalc_cylinder_button = QPushButton('Recalculate one cylinder')
        recalc_cylinder_button.clicked.connect(self.recalc_one_cylinder)

        recalc_pca_cylinder_button = QPushButton('Recalculate pca cylinder')
        recalc_pca_cylinder_button.clicked.connect(self.recalc_pca_cylinder)

        recalc_fit_cylinder_button = QPushButton('Recalculate fit cylinder')
        recalc_fit_cylinder_button.clicked.connect(self.recalc_fit_cylinder)

        new_id_button = QPushButton('Random new id')
        new_id_button.clicked.connect(self.new_random_id)

        compute_mesh_button = QPushButton('Compute mesh')
        compute_mesh_button.clicked.connect(self.compute_mesh)

        self.save_as_field = QLineEdit('')

        magic_button = QPushButton('Click here for magic')
        magic_button.clicked.connect(self.magic)

        ml_labeling_callbacks = {
            'refresh_superpoints': self.reload_superpoint_graph,
            'highlight_edge': self.highlight_edge
        }
        self.ml_labeling_panel = DataLabelingPanel(ml_labeling_callbacks, '/home/main/data/tree_edge_data')
        self.ml_labeling_panel.hide()
        labeling_button = QPushButton('Labeling Panel')
        labeling_button.clicked.connect(partial(self.toggle_window, self.ml_labeling_panel))
        annotator_callbacks = {
            'read_point_cloud': self.read_point_cloud,
            'load_tree': self.load_tree,
            'rotate_turntable': self.rotate_turntable,
        }
        self.annotation_panel = PointCloudAnnotator(self.glWidget, annotator_callbacks)
        self.annotation_panel.hide()
        annotation_button = QPushButton('Annotation Panel')
        annotation_button.clicked.connect(partial(self.toggle_window, self.annotation_panel))


        resets = QGroupBox('Resets')
        resets_layout = QVBoxLayout()
        # resets_layout.addWidget(recalc_neighbors_button)
        # resets_layout.addWidget(recalc_cylinder_button)
        # resets_layout.addWidget(recalc_pca_cylinder_button)
        # resets_layout.addWidget(recalc_fit_cylinder_button)
        # resets_layout.addWidget(new_id_button)
        resets_layout.addWidget(compute_mesh_button)
        resets_layout.addWidget(self.save_as_field)
        resets_layout.addWidget(magic_button)
        resets_layout.addWidget(labeling_button)
        resets_layout.addWidget(annotation_button)
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
        self.mark_neighbor_pca = SliderFloatDisplay('Mark Neighbor PCA', 0.1, 0.9, 0.75)
        self.mark_neighbor_fit = SliderFloatDisplay('Mark Neighbor Fit', 0.1, 0.9, 0.5)

        params_labels_layout.addWidget(self.mark_neighbor_pca)
        params_labels_layout.addWidget(self.mark_neighbor_fit)

        params_labels.setLayout(params_labels_layout)

        # Put all the pieces in one box
        right_side_layout = QVBoxLayout()

        right_side_layout.addWidget(resets)
        right_side_layout.addStretch()
        right_side_layout.addWidget(params_neighbors)
        right_side_layout.addWidget(params_labels)

        params_neighbors.hide()
        params_labels.hide()

        return right_side_layout

    @property
    def pca_cylinder_file(self):
        return os.path.join(self.config_dir, 'cylinders_pca.txt')

    @property
    def fit_cylinder_file(self):
        return os.path.join(self.config_dir, 'cylinders_fit.txt')

    def toggle_window(self, window):
        window.setGeometry(QRect(100, 100, 400, 200))
        window.show()

    # Recalcualte the bins based on the current smallest branch value
    def recalc_bins(self):
        self.glWidget.my_pcd.create_bins(self.smallest_branch_width.value())
        self.glWidget.make_bin_gl_list()
        self.repaint()

    def recalc_one_cylinder(self):
        if not hasattr(self.glWidget, "selected_point"):
            self.new_random_id()
        pt_ids = self.glWidget.my_pcd.find_connected( self.glWidget.selected_point, self.connected_neighborhood_radius.value() )
        self.glWidget.cyl = self.glWidget.my_pcd.fit_cylinder(self.glWidget.selected_point, [ret[0] for ret in pt_ids])
        print( self.glWidget.cyl )
        self.glWidget.update()
        self.repaint()

    def recalc_pca_cylinder(self):
        print('Running PCA stuff...')
        self.glWidget.cyl_cover.find_good_pca(0.5, self.height(), self.smallest_branch_width.value(), self.largest_branch_width.value())
        print('All done running PCA stuff!')

        self.glWidget.cyl_cover.write(self.config_dir)
        # with open(self.pca_cylinder_file, "w") as fid:
        #     self.glWidget.cyl_cover.write(fid)
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def recalc_fit_cylinder(self):
        self.glWidget.cyl_cover.optimize_cyl()
        self.glWidget.cyl_cover.write(self.config_dir)

        # with open(self.fit_cylinder_file, "w") as fid:
        #     self.glWidget.cyl_cover.write(fid)
        self.set_closeup_slider()
        self.glWidget.update()
        self.repaint()

    def new_random_id(self):
        # self.glWidget.initialize_mesh()
        id = np.random.uniform(0, len(self.glWidget.my_pcd.pc_data) )
        self.glWidget.selected_point = int( np.floor( id ) )
        self.glWidget.update()
        self.repaint()

    def set_closeup_slider(self):
        if self.glWidget.show_bins:
            self.show_closeup_slider.setMaximum(len(self.glWidget.bin_mapping))
            self.show_min_val_slider.set_range(1, 45)
            self.show_max_val_slider.set_range(10, 65)
        if self.glWidget.show_pca_cylinders:
            self.show_closeup_slider.setMaximum(len(self.glWidget.cyl_cover.cyls_pca))
            self.show_min_val_slider.set_range(0.0, 30.0)
            self.show_max_val_slider.set_range(4.0, 50.0)
        if self.glWidget.show_fitted_cylinders:
            self.show_closeup_slider.setMaximum(len(self.glWidget.cyl_cover.cyls_fitted))
            self.show_min_val_slider.set_range(0.0, 10.0)
            self.show_max_val_slider.set_range(1.0, 10.0)

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

    def show_points(self):
        self.glWidget.show_points = self.show_points_button.isChecked()
        self.glWidget.update()
        self.repaint()

    def show_skeleton(self):
        self.glWidget.show_skeleton = not self.glWidget.show_skeleton
        self.glWidget.initialize_skeleton()
        self.glWidget.update()
        self.repaint()


    def compute_mesh(self):
        self.glWidget.compute_mesh()
        save_text = self.save_as_field.text().strip()
        if save_text:
            self.glWidget.save_mesh(save_text)

    @staticmethod
    def get_file_hash(fname):
        fname = fname.strip()
        file_hash = hashlib.md5(fname.encode('utf-8')).hexdigest()[:16]
        return file_hash

    def read_point_cloud(self, *_, fname=None, config=None):
        if fname is None:
            fname = self.path_name.text().strip()

        self.current_file = fname
        file_base = os.path.normpath(fname).split(os.sep)[-2]
        print('Base: {}'.format(file_base))
        file_hash = self.get_file_hash(file_base)
        ver = self.version_name.text().strip()
        if ver:
            file_hash += '_{}'.format(ver)
        config_dir = os.path.join(CONFIG, file_hash)
        self.config_dir = config_dir
        self.loaded_hash_label.setText('File hash: {}'.format(file_hash))

        adjustment = self.tf_widget.get_desired_transform()

        if os.path.exists(os.path.join(config_dir, 'blah.txt')):
            pass
        else:
            try:
                self.glWidget.my_pcd.load_point_cloud(fname, adjustment=adjustment)
            except OSError:
                print('Couldn\'t find file!')
                return

            self.pc_management_panel.set_bounds_from_pc(self.glWidget.my_pcd.points)

            if not os.path.exists(config_dir):
                os.mkdir(config_dir)
            config_file = os.path.join(self.config_dir, 'config.pickle')
            try:
                if config is None:
                    with open(config_file, 'rb') as fh:
                        config = pickle.load(fh)
                self.pc_management_panel.load_config(config)
                print('Loaded settings from config!')
            except IOError:
                self.pc_management_panel.load_config({})

            self.glWidget.refresh_downsampled_points()

            # self.glWidget.reset_model()
            self.glWidget.my_pcd.create_bins(self.smallest_branch_width.value())
            self.glWidget.make_pcd_gl_list()
            self.glWidget.cyl_cover = CylinderCover(self.glWidget.my_pcd)

        config_update = {
            'fname': fname,
            'ver': ver,
        }

        with open('last_config', 'wb') as fh:
            pickle.dump(config_update, fh)
        # self.glWidget.reset_model()
        self.glWidget.update()
        self.repaint()

    def load_results_dict(self, info):
        self.read_point_cloud(fname=info['config']['source'], config=info['config'])
        self.glWidget.tree = info['tree']
        self.glWidget.tree.assign_edge_colors()
        self.glWidget.make_pcd_gl_list()
        self.glWidget.initialize_skeleton()

        self.glWidget.update()
        self.repaint()

    def load_tree(self, tree):
        self.glWidget.show_skeleton = True
        self.glWidget.tree = tree
        self.glWidget.tree.assign_edge_colors()
        self.glWidget.make_pcd_gl_list()
        self.glWidget.initialize_skeleton()
        self.glWidget.update()
        self.repaint()

    def rotate_turntable(self, angle):
        if not self.glWidget.show_skeleton:
            self.show_skeleton()
        current_angle = self.turntable.value()
        new_val = (current_angle + angle) % 360
        self.glWidget.set_turntable_rotation(new_val)

    def resample(self, cover_radius, neighbor_radius):

        pc = self.glWidget.tree.resample(cover_radius, neighbor_radius)
        self.glWidget.reset_model(pc)
        self.glWidget.make_pcd_gl_list()
        self.glWidget.update()
        self.repaint()

    def reload_superpoint_graph(self, radius=0.10):
        graph = self.glWidget.tree.load_superpoint_graph(radius=radius)
        data = {
            'graph': graph,
            'points': self.glWidget.tree.points,
            'source_file': self.current_file,
            'radius': radius,
        }
        return data

    def highlight_edge(self, edge):

        pt_indexes = self.glWidget.tree.highlight_edge(edge)
        pts = self.glWidget.tree.points[list(pt_indexes)]
        graph = self.glWidget.tree.superpoint_graph
        start = graph.nodes[edge[0]]['point']
        end = graph.nodes[edge[1]]['point']

        for edge_c in graph.edges:
            if edge_c == edge:
                graph.edges[edge_c]['color'] = (0.1, 0.9, 0.9)
            else:
                graph.edges[edge_c].pop('color', None)

        self.glWidget.make_pcd_gl_list()
        self.glWidget.initialize_skeleton()
        self.glWidget.update()
        self.repaint()

        return pts, start, end


    def classify_and_highlight_edges(self, replay_counter=None):

        self.glWidget.tree.assign_edge_colors(replay_counter=replay_counter)
        self.glWidget.make_pcd_gl_list()
        self.glWidget.initialize_skeleton()
        self.glWidget.update()
        self.repaint()

    def magic(self):
        self.glWidget.magic()
        # self.glWidget.tree.classify_edges()
        # self.glWidget.tree.assign_edge_colors()
        # self.glWidget.make_pcd_gl_list()
        # self.glWidget.initialize_skeleton()
        # self.glWidget.update()
        # self.repaint()

    def read_pca_cylinders(self):
        fname = self.pca_cylinder_file
        try:
            with open(fname, "r") as fid:
                self.glWidget.cyl_cover.read(fid)
        except FileNotFoundError:
            print("File not found {0}".format(fname))

    def read_fit_cylinders(self):
        fname = self.fit_cylinder_file
        try:
            with open(fname, "r") as fid:
                self.glWidget.cyl_cover.read(fid)
        except FileNotFoundError:
            print("File not found {0}".format(fname))

    def redraw_self(self):
        self.glWidget.update()
        self.repaint()

    def update_pc(self, axis_filters_dict):
        self.glWidget.axis_filters = axis_filters_dict
        self.glWidget.refresh_filters()
        self.glWidget.make_pcd_gl_list()
        self.redraw_self()

    def create_new_tree(self, num_points):
        self.glWidget.create_new_tree(num_points)
        self.glWidget.tree.classify_edges()

    def save_config(self, config):
        config_path = os.path.join(self.config_dir, 'config.pickle')
        with open(config_path, 'wb') as fh:
            pickle.dump(config, fh)
        print('Saved config to: {}'.format(config_path))

    def skeletonize(self, params=None):
        if self.glWidget.tree is None:
            print('Please initialize a tree before you can run skeletonization!')
            return

        self.glWidget.tree.set_params(params)
        self.glWidget.tree.skeletonize()
        self.glWidget.tree.thinned_tree.find_side_branches()
        self.classify_and_highlight_edges()

        return self.get_current_graph()

    def save_active_skeleton(self, base_skeleton):
        to_save = {
            'base': base_skeleton,
            'repaired': self.get_current_graph()
        }

        all_files = os.listdir(self.config_dir)
        counter = 0
        file_template = 'tree_{}.pickle'
        while True:
            file_name = file_template.format(counter)
            if file_name in all_files:
                counter += 1
                continue

            file_path = os.path.join(self.config_dir, file_name)
            with open(file_path, 'wb') as fh:
                pickle.dump(to_save, fh)
            print('Saved trees to: {}'.format(file_path))

            break

    def get_current_graph(self):
        return self.glWidget.tree.thinned_tree.current_graph

    def update_repair_mode(self, enable, value):
        # Repairs - For running repairs on the tree

        self.glWidget.repair_mode = enable
        self.glWidget.repair_value = value

    def enable_polygon_mode(self, toggle):
        self.glWidget.polygon_filter_mode = toggle

    def apply_polygons(self, polygons):
        self.glWidget.polygon_filters = polygons
        self.glWidget.refresh_filters()
        self.glWidget.make_pcd_gl_list()
        self.glWidget.update()

    def replay_history(self):
        if self.glWidget.tree.tree_population is None:
            print('No history to replay!')
            return
        if self.replay_counter != 0:
            print('Timer in progress!')
            return
        self.replay_timer.start(200)

    def replay_history_update(self):
        try:
            self.classify_and_highlight_edges(self.replay_counter)
        except IndexError:
            self.replay_timer.stop()
            self.replay_counter = 0
            return


        self.replay_counter += 1

    def set_trunk_node(self, point=None):
        if point is None:
            self.glWidget.trunk_node = None
            self.glWidget.set_trunk_mode = True
        else:
            self.glWidget.trunk_node = point
        self.glWidget.make_pcd_gl_list()
        self.glWidget.update()


class PointCloudOrientationWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.widgets = {}
        for default_idx, direction in enumerate(['Right', 'Up', 'In']):
            sub_layout = QHBoxLayout()
            checkbox = QCheckBox()
            dropdown = QComboBox()
            for val, label in enumerate(['X', 'Y', 'Z']):
                dropdown.addItem(label, val)
            dropdown.setCurrentIndex(default_idx)
            self.widgets[direction] = {'sign': checkbox, 'axis': dropdown}
            for widget in [QLabel('{}: '.format(direction)), QLabel('Neg'), checkbox, dropdown]:
                sub_layout.addWidget(widget)

            dropdown.currentIndexChanged.connect(self.validate_selection)
            checkbox.clicked.connect(self.validate_selection)

            layout.addLayout(sub_layout)

        self.widgets['In']['sign'].setDisabled(True)
        self.widgets['In']['axis'].setDisabled(True)

    def get_axis_sign(self, direction):
        sign = -1 if self.widgets[direction]['sign'].isChecked() else 1
        axis = self.widgets[direction]['axis'].currentIndex()
        return axis, sign

    def validate_selection(self):
        right_ax, right_sign = self.get_axis_sign('Right')
        up_ax, up_sign = self.get_axis_sign('Up')
        if right_ax == up_ax:
            if right_ax == 0:
                up_ax = 1
            else:
                up_ax = 0
            self.widgets['Up']['axis'].setCurrentIndex(up_ax)

        right_vec = np.array([0, 0, 0])
        right_vec[right_ax] = right_sign
        up_vec = np.array([0, 0, 0])
        up_vec[up_ax] = up_sign

        in_vec = np.cross(right_vec, up_vec)
        in_ax = np.where(in_vec != 0)[0][0]
        in_sign = 1 if in_vec[in_ax] > 0 else -1
        self.widgets['In']['sign'].setChecked(in_sign < 0)
        self.widgets['In']['axis'].setCurrentIndex(in_ax)

    def get_desired_transform(self):
        # By default, the algorithm uses the convention of X=right, Y=down, Z=out, so we need to transform it
        default_tf = np.array([
            [1,0,0],
            [0,-1,0],
            [0,0,-1]])

        current_tf = np.zeros((3,3))
        for i, direction in enumerate(['Right', 'Up', 'In']):
            axis, sign = self.get_axis_sign(direction)
            current_tf[:,i][axis] = sign

        return default_tf @ current_tf


if __name__ == '__main__':
    app = QApplication([])

    settings = {
        'pcd_file': None, #'/home/main/data/point_clouds/bag_5/cloud_final.ply',
        'cover_file': None, # 'data/test_cyl_cover.txt',
    }

    if len(sys.argv) > 1:
        settings['pcd_file'] = sys.argv[1]

    if len(sys.argv) > 2:
        settings['cover_file'] = sys.argv[2]

    gui = PointCloudViewerGUI(**settings)

    gui.show()

    app.exec_()
