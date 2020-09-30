#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFrame

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QSlider, QRadioButton, QComboBox
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

import os

from functools import partial
from DataLabelingPanel import DataLabelingPanel, LabelAndText


ROOT = os.path.dirname(os.path.realpath(__file__))
CONFIG = os.path.join(ROOT, 'config')
if not os.path.exists(CONFIG):
    os.mkdir(CONFIG)

class FloatSlider(QWidget):
    def __init__(self, start, end, ticks):
        super(FloatSlider, self).__init__()
        self.start = start
        self.end = end
        self.ticks = ticks

        layout = QHBoxLayout()
        self.setLayout(layout)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(ticks)
        slider.valueChanged.connect(self.update_label)
        self.slider = slider
        layout.addWidget(self.slider)

        self.label = QLabel()
        layout.addWidget(self.label)

        self.update_label()

    def value(self):
        return self.start + self.slider.value() / self.ticks * (self.end - self.start)

    def set_value(self, val):
        tick = int((val - self.start) / (self.end - self.start) * self.ticks)
        if tick < 0:
            tick = 0
        if tick > self.ticks:
            tick = self.ticks
        self.slider.setValue(tick)
        self.update_label()

    def update_label(self):
        text = '{:.3f}'.format(self.value())
        self.label.setText(text)

    def reset(self):
        self.slider.setValue(0)
        self.update_label()

    def change_bounds(self, low, high):
        self.start = low
        self.end = high
        self.reset()

class DoubleSlider(QWidget):
    def __init__(self, name, low, high, ticks):

        super(DoubleSlider, self).__init__()

        layout = QVBoxLayout()
        self.setLayout(layout)

        main_widget = QGroupBox(name)
        main_widget_layout = QVBoxLayout()
        main_widget.setLayout(main_widget_layout)
        layout.addWidget(main_widget)

        self.low_slider = FloatSlider(low, high, ticks)
        self.high_slider = FloatSlider(low, high, ticks)
        self.low_slider.set_value(low)
        self.high_slider.set_value(high)

        main_widget_layout.addWidget(self.low_slider)
        main_widget_layout.addWidget(self.high_slider)

        self.low_slider.slider.valueChanged.connect(lambda: self.adjust_sliders(True))
        self.high_slider.slider.valueChanged.connect(lambda: self.adjust_sliders(False))

    def adjust_sliders(self, low_val=True):
        low = self.low_slider.value()
        high = self.high_slider.value()
        if low > high:
            if low_val:
                self.high_slider.set_value(low)
            else:
                self.low_slider.set_value(high)

    def value(self):
        return self.low_slider.value(), self.high_slider.value()

    def set_value(self, low, high):
        if low > high:
            raise ValueError("Your first value should be lower than your second one")
        self.low_slider.set_value(low)
        self.high_slider.set_value(high)

    def change_bounds(self, low, high):
        self.low_slider.change_bounds(low, high)
        self.high_slider.change_bounds(low, high)
        self.high_slider.set_value(high)

    def connect_to_value_changed(self, event):
        self.low_slider.slider.valueChanged.connect(event)
        self.high_slider.slider.valueChanged.connect(event)

class PointCloudManagementPanel(QWidget):
    def __init__(self, callbacks=None):
        super(PointCloudManagementPanel, self).__init__()

        self.callbacks = callbacks
        self.saved_skeleton = None
        self.polygons = []

        layout = QHBoxLayout()
        self.setLayout(layout)

        # For point cloud filtering
        pc_zone_widget = QGroupBox("Point Cloud")
        pc_layout = QVBoxLayout()
        pc_zone_widget.setLayout(pc_layout)

        self.x_sliders = DoubleSlider('x', 0, 1, 1000)
        self.y_sliders = DoubleSlider('y', 0, 1, 1000)
        self.z_sliders = DoubleSlider('z', 0, 1, 1000)
        self.trusted = QCheckBox('Trusted')
        self.trusted.setChecked(False)
        self.enable_polygon_box = QCheckBox('Polygon Filter')
        self.enable_polygon_box.setChecked(False)
        self.polygon_label = QLabel('')
        delete_polygons = QPushButton('Delete all polygons')
        self.update_polygon_label()
        commit_button = QPushButton('Commit')

        self.x_sliders.connect_to_value_changed(self.update_pc)
        self.y_sliders.connect_to_value_changed(self.update_pc)
        self.z_sliders.connect_to_value_changed(self.update_pc)
        self.enable_polygon_box.clicked.connect(self.enable_polygon_mode)
        delete_polygons.clicked.connect(self.delete_polygons)

        commit_button.clicked.connect(self.save_config)

        widgets = [self.x_sliders, self.y_sliders, self.z_sliders, self.trusted,
                   self.enable_polygon_box, self.polygon_label, delete_polygons, commit_button]
        for widget in widgets:
            pc_layout.addWidget(widget)
        pc_layout.addStretch()
        layout.addWidget(pc_zone_widget)

        # FOr tree management
        tree_zone_widget = QGroupBox("Tree")
        tree_layout = QVBoxLayout()
        tree_zone_widget.setLayout(tree_layout)
        layout.addWidget(tree_zone_widget)

        self.tree_points = LabelAndText('Tree Pts', '50000')
        make_tree_button = QPushButton('Create Tree')
        make_tree_button.clicked.connect(self.create_tree)
        tree_layout.addWidget(self.tree_points)
        tree_layout.addWidget(make_tree_button)
        tree_layout.addStretch()

        # For handling skeletonization
        skel_zone_widget = QGroupBox("Skeletonization")
        skel_layout = QVBoxLayout()
        skel_zone_widget.setLayout(skel_layout)
        layout.addWidget(skel_zone_widget)

        self.skel_status = QLabel('Status: No tree initialized')
        generate_skeleton_button = QPushButton("Generate skeleton")
        self.enable_repair = QCheckBox('Enable repair mode')
        self.repair_value_menu = QComboBox()
        for val, label in [(0, 'Trunk'), (1, 'Support'), (2, 'Leader')]:
            self.repair_value_menu.addItem(label, val)
        self.repair_value_menu.setDisabled(True)
        self.save_skeleton_button = QPushButton('Save repaired tree')

        widgets = [self.skel_status, generate_skeleton_button, self.enable_repair, self.repair_value_menu,
                   self.save_skeleton_button]
        for widget in widgets:
            skel_layout.addWidget(widget)
        skel_layout.addStretch()

        generate_skeleton_button.clicked.connect(self.skeletonize)
        self.save_skeleton_button.clicked.connect(self.save_skeleton)
        self.enable_repair.clicked.connect(self.update_repair_mode)
        self.repair_value_menu.currentIndexChanged.connect(self.update_repair_mode)

        self.fresh_initialize()


    def fresh_initialize(self):
        self.saved_skeleton = None
        self.skel_status.setText('Status: No tree initialized')
        self.enable_repair.setChecked(False)
        self.enable_repair.setDisabled(True)
        self.repair_value_menu.setDisabled(True)
        self.save_skeleton_button.setDisabled(True)


    def set_bounds_from_pc(self, pc):
        mins = pc.min(axis=0)
        maxs = pc.max(axis=0)

        self.x_sliders.change_bounds(mins[0], maxs[0])
        self.y_sliders.change_bounds(mins[1], maxs[1])
        self.z_sliders.change_bounds(mins[2], maxs[2])

    @property
    def values_dict(self):
        return {
            'x': self.x_sliders.value(),
            'y': self.y_sliders.value(),
            'z': self.z_sliders.value()
        }

    @property
    def config_dict(self):
        return {
            'trusted': self.trusted.isChecked(),
            'bounds': self.values_dict,
            'polygons': self.polygons,
        }

    def update_pc(self):
        self.callbacks['update_pc'](self.values_dict)

    def save_config(self):
        self.callbacks['save_config'](self.config_dict)

    def load_config(self, config):
        self.load_bounds_dict(config.get('bounds'))
        self.trusted.setChecked(config.get('trusted', False))
        self.polygons = config.get('polygons', [])
        self.apply_polygons()

    def load_bounds_dict(self, bounds_dict):
        if not bounds_dict:
            return
        self.x_sliders.set_value(*bounds_dict['x'])
        self.y_sliders.set_value(*bounds_dict['y'])
        self.z_sliders.set_value(*bounds_dict['z'])

    def create_tree(self):
        self.callbacks['create_new_tree'](int(self.tree_points.value()))
        self.skel_status.setText('Status: Tree loaded, not skeletonized')

    def skeletonize(self):
        skeleton = self.callbacks['skeletonize']()
        self.saved_skeleton = skeleton.copy()
        self.skel_status.setText('Status: Skeletonized')
        self.enable_repair.setDisabled(False)
        self.save_skeleton_button.setDisabled(False)

    def save_skeleton(self):

        assert self.saved_skeleton is not None
        self.callbacks['save_skeleton'](self.saved_skeleton)

    def update_repair_mode(self):
        repair_mode = self.enable_repair.isChecked()
        if repair_mode:
            repair_value = self.repair_value_menu.currentData()
            self.repair_value_menu.setEnabled(True)
        else:
            repair_value = None
            self.repair_value_menu.setEnabled(False)

        self.callbacks['update_repair_mode'](repair_mode, repair_value)


    def update_polygon_label(self):
        n = len(self.polygons)
        self.polygon_label.setText('{} active polygon filter(s)'.format(n))

    def enable_polygon_mode(self):
        toggle = self.enable_polygon_box.isChecked()
        self.callbacks['enable_polygon_mode'](toggle)

    def apply_polygons(self):
        self.callbacks['apply_polygons'](self.polygons)
        self.update_polygon_label()

    def delete_polygons(self):
        self.polygons = []
        self.apply_polygons()

    def update_polygons(self, polygons):
        self.polygons = polygons
        self.update_polygon_label()
