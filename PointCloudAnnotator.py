from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFrame

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QSlider, QRadioButton, \
    QComboBox, QFrame
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QRect

import os
from pyqt_utils import LabelAndText, CanvasWithOverlay, LabelAndWidget
from tree_model import TreeModel
import pickle
from collections import defaultdict
from functools import partial

class Paginator(QWidget):
    def __init__(self):
        super(Paginator, self).__init__()

        self.combo_box = QComboBox()
        self.bound_actions = []
        self.disable_update = False

        self.left_button = QPushButton('<')
        self.right_button = QPushButton('>')


        layout = QHBoxLayout()
        self.setLayout(layout)

        widgets = [self.left_button, self.combo_box, self.right_button]
        for widget in widgets:
            layout.addWidget(widget)

        self.combo_box.currentIndexChanged.connect(self.set_index)
        self.left_button.clicked.connect(partial(self.update_index, -1))
        self.right_button.clicked.connect(partial(self.update_index, 1))

    def update_index(self, delta=0):

        old_i = self.combo_box.currentIndex()
        new_i = old_i + delta
        if new_i < 0:
            new_i = 0
        ct = self.combo_box.count()
        if new_i >= ct:
            new_i = ct - 1
        if delta and new_i == old_i:
            return
        self.set_index(new_i)

    def set_index(self, i=None):
        if self.disable_update:
            return

        if i is None:
            i = self.combo_box.currentIndex()

        self.left_button.setEnabled(True)
        self.right_button.setEnabled(True)
        if i <= 0:
            self.left_button.setEnabled(False)
        if i >= self.combo_box.count() - 1:
            self.right_button.setEnabled(False)

        self.disable_update = True
        self.combo_box.setCurrentIndex(i)
        self.disable_update = False

        for action in self.bound_actions:
            action()

    def bind_action(self, action):
        self.bound_actions.append(action)

    def set_items(self, items):
        self.disable_update = True
        try:
            self.combo_box.clear()

            for item in items:
                self.combo_box.addItem(str(item), item)
        finally:
            self.disable_update = False
        self.set_index(0)

    def value(self):
        return self.combo_box.currentData()

class PointCloudAnnotator(QWidget):
    def __init__(self, gl_widget, callbacks, max_height=600):
        super(PointCloudAnnotator, self).__init__()


        layout = QVBoxLayout()
        self.setLayout(layout)

        self.gl_widget = gl_widget
        self.callbacks = callbacks
        self.skels_to_load = {}
        self.subid_data = {}
        self.max_height = max_height


        settings_widget = QWidget()
        settings_layout = QHBoxLayout()
        settings_widget.setLayout(settings_layout)

        self.pc_directory = LabelAndText('Point Cloud Directory')
        self.results_directory = LabelAndText('Results Directory', '/home/main/data/skeletonization_results')
        self.results_prefix = LabelAndText('Results Prefix', 'skeleton')
        refresh_button = QPushButton('Refresh')

        for widget in [self.pc_directory, self.results_directory, self.results_prefix, refresh_button]:
            settings_layout.addWidget(widget)


        selection_widget = QWidget()
        selection_layout = QHBoxLayout()
        selection_widget.setLayout(selection_layout)

        self.skeleton_menu = QComboBox()
        self.paginator = Paginator()
        selection_layout.addWidget(LabelAndWidget('Select Skeleton', self.skeleton_menu))
        selection_layout.addWidget(LabelAndWidget('Select Child ID', self.paginator))


        canvas_widget = QWidget()
        canvas_layout = QHBoxLayout()
        canvas_widget.setLayout(canvas_layout)
        self.canvas = CanvasWithOverlay((1000, 750), hold_to_draw=True, pen_size=(5,1,11,2))
        canvas_layout.addWidget(self.canvas)
        radio_container = QVBoxLayout()

        labels = ['Trunk', 'Support', 'Leader', 'Side Branch', 'Exclude/Other']
        colors = [(0.7, 0.65, 0.5), (1.0, 0.5, 0.7), (0.5, 0.7, 1.0), (0.25, 0.75, 0.4),
                  (0.5, 0.8, 0.8)]
        def callback(c):
            self.canvas.set_pen_color(c)

        for label, color in zip(labels, colors):
            radio = QRadioButton(label)
            radio_container.addWidget(radio)

            radio.toggled.connect(partial(callback, color))
        update_button = QPushButton('Reset')
        radio_container.addWidget(update_button)

        canvas_layout.addLayout(radio_container)


        controls_layout = QVBoxLayout()
        left_45_button = QPushButton('-45')
        right_45_button = QPushButton('45')

        controls_layout.addWidget(left_45_button)
        controls_layout.addWidget(right_45_button)

        left_45_button.clicked.connect(partial(self.rotate_turntable, -45))
        right_45_button.clicked.connect(partial(self.rotate_turntable, 45))

        canvas_layout.addLayout(controls_layout)

        save_rez_widget = QWidget()
        save_rez_layout = QHBoxLayout()
        save_rez_widget.setLayout(save_rez_layout)
        self.comment_box = QLineEdit()
        save_button = QPushButton('Save')
        save_rez_layout.addWidget(self.comment_box)
        save_rez_layout.addWidget(save_button)

        widgets = [settings_widget, selection_widget, canvas_widget, save_rez_widget]
        for widget in widgets:
            layout.addWidget(widget)

        update_button.clicked.connect(self.update_canvas)
        refresh_button.clicked.connect(self.update_results_files)
        self.skeleton_menu.currentIndexChanged.connect(self.update_active_skeleton)
        self.paginator.bind_action(self.update_sub_skeleton)
        save_button.clicked.connect(self.save)

    def update_canvas(self):
        rect = QRect(0, 0, self.gl_widget.width(), self.gl_widget.height())
        pixmap = self.gl_widget.grab(rect)
        if pixmap.height() > self.max_height:
            pixmap = pixmap.scaled(pixmap.width(), self.max_height, Qt.KeepAspectRatio)
        self.canvas.update_base_pixmap(pixmap)

    def update_results_files(self):
        new_dict = defaultdict(list)
        all_files = [f.replace('.pickle', '') for f in os.listdir(self.results_directory.text()) if f.startswith(self.results_prefix.text())]
        for file_base in all_files:
            comps = file_base.split('_')
            skel_id = int(comps[-2])
            skel_subid = int(comps[-1])
            new_dict[skel_id].append(skel_subid)
        self.skels_to_load = new_dict
        self.update_skeleton_menu()

    def update_skeleton_menu(self):
        self.skeleton_menu.clear()

        for k in sorted(self.skels_to_load):
            self.skeleton_menu.addItem(str(k), k)

    def update_active_skeleton(self):
        skel_id = self.skeleton_menu.currentData()
        sub_ids = sorted(self.skels_to_load[skel_id])
        self.paginator.set_items(sub_ids)
        self.subid_data = {}
        for sub_id in sub_ids:
            file_name = '{}_{}_{}.pickle'.format(self.results_prefix.text(), skel_id, sub_id)
            file_path = os.path.join(self.results_directory.text(), file_name)
            with open(file_path, 'rb') as fh:
                rez = pickle.load(fh)
            self.subid_data[sub_id] = rez

        source = self.subid_data[sub_ids[0]]['config']['source']
        pc_dir = self.pc_directory.text()
        if pc_dir:
            base_comps = os.path.split(source)[-2:]
            source = os.path.join(pc_dir, *base_comps)


        self.callbacks['read_point_cloud'](fname=source)
        self.update_sub_skeleton()

    def update_sub_skeleton(self):
        if not self.subid_data:
            return

        sub_id = self.paginator.value()
        tree = self.subid_data[sub_id]['tree']
        self.callbacks['load_tree'](tree)
        self.update_canvas()

    def save(self):
        comment = self.comment_box.text().strip()
        if not comment:
            comment = '(No comment)'

        current_img = self.canvas.get_combined_pixmap()

        skel_id = self.skeleton_menu.currentData()
        sub_id = self.paginator.value()
        prefix = self.results_prefix.text()
        results_dir = self.results_directory.text()

        comment_dir = os.path.join(results_dir, 'comments')
        if not os.path.exists(comment_dir):
            os.mkdir(comment_dir)
        file_base = '{}_{}_{}'.format(prefix, skel_id, sub_id)
        i = 1
        while True:
            img_file = file_base + '_{}.png'.format(i)
            img_path = os.path.join(comment_dir, img_file)
            if os.path.exists(img_path):
                i += 1
                continue
            current_img.save(img_path)
            break

        with open(os.path.join(comment_dir, file_base + '_comments.txt'), 'a+') as fh:
            fh.write('{}: {}{}'.format(i, comment, os.linesep))
        print('Saved feedback!')

        self.comment_box.setText('')
        self.canvas.reset_overlay()

    def rotate_turntable(self, angle):
        self.callbacks['rotate_turntable'](angle)
        self.update_canvas()
