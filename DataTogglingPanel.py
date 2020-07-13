#!/usr/bin/env python3

# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton

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

    def update_label(self):
        text = '{:.3f}'.format(self.value())
        self.label.setText(text)

    def reset(self):
        self.slider.setValue(0)
        self.update_label()


class DataTogglingPanel(QWidget):
    def __init__(self, callback=None):
        super(DataTogglingPanel, self).__init__()

        self.callback = callback

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Connectedness Options
        layout.addWidget(QLabel('Connectedness Options'))
        connectedness_layout = QHBoxLayout()

        self.connected_menu = QComboBox()
        self.connected_menu.addItem("Connected")
        self.connected_menu.addItem("Disconnected")
        self.connected_threshold = FloatSlider(0, 1, 1000)
        connectedness_layout.addWidget(self.connected_menu)
        connectedness_layout.addWidget(self.connected_threshold)
        layout.addLayout(connectedness_layout)

        # Categorization-Type Toggle
        toggle_layout = QHBoxLayout()

        self.radio_multi = QRadioButton('Multi Category')
        self.radio_single = QRadioButton('Single Category')
        toggle_layout.addWidget(self.radio_multi)
        toggle_layout.addWidget(self.radio_single)
        self.radio_multi.toggled.connect(partial(self.toggle_category, True))
        self.radio_multi.toggled.connect(partial(self.toggle_category, True))


        layout.addLayout(toggle_layout)
        # Single category toggle

        single_layout = QHBoxLayout()
        self.single_category_menu = QComboBox()
        for cat in DataLabelingPanel.ALL_CLASSES[:-1]:
            self.single_category_menu.addItem(cat)
        self.single_category_threshold = FloatSlider(0, 1, 1000)
        single_layout.addWidget(self.single_category_menu)
        single_layout.addWidget(self.single_category_threshold)
        layout.addLayout(single_layout)


        # For running the skeleton thinning algorithm
        thinning_layout = QHBoxLayout()
        self.thinning_checkbox = QCheckBox()
        self.thinning_checkbox.setChecked(False)
        self.thinning_lower = LabelAndText('Low', '0.1')
        self.thinning_upper = LabelAndText('High', '0.5')
        self.thinning_checkbox.toggled.connect(self.thinning_toggle)
        self.thinning_toggle()
        for widget in [QLabel('Run Thinning'), self.thinning_checkbox, self.thinning_lower, self.thinning_upper]:
            thinning_layout.addWidget(widget)
        layout.addLayout(thinning_layout)

        show_button = QPushButton('Show')
        show_button.clicked.connect(self.commit)
        layout.addWidget(show_button)



        #
        # grid = QGridLayout()
        # layout.addWidget(QLabel('Options'))
        # layout.addLayout(grid)
        #
        # bottom = QHBoxLayout()
        # layout.addLayout(bottom)
        #
        # commit_button = QPushButton("Commit")
        # commit_button.clicked.connect(self.commit)
        # reset_button = QPushButton("Reset")
        # reset_button.clicked.connect(self.reset)
        # bottom.addWidget(commit_button)
        # bottom.addWidget(reset_button)
        #
        # self.checkboxes = {}
        # self.thresholds = {}
        # self.colors = {}
        # self.color_buttons = {}
        # for i, cat in enumerate(DataLabelingPanel.ALL_CLASSES):
        #     label = cat
        #     checkbox = QCheckBox()
        #     checkbox.setChecked(True)
        #     edit = FloatSlider(0.0, 1.0, 1000)
        #     self.checkboxes[cat] = checkbox
        #     self.thresholds[cat] = edit
        #
        #     grid.addWidget(QLabel(label), i, 0)
        #     grid.addWidget(checkbox, i, 1)
        #     grid.addWidget(edit, i, 2)
        #
        #     only_button = QPushButton('Only')
        #     only_button.clicked.connect(partial(self.toggle_only, cat))
        #
        #     except_button = QPushButton('Except')
        #     except_button.clicked.connect(partial(self.toggle_except, cat))
        #
        #     grid.addWidget(only_button, i, 3)
        #     grid.addWidget(except_button, i, 4)
        #
        #     color_button = QPushButton('Set Color')
        #     color_button.clicked.connect(partial(self.change_color, cat))
        #     self.color_buttons[cat] = color_button
        #     grid.addWidget(color_button, i, 5)
    def commit(self):

        # Format the data differently depending on if we want a single category value or a multi class
        is_multi = self.radio_multi.isChecked()
        if is_multi:
            raise NotImplementedError
        else:
            data = {
                'category': self.single_category_menu.currentIndex(),
                'threshold': self.single_category_threshold.value()
            }

        # For running thinning algo
        thinning_params = None
        if self.thinning_checkbox.isChecked():
            thinning_params = (self.thinning_lower.value(), self.thinning_upper.value())

        data = {
            'show_connected': self.connected_menu.currentIndex() == 0,
            'thinning': thinning_params,
            'connection_threshold': self.connected_threshold.value(),
            'multi_classify': is_multi,
            'data': data
        }

        self.callback(data)

    def thinning_toggle(self):
        is_checked = self.thinning_checkbox.isChecked()
        self.thinning_lower.textbox.setDisabled(not is_checked)
        self.thinning_upper.textbox.setDisabled(not is_checked)

    def toggle_category(self, is_multi):
        pass
    #
    # def toggle_except(self, undesired):
    #
    #
    #     for cat in DataLabelingPanel.ALL_CLASSES:
    #         self.checkboxes[cat].setChecked(cat != undesired)
    #     self.commit()
    #
    # def toggle_only(self, desired):
    #     for cat in DataLabelingPanel.ALL_CLASSES:
    #         self.checkboxes[cat].setChecked(cat == desired)
    #     self.commit()
    #
    # def reset(self):
    #     for cat in DataLabelingPanel.ALL_CLASSES:
    #         self.checkboxes[cat].setChecked(True)
    #         self.thresholds[cat].reset()
    #         self.color_buttons[cat].setStyleSheet("background-color: rgb(255,255,255)")
    #     self.colors = {}
    #     self.commit()
    #
    # def change_color(self, cat):
    #     color = QColorDialog.getColor()
    #     r, g, b, _ = color.getRgb()
    #     self.color_buttons[cat].setStyleSheet("background-color: rgb({},{},{})".format(r, g, b))
    #
    #     print(color.getRgbF())
    #     self.colors[cat] = color
    #
    # def commit(self):
    #
    #     state = {cat: {'threshold': self.thresholds[cat].value(),
    #                    'active': self.checkboxes[cat].isChecked(),
    #                    'color': self.colors.get(cat, QColor.fromRgb(255, 255, 255)).getRgbF()}
    #              for cat in DataLabelingPanel.ALL_CLASSES}
    #     if self.callback is None:
    #         print(state)
    #     else:
    #         self.callback(state)
