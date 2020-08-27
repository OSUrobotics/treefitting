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
        self.radio_multi.setChecked(True)
        self.radio_multi.toggled.connect(partial(self.toggle_category, True))
        self.radio_single.toggled.connect(partial(self.toggle_category, False))


        layout.addLayout(toggle_layout)
        # Single category toggle

        self.single_widget = QFrame()
        single_layout = QHBoxLayout()
        self.single_category_menu = QComboBox()
        for cat in DataLabelingPanel.ALL_CLASSES[:-1]:
            self.single_category_menu.addItem(cat)
        self.single_category_threshold = FloatSlider(0, 1, 1000)
        single_layout.addWidget(self.single_category_menu)
        single_layout.addWidget(self.single_category_threshold)
        self.single_widget.setLayout(single_layout)
        layout.addWidget(self.single_widget)

        # For multi category toggle - consider only the maximum values
        self.multi_widget = QFrame()
        multi_layout = QGridLayout()
        self.multi_vals = {}
        for i, cat in enumerate(DataLabelingPanel.ALL_CLASSES[:-1]):

            checkbox = QCheckBox()
            checkbox.setChecked(True)

            color_button = QPushButton('Set Color')
            color_button.clicked.connect(partial(self.change_color, i, None))

            tol_slider = FloatSlider(0, 1, 1000)

            multi_layout.addWidget(QLabel(cat), i, 0)
            multi_layout.addWidget(checkbox, i, 1)
            multi_layout.addWidget(tol_slider, i, 2)
            multi_layout.addWidget(color_button, i, 3)
            self.multi_vals[i] = {
                'checkbox': checkbox,
                'color_button': color_button,
                'color': False,
                'threshold': tol_slider
            }
        self.multi_widget.setLayout(multi_layout)
        layout.addWidget(self.multi_widget)

        # For iteratively running the skeleton correction algorithm
        correction_layout = QGridLayout()
        self.correction_checkbox = QCheckBox()
        self.correction_checkbox.setChecked(True)
        self.foundation_checkbox = QCheckBox()
        self.foundation_checkbox.setChecked(True)

        to_add = [('Run Correction', self.correction_checkbox),
                                   ('Show Foundation', self.foundation_checkbox)]

        for i, (label_text, widget) in enumerate(to_add):
            correction_layout.addWidget(QLabel(label_text), i, 0)
            correction_layout.addWidget(widget, i, 1)
        layout.addLayout(correction_layout)

        show_button = QPushButton('Show')
        show_button.clicked.connect(self.commit)
        layout.addWidget(show_button)

        # Setup
        self.toggle_category(True)
        default_colors = [
            (0.6, 0.55, 0.4),
            (0.95, 0.4, 0.6),
            (0.4, 0.6, 0.9),
            (0.9, 0.9, 0.0),
            (0.4, 0.7, 0.7),

        ]
        for i, color in enumerate(default_colors):
            self.change_color(i, color)


    def commit(self):

        # Format the data differently depending on if we want a single category value or a multi class
        is_multi = self.radio_multi.isChecked()
        if is_multi:
            data = {}
            for i, vals in self.multi_vals.items():
                color = vals['color'] if vals['checkbox'].isChecked() else False
                threshold = vals['threshold'].value()
                data[i] = {'color': color, 'threshold': threshold}
        else:
            data = {
                'category': self.single_category_menu.currentIndex(),
                'threshold': self.single_category_threshold.value()
            }

        data = {
            'show_connected': self.connected_menu.currentIndex() == 0,
            'correction': self.correction_checkbox.isChecked(),
            'show_foundation': self.foundation_checkbox.isChecked(),
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
        if is_multi:
            self.single_widget.hide()
            self.multi_widget.show()
        else:
            self.single_widget.show()
            self.multi_widget.hide()

    def change_color(self, cat, color=None):
        if color is None:
            color = QColorDialog.getColor()
            r, g, b, _ = color.getRgb()
            rf, gf, bf, _ = color.getRgbF()
        else:
            rf, gf, bf = color
            r = int(color[0] * 256)
            g = int(color[1] * 256)
            b = int(color[2] * 256)


        self.multi_vals[cat]['color_button'].setStyleSheet("background-color: rgb({},{},{})".format(r, g, b))
        self.multi_vals[cat]['color'] = (rf, gf, bf)
