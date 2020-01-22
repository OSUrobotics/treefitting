#!/usr/bin/env python3

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QSlider, QWidget, QLabel, QPushButton


# A helper class that implements a slider with given start and end float value; displays values
class SliderFloatDisplay(QWidget):
    gui = None

    def __init__(self, name, low, high, initial_value, ticks=100):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param low: Minimum value slider returns
        :param high: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        :param ticks: Resolution of slider - all sliders are integer/fixed number of ticks
        """
        # Save input values
        self.name = name
        self.low = low
        self.range = high - low
        self.ticks = ticks
        self.b_recalc_ids = False

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(ticks)
        # call back - call change_value when slider changed
        self.slider.valueChanged.connect(self.change_value)

        self.display = QLabel()
        self.set_value(initial_value)
        self.change_value()

        layout.addWidget(self.display)
        layout.addWidget(self.slider)

    # Use this to get the value between low/high
    def value(self):
        """Return the current value of the slider"""
        return (self.slider.value() / self.ticks) * self.range + self.low

    # Called when the value changes - resets text
    def change_value(self):
        self.display.setText('{0}: {1:.3f}'.format(self.name, self.value()))
        if self.b_recalc_ids:
            try:
                SliderFloatDisplay.gui.glWidget.recalc_gl_ids()
            except (NameError, AttributeError):
                pass
        try:
            SliderFloatDisplay.gui.repaint()
            SliderFloatDisplay.gui.glWidget.update()
        except (NameError, AttributeError):
            pass

    # Use this to change the value (does clamping)
    def set_value(self, value_f):
        value_tick = self.ticks * (value_f - self.low) / self.range
        value_tick = min(max(0, value_tick), self.ticks)
        self.slider.setValue(int(value_tick))
        self.change_value()

    def set_range(self, min_v, max_v):
        self.slider.setMinimum(min_v)
        self.slider.setMaximum(max_v)
        self.set_value(0.5 * (min_v + max_v))


class SliderIntDisplay(QWidget):
    gui = None

    def __init__(self, name, low=0, high=10, initial_value=0):
        """
        Give me a name, the low and high values, and an initial value to set
        :param name: Name displayed on slider
        :param low: Minimum value slider returns
        :param high: Maximum value slider returns
        :param initial_value: Should be a value between low and high
        """
        # Save input values
        self.name = name

        # I'm a widget with a text value next to a slider
        QWidget.__init__(self)
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(low)
        self.slider.setMaximum(high)
        # call back - call change_value when slider changed
        self.slider.valueChanged.connect(self.change_value)

        self.display = QLabel()
        if low <= initial_value < high:
            self.set_value(initial_value)
        self.change_value()

        self.left_button = QPushButton("<")
        self.left_button.clicked.connect(self.decrement)
        self.right_button = QPushButton(">")
        self.right_button.clicked.connect(self.increment)

        layout.addWidget(self.left_button)
        layout.addWidget(self.display)
        layout.addWidget(self.slider)
        layout.addWidget(self.right_button)

    def value(self):
        return self.slider.value()

    def setMaximum(self, val):
        self.slider.setMaximum(val)
        if self.slider.value() >= val:
            self.slider.setValue(0)

    # Called when the value changes - resets text
    def change_value(self):
        self.display.setText('{0}: {1}'.format(self.name, self.slider.value()))
        try:
            SliderIntDisplay.gui.repaint()
            SliderIntDisplay.gui.glWidget.update()
        except (NameError, AttributeError):
            pass

    # Use this to change the value
    def set_value(self, value):
        self.slider.setValue(value)
        self.change_value()

    def decrement(self):
        if self.slider.value() > 0:
            self.slider.setValue(self.slider.value()-1)
            self.change_value()

    def increment(self):
        if self.slider.value() < self.slider.maximum()-1:
            self.slider.setValue(self.slider.value()+1)
            self.change_value()
