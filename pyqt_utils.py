# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFrame

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QSlider, QRadioButton, QComboBox
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt

class LabelAndText(QWidget):
    def __init__(self, label, starting_text='', force_int=False):
        super(LabelAndText, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.label = QLabel(label)
        self.textbox = QLineEdit(starting_text)
        self.force_int = force_int

        layout.addWidget(self.label)
        layout.addWidget(self.textbox)

    def __getattr__(self, attrib):
        # For easy binding to the textbox values
        return self.textbox.__getattr__(attrib)

    def text(self):
        return self.textbox.text()

    def value(self):
        if self.force_int:
            return int(self.text())
        return float(self.text())
