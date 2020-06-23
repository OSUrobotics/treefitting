from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton
import PyQt5.QtCore as QtCore
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog
from PyQt5.QtGui import QPainter, QPixmap, QPen
import sys
import os
from MachineLearningPanel import CanvasWithOverlay

class TreeGenerator(QMainWindow):
    def __init__(self, root=None):
        QMainWindow.__init__(self)
        self.setWindowTitle('Tree generator and labeller')

        self.root = root
        if self.root is None:
            self.root = '/home/main/data/fake_2d_trees/templates'
        self.counter = len(os.listdir(self.root))


        # The layout of the interface
        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout()
        widget.setLayout(main_layout)

        # Add drawing interface
        self.canvas = CanvasWithOverlay(400, hold_to_draw=True,
                                        pen_size=(5, 2, 10, 1))
        main_layout.addWidget(self.canvas)

    def save(self):
        self.canvas.overlay_pixmap.save(os.path.join(self.root, '{}.png'.format(self.counter)))
        self.canvas.reset_overlay()
        self.counter += 1
        self.canvas.disable_erase_mode()

    def keyPressEvent(self, event):
        pressed = event.key()
        if pressed == QtCore.Qt.Key_E:
            if not self.canvas.drawing_mode:
                self.canvas.reset_overlay()
            else:
                self.canvas.enable_erase_mode()

        elif pressed == QtCore.Qt.Key_D:
            self.canvas.disable_erase_mode()

        elif pressed == QtCore.Qt.Key_A:
            self.refresh(commit=True)

        elif pressed == QtCore.Qt.Key_S:
            self.save()




from MachineLearningPanel import CanvasWithOverlay

if __name__ == '__main__':
    app = QApplication([])

    gui = TreeGenerator()

    gui.show()

    app.exec_()
