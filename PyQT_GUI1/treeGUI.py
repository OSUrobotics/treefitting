import sys
from PyQt6.QtCore import *
from PyQt6.QtWidgets import * #QApplication, QWidget, QVBoxLayout, QLabel, QDesktopWidget
from PyQt6.QtGui import *
from PyQt6.QtCharts import *
import treeGUI_features as features
from treeGUI_windows import windows

class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("tree analyzer GUI")
        self.resize(1280, 800)

        # Attributes to store images and current index
        self.images = []  # List of file paths for the images
        self.image_index = 0  # Index of the currently displayed image

        # Get the layout and widgets from the external function, passing self as parent
        layout, self.main_image_label, self.left_button, self.right_button, self.file_button = windows(self)
        self.setLayout(layout)

        # Connect signals to features module functions
        self.left_button.clicked.connect(lambda: features.on_left_button_clicked(self))
        self.file_button.clicked.connect(lambda: features.on_file_selected(self))
        self.right_button.clicked.connect(lambda: features.on_right_button_clicked(self))

        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # Button handler functions (optional: if you want them in GUI)
    def on_left_button_clicked(self):
        features.on_left_button_clicked(self)

    def on_right_button_clicked(self):
        features.on_right_button_clicked(self)

    def on_file_selected(self):
        features.on_file_selected(self)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec())
