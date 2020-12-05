# Get OpenGL
from PyQt5.QtWidgets import QMainWindow, QCheckBox, QGroupBox, QGridLayout, QVBoxLayout, QHBoxLayout, QPushButton, QFrame

from PyQt5.QtWidgets import QApplication, QHBoxLayout, QWidget, QLabel, QLineEdit, QColorDialog, QSlider, QRadioButton, QComboBox
from PyQt5.QtGui import QColor
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap, QPen


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

class LabelAndWidget(QWidget):
    def __init__(self, text, widget, vertical=True):
        super(LabelAndWidget, self).__init__()
        if vertical:
            layout = QVBoxLayout()
        else:
            layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(QLabel(text))
        layout.addWidget(widget)


# https://stackoverflow.com/questions/53420826/overlay-two-pixmaps-with-alpha-value-using-qpainter
def overlay_pixmap(base, overlay):
    # Assumes both have same size
    rez = QPixmap(base.size())
    rez.fill(QtCore.Qt.transparent)
    painter = QPainter(rez)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.drawPixmap(QtCore.QPoint(), base)
    painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
    painter.drawPixmap(rez.rect(), overlay, overlay.rect())
    painter.end()

    return rez

class CanvasWithOverlay(QWidget):
    def __init__(self, image_size=512, hold_to_draw=False, pen_size=None, show_diagnostic_labels=True):
        """
        Initializes a canvas which can display a base image underneath it.
        :param image_size: Either an integer or a 2-tuple of integers
        :param hold_to_draw: If True, drawing will be done by holding down the mouse button.
                             Otherwise it draws every time you click
        :param pen_size: If not specified, either a pe
        """
        super(CanvasWithOverlay, self).__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.hold_to_draw = hold_to_draw
        if isinstance(image_size, int):
            self.x = self.y = image_size
        else:
            self.x, self.y = image_size
        self.pen_size = 1
        self.pen_range = (1, 1)
        self.pen_ticks = 0
        self.pen_color = QtCore.Qt.green
        if pen_size is not None:
            if isinstance(pen_size, int):
                self.pen_size = pen_size
            else:   # 4-tuple
                self.pen_size, low, high, self.pen_ticks = pen_size
                self.pen_range = (low, high)

        # Image label setup
        self.image_label = QLabel()
        self.image_label.setStyleSheet('padding:15px')
        layout.addWidget(self.image_label)
        self.padding = 15
        self.base_pixmap = QPixmap(self.x, self.y)
        self.base_pixmap.fill(QtCore.Qt.black)
        self.overlay_pixmap = QPixmap(self.x, self.y)
        self.overlay_pixmap.fill(QtCore.Qt.transparent)
        self.image_label.setPixmap(self.base_pixmap)
        self.image_label.mousePressEvent = self.handle_mouse_press
        self.image_label.mouseMoveEvent = self.handle_mouse_move
        self.image_label.mouseReleaseEvent = self.handle_mouse_move

        # Diagnostic label setup
        diagnostic_layout = QHBoxLayout()
        self.pen_label = QLabel('Pen size: {}px'.format(self.pen_size))
        self.mode_label = QLabel('Drawing mode')
        self.switch_button = QPushButton('Eraser')
        self.switch_button.clicked.connect(self.toggle_erase)
        diagnostic_layout.addWidget(self.pen_label)
        diagnostic_layout.addWidget(self.mode_label)
        diagnostic_layout.addWidget(self.switch_button)
        if show_diagnostic_labels:
            layout.addLayout(diagnostic_layout)

        # State tracking
        self.drawing_mode = True
        self.last_x = None
        self.last_y = None

    def set_pen_color(self, val):
        assert len(val) == 3
        if any(map(lambda v: isinstance(v, float), val)):
            val = (int(v * 255) for v in val)
        self.pen_color = QColor(*val)

    def draw(self, x, y):
        x -= self.padding
        y -= self.padding
        if self.last_x is None:
            self.last_x = x
            self.last_y = y
            return

        painter = QPainter(self.overlay_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = painter.pen()
        pen.setWidth(self.pen_size)
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setColor(self.pen_color)
        painter.setPen(pen)
        if self.drawing_mode:
            painter.drawLine(self.last_x, self.last_y, x, y)
        else:
            # Erase via rectangles
            rect = QtCore.QRect(QtCore.QPoint(), self.pen_size * QtCore.QSize())
            rect.moveCenter(QtCore.QPoint(x, y))
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.eraseRect(rect)

        painter.end()

        self.last_x, self.last_y = x, y

        combined = overlay_pixmap(self.base_pixmap, self.overlay_pixmap)
        self.image_label.setPixmap(combined)
        self.update()

    def handle_mouse_press(self, e):
        x, y = e.x(), e.y()
        if not self.hold_to_draw:
            self.draw(x, y)
        else:
            self.reset_state()

    def handle_mouse_move(self, e):
        x, y = e.x(), e.y()
        if self.hold_to_draw:
            self.draw(x, y)

    def handle_mouse_release(self, _):
        if self.hold_to_draw:
            self.reset_state()

    def reset_state(self):
        self.last_x = None
        self.last_y = None

    def reset_overlay(self):
        self.overlay_pixmap = QPixmap(self.x, self.y)
        self.overlay_pixmap.fill(QtCore.Qt.transparent)
        self.image_label.setPixmap(self.base_pixmap)
        self.reset_state()

    def update_base(self, img):
        self.base_pixmap = QPixmap(img).scaled(self.x, self.y)
        self.reset_overlay()

    def update_base_pixmap(self, pixmap):
        assert isinstance(pixmap, QPixmap)
        self.x = pixmap.width()
        self.y = pixmap.height()
        self.base_pixmap = pixmap.copy()
        self.reset_overlay()

    def enable_erase_mode(self):
        self.mode_label.setText('Erasing mode')
        self.switch_button.setText('Pen')
        self.drawing_mode = False

    def disable_erase_mode(self):
        self.mode_label.setText('Drawing mode')
        self.switch_button.setText('Eraser')
        self.drawing_mode = True

    def toggle_erase(self):
        if self.drawing_mode:
            self.enable_erase_mode()
        else:
            self.disable_erase_mode()

    def wheelEvent(self, e):
        change = e.angleDelta().y()
        if change > 0:
            new_size = self.pen_size + self.pen_ticks
        else:
            new_size = self.pen_size - self.pen_ticks
        if self.pen_range[0] <= new_size <= self.pen_range[1]:
            self.pen_size = new_size
            self.pen_label.setText('Pen size: {}px'.format(new_size))

    def get_combined_pixmap(self):
        return overlay_pixmap(self.base_pixmap, self.overlay_pixmap)