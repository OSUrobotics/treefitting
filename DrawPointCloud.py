import sys

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)

import OpenGL.GL as GL

from Cylinder import Cylinder
from CylinderCover import CylinderCover
import numpy as np


class Window(QWidget):
    def __init__(self):
        super(Window, self).__init__()

        self.glWidget = DrawPointCloud(self)

        self.up_down = self.create_slider()
        self.turntable = self.create_slider()

        self.up_down.valueChanged.connect(self.glWidget.set_up_down_rotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.setValue)
        self.turntable.valueChanged.connect(self.glWidget.set_turntable_rotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.setValue)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.glWidget)
        main_layout.addWidget(self.turntable)
        main_layout.addWidget(self.up_down)
        self.setLayout(main_layout)

        self.up_down.setValue(15 * 16)
        self.turntable.setValue(345 * 16)

        self.setWindowTitle("Hello GL")

    @staticmethod
    def create_slider():
        slider = QSlider(Qt.Vertical)

        slider.setRange(0, 360 * 16)
        slider.setSingleStep(16)
        slider.setPageStep(15 * 16)
        slider.setTickInterval(15 * 16)
        slider.setTickPosition(QSlider.TicksRight)

        return slider


class DrawPointCloud(QOpenGLWidget):
    upDownRotationChanged = pyqtSignal(int)
    turntableRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)

    def __init__(self, gui, parent=None):
        super(DrawPointCloud, self).__init__(parent)

        self.object = 0
        self.up_down = 0
        self.turntable = 0
        self.zRot = 00

        self.pt_center = np.array([0, 0, 0])
        self.radius = 0.1

        self.pcd_gl_list = -1
        self.bin_gl_list = -1
        self.pcd_isolated_gl_list = -1
        self.pcd_bad_fit_gl_list = -1

        self.selected_point = 0

        self.gui = gui
        self.cyl = Cylinder()

        self.cyl_cover = CylinderCover()
        with open("data/cyl_cover_all.txt", "r") as fid:
            self.cyl_cover.read(fid)
        self.my_pcd = self.cyl_cover.my_pcd
        self.bin_mapping = self.set_bin_mapping()

        self.lastPos = QPoint()

        self.show_closeup = False
        self.show_one = False
        self.show_pca_cylinders = False
        self.show_fitted_cylinders = False
        self.show_bins = False
        self.show_isolated = False
        self.last_cyl = -1

        self.axis_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]

    def set_bin_mapping(self):
        bin_map = []
        for k, b in self.my_pcd.bin_list.items():
            bin_map.append((b[0], k))
        return bin_map

    @staticmethod
    def get_opengl_info():
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            GL.glGetString(GL.GL_VENDOR),
            GL.glGetString(GL.GL_RENDERER),
            GL.glGetString(GL.GL_VERSION),
            GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)
        )

        return info

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(1200, 1200)

    def set_up_down_rotation(self, angle):
        angle = self.normalize_angle(angle)
        if angle != self.up_down:
            self.up_down = angle
            self.upDownRotationChanged.emit(angle)
            self.update()

    def set_turntable_rotation(self, angle):
        angle = self.normalize_angle(angle)
        if angle != self.turntable:
            self.turntable = angle
            self.turntableRotationChanged.emit(angle)
            self.update()

    def initializeGL(self):
        print(self.get_opengl_info())

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.pcd_gl_list = self.make_pcd_gl_list()
        self.pcd_isolated_gl_list = self.make_isolated_gl_list()
        self.bin_gl_list = self.make_bin_gl_list()
        GL.glShadeModel(GL.GL_FLAT)
        #  GL.glEnable(GL.GL_DEPTH_TEST)
        #  GL.glEnable(GL.GL_CULL_FACE)

    def recalc_gl_ids(self):
        self.make_bin_gl_list()

    @staticmethod
    def draw_box(x_center, y_center, width, height=0):
        GL.glLoadIdentity()
        GL.glLineWidth(2.0)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glColor3d(0.75, 0.5, 0.75)
        bin_width = width / 2.0
        bin_height = height / 2.0
        if abs(bin_height) < 0.00001:
            bin_height = bin_width
        GL.glVertex2d(x_center - bin_width, y_center - bin_height)
        GL.glVertex2d(x_center - bin_width, y_center + bin_height)
        GL.glVertex2d(x_center + bin_width, y_center + bin_height)
        GL.glVertex2d(x_center + bin_width, y_center - bin_height)
        GL.glEnd()

    @staticmethod
    def draw_circle(x_center, y_center, circ_radius):
        GL.glLoadIdentity()
        GL.glLineWidth(2.0)

        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glColor4d(0.75, 0.25, 0.5, 1.0)
        for t in np.linspace(0, 2 * np.pi, 16):
            GL.glVertex2d(x_center + circ_radius * np.cos(t), y_center + circ_radius * np.sin(t))
        GL.glEnd()

    def draw_cyl(self, cyl):
        if hasattr(cyl, "pts") and self.show_one:
            GL.glPointSize(10)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(0.95, 0.9, 0.7)
            for p in cyl.pts:
                GL.glVertex3d(p[0], p[1], p[2])
            GL.glEnd()
        elif hasattr(cyl, "pts_ids") and self.show_one:
            GL.glPointSize(10)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(0.95, 0.9, 0.7)
            for pt_id in cyl.pts_ids:
                p = self.my_pcd.pt(pt_id)
                GL.glVertex3d(p[0], p[1], p[2])
            GL.glEnd()

        GL.glLineWidth(4)
        GL.glBegin(GL.GL_LINES)
        GL.glColor3f(1.0, 0.0, 0.0)
        l1 = cyl.pt_center + 0.5 * cyl.height * cyl.axis_vec
        l2 = cyl.pt_center - 0.5 * cyl.height * cyl.axis_vec
        GL.glVertex3d(l1[0], l1[1], l1[2])
        GL.glVertex3d(l2[0], l2[1], l2[2])

        GL.glColor3f(0.0, 1.0, 0.0)
        l1 = cyl.pt_center + cyl.radius * cyl.x_vec
        l2 = cyl.pt_center - cyl.radius * cyl.x_vec
        GL.glVertex3d(l1[0], l1[1], l1[2])
        GL.glVertex3d(l2[0], l2[1], l2[2])

        GL.glColor3f(0.0, 0.0, 1.0)
        l1 = cyl.pt_center + cyl.radius * cyl.y_vec
        l2 = cyl.pt_center - cyl.radius * cyl.y_vec
        GL.glVertex3d(l1[0], l1[1], l1[2])
        GL.glVertex3d(l2[0], l2[1], l2[2])
        GL.glEnd()

    def draw_bin_size(self, radius):
        GL.glLoadIdentity()
        scl = 2.0 / radius

        x_center = -0.95
        if self.show_closeup:
            x_center = 0.0

        # Actual bin size
        # Neighbor size is twice bin size
        if hasattr(self.my_pcd, "div"):
            self.draw_box(x_center, 0, scl * self.my_pcd.div)
            self.draw_circle(x_center, 0, scl * 2 * self.my_pcd.div)

        # What to set the bin size to
        try:
            height = scl * self.gui.branch_height.value()
            self.draw_box(x_center, -0.1, scl * self.gui.smallest_branch_width.value(), height)
            self.draw_box(x_center, -0.1, scl * self.gui.largest_branch_width.value(), height)
            self.draw_circle(x_center, -0.1, scl * self.gui.connected_neighborhood_radius.value())
        except AttributeError:
            pass

    def draw_bins(self):
        if self.bin_gl_list != -1:
            GL.glCallList(self.bin_gl_list)
        if self.show_closeup:
            which_bin = self.gui.show_closeup_slider.value()
            pt_ids = self.my_pcd.bin_list[self.bin_mapping[which_bin][1]]
            GL.glPointSize(10)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(0.0, 1.0, 0.0)
            for p_id in pt_ids:
                p = self.my_pcd.pt(p_id)
                GL.glVertex3d(p[0], p[1], p[2])
            GL.glEnd()

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        pt_center = self.pt_center
        radius = self.radius
        try:
            which_one = self.gui.show_closeup_slider.value()
        except AttributeError:
            which_one = 0

        if self.show_closeup:
            if self.show_bins:
                pt_center = self.my_pcd.pt(self.bin_mapping[which_one][0])
                radius = 2 * self.my_pcd.div
            elif self.show_pca_cylinders:
                pt_center = self.cyl_cover.cyls_pca[which_one].pt_center
                radius = self.cyl_cover.cyls_pca[which_one].height
            elif self.show_fitted_cylinders:
                pt_center = self.cyl_cover.cyls_fitted[which_one].pt_center
                radius = self.cyl_cover.cyls_fitted[which_one].height
            elif hasattr(self, "selected_point") and hasattr(self.my_pcd, "connected_radius"):
                pt_center = self.cyl.pt_center
                radius = 2 * self.my_pcd.connected_radius

        if hasattr(self.gui, "zoom"):
            radius = radius / self.gui.zoom.value()

        GL.glLoadIdentity()
        GL.glRotated(self.up_down, 1.0, 0.0, 0.0)
        GL.glRotated(self.turntable, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot, 0.0, 0.0, 1.0)
        GL.glScaled(2/radius, 2/radius, 2/radius)
        GL.glTranslated(-pt_center[0], -pt_center[1], -pt_center[2])
        if self.show_bins:
            self.draw_bins()
        else:
            GL.glCallList(self.pcd_gl_list)
            if self.show_isolated is True:
                GL.glCallList(self.pcd_isolated_gl_list)

        if hasattr(self, "cyl"):
            if len(self.cyl.data.pts_ids) > 0:
                self.draw_cyl(self.cyl)

        if self.show_pca_cylinders:
            if self.show_one:
                if self.last_cyl != which_one:
                    print(self.cyl_cover.cyls_pca[which_one])
                    self.last_cyl = which_one
                self.selected_point = self.cyl_cover.cyls_pca[which_one].id
                self.draw_cyl(self.cyl_cover.cyls_pca[which_one])
            else:
                clip_min = self.gui.show_min_val_slider.value()
                clip_max = self.gui.show_max_val_slider.value()
                for c in self.cyl_cover.cyls_pca:
                    if clip_min <= c.data.pca_err <= clip_max:
                        self.draw_cyl(c)

        if self.show_fitted_cylinders:
            if self.show_one:
                if self.last_cyl != which_one:
                    print(self.cyl_cover.cyls_fitted[which_one])
                    self.last_cyl = which_one
                self.selected_point = self.cyl_cover.cyls_fitted[which_one].id
                self.draw_cyl(self.cyl_cover.cyls_fitted[which_one])
            else:
                clip_min = self.gui.show_min_val_slider.value()
                clip_max = self.gui.show_max_val_slider.value()
                for c in self.cyl_cover.cyls_fitted:
                    if clip_min <= c.err <= clip_max:
                        self.draw_cyl(c)

        if hasattr(self, "selected_point"):
            GL.glPointSize(10)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(1.0, 0.0, 0.0)
            try:
                p = self.my_pcd.pt(self.selected_point)
            except IndexError:
                pass
            GL.glVertex3d(p[0], p[1], p[2])
            GL.glEnd()

        self.draw_bin_size(radius)

    @staticmethod
    def resizeGL(width, height):
        side = min(width, height)
        if side < 0:
            return

        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-1.15, 1.15, -1.15, 1.15, -1.5, 1.5)
        GL.glMatrixMode(GL.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.set_up_down_rotation(self.up_down + 4 * dy)
            self.set_turntable_rotation(self.turntable + 4 * dx)

        self.lastPos = event.pos()

    def make_pcd_gl_list(self):
        self.pt_center = [0.5*(self.my_pcd.max_pt[i] + self.my_pcd.min_pt[i]) for i in range(0, 3)]
        self.radius = 0
        for i in range(0, 3):
            self.radius = max(self.radius, self.my_pcd.max_pt[i] - self.my_pcd.min_pt[i])

        if self.pcd_gl_list == -1:
            self.pcd_gl_list = GL.glGenLists(1)

        GL.glNewList(self.pcd_gl_list, GL.GL_COMPILE)

        GL.glPointSize(2)
        GL.glBegin(GL.GL_POINTS)
        n_neigh = 20
        for i,p in enumerate(self.my_pcd.pts()):
            if hasattr(self.my_pcd, "neighbors"):
                n_neigh = len(self.my_pcd.neighbors)
                if n_neigh  < 20:
                    GL.glColor3d(0.8, 0.2, 0.0)
                elif n_neigh < 30:
                    GL.glColor3d(0.6, 0.2, 0.0)
                else:
                    GL.glColor3d(0.2, 0.8, 0.2)
            else:
                GL.glColor3d(0.2, 0.8, 0.2)
            if n_neigh > 15:
                GL.glVertex3d(p[0], p[1], p[2])

        GL.glEnd()
        GL.glEndList()

        return self.pcd_gl_list

    def make_isolated_gl_list(self):
        if self.pcd_isolated_gl_list == -1:
            self.pcd_isolated_gl_list = GL.glGenLists(1)

        GL.glNewList(self.pcd_isolated_gl_list, GL.GL_COMPILE)

        GL.glPointSize(2)
        GL.glBegin(GL.GL_POINTS)
        for i, p in enumerate(self.my_pcd.pts()):
            if hasattr(self.my_pcd, "neighbors"):
                n_neigh = len(self.my_pcd.neighbors)
                if n_neigh == 0:
                    GL.glColor3d(0.8, 0.2, 0.0)
                elif n_neigh < 5:
                    GL.glColor3d(0.6, 0.2, 0.0)
                else:
                    GL.glColor3d(0.2, 0.8, 0.2)

                if n_neigh < 15:
                    GL.glVertex3d(p[0], p[1], p[2])

        GL.glEnd()
        GL.glEndList()

        return self.pcd_isolated_gl_list

    def make_bin_gl_list(self):
        if self.bin_gl_list == -1:
            self.bin_gl_list = GL.glGenLists(1)

        GL.glNewList(self.bin_gl_list, GL.GL_COMPILE)

        GL.glPointSize(3)
        try:
            clip_min = self.gui.show_min_val_slider.value()
            clip_max = self.gui.show_max_val_slider.value()
        except AttributeError:
            clip_min = 0
            clip_max = 500
        GL.glBegin(GL.GL_POINTS)
        for b in self.my_pcd.bin_list.values():
            if clip_min <= len(b) <= clip_max:
                d_col = max(0.1, min(1.0, len(b) / 15.0))
                GL.glColor3d(d_col, d_col, d_col)

                p = self.my_pcd.pt(b[0])
                GL.glVertex3d(p[0], p[1], p[2])

        GL.glEnd()
        GL.glEndList()

        return self.bin_gl_list

    def normalize_angle(self, angle):
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle

    def set_color(self, c):
        GL.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
