import sys
import math

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)

import OpenGL.GL as gl

from MyPointCloud import MyPointCloud
from Cylinder import Cylinder
from CylinderCover import CylinderCover
import numpy as np


class Window(QWidget):

    def __init__(self):
        super(Window, self).__init__()

        self.glWidget = DrawPointCloud(self)

        self.up_down = self.createSlider()
        self.turntable = self.createSlider()

        self.up_down.valueChanged.connect(self.glWidget.setUpDownRotation)
        self.glWidget.upDownRotationChanged.connect(self.up_down.setValue)
        self.turntable.valueChanged.connect(self.glWidget.setTurntableRotation)
        self.glWidget.turntableRotationChanged.connect(self.turntable.setValue)

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        mainLayout.addWidget(self.turntable)
        mainLayout.addWidget(self.up_down)
        self.setLayout(mainLayout)

        self.up_down.setValue(15 * 16)
        self.turntable.setValue(345 * 16)

        self.setWindowTitle("Hello GL")

    def createSlider(self):
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
        self.zRot = 90

        self.pcd_gl_list = -1
        self.bin_gl_list = -1
        self.pcd_isolated_gl_list = -1
        self.pcd_bad_fit_gl_list = -1

        self.gui = gui
        self.cyl = Cylinder()

        self.cyl_cover = CylinderCover("data/MyPointCloud.pickle")
        self.my_pcd = self.cyl_cover.my_pcd
        self.bin_mapping = self.set_bin_mapping()

        with open("data/cyl_cover_all.txt", 'r') as f:
            self.cyl_cover.read(f)

        self.lastPos = QPoint()

        self.show_closeup = False
        self.show_one = False
        self.show_pca_cylinders = False
        self.show_fitted_cylinders = False
        self.show_bins = False
        self.show_isolated = False
        self.last_cyl = -1

        self.axis_colors = [[1.0,0,0], [0,1.0,0], [0,0,1.0]]

    def set_bin_mapping(self):
        bin_map = []
        for k, b in self.my_pcd.bin_list.items():
            bin_map.append((b[0], k))
        return bin_map

    def getOpenglInfo(self):
        info = """
            Vendor: {0}
            Renderer: {1}
            OpenGL Version: {2}
            Shader Version: {3}
        """.format(
            gl.glGetString(gl.GL_VENDOR),
            gl.glGetString(gl.GL_RENDERER),
            gl.glGetString(gl.GL_VERSION),
            gl.glGetString(gl.GL_SHADING_LANGUAGE_VERSION)
        )

        return info

    def minimumSizeHint(self):
        return QSize(50, 50)

    def sizeHint(self):
        return QSize(1200, 1200)

    def setUpDownRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.up_down:
            self.up_down = angle
            self.upDownRotationChanged.emit(angle)
            self.update()

    def setTurntableRotation(self, angle):
        angle = self.normalizeAngle(angle)
        if angle != self.turntable:
            self.turntable = angle
            self.turntableRotationChanged.emit(angle)
            self.update()

    def initializeGL(self):
        print(self.getOpenglInfo())

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.pcd_gl_list = self.make_pcd_gl_list()
        self.pcd_isolated_gl_list = self.make_isolated_gl_list()
        self.bin_gl_list = self.make_bin_gl_list()
        gl.glShadeModel(gl.GL_FLAT)
        #gl.glEnable(gl.GL_DEPTH_TEST)
        #gl.glEnable(gl.GL_CULL_FACE)

    def drawBox(self, x_center, y_center, width, height=0):
        gl.glLoadIdentity()
        gl.glLineWidth(2.0)
        gl.glBegin( gl.GL_LINE_LOOP )
        gl.glColor3d(0.75, 0.5, 0.75)
        bin_width = width / 2.0
        bin_height = height / 2.0
        if abs(bin_height) < 0.00001:
            bin_height = bin_width
        gl.glVertex2d(x_center - bin_width, y_center - bin_height)
        gl.glVertex2d(x_center - bin_width, y_center + bin_height)
        gl.glVertex2d(x_center + bin_width, y_center + bin_height)
        gl.glVertex2d(x_center + bin_width, y_center - bin_height)
        gl.glEnd()

    def drawCircle(self, x_center, y_center, circ_radius):
        gl.glLoadIdentity()
        gl.glLineWidth(2.0)

        gl.glBegin( gl.GL_LINE_LOOP )
        gl.glColor4d(0.75, 0.25, 0.5, 1.0)
        for t in np.linspace(0, 2 * np.pi, 16):
            gl.glVertex2d( x_center + circ_radius * np.cos(t), y_center + circ_radius * np.sin(t) )
        gl.glEnd()

    def draw_cyl(self, cyl):
        if hasattr(cyl, "pts") and self.show_one == True:
            gl.glPointSize(10)
            gl.glBegin(gl.GL_POINTS)
            gl.glColor3f(0.95, 0.9, 0.7)
            for p in cyl.pts:
                gl.glVertex3d(p[0], p[1], p[2])
            gl.glEnd()
        elif hasattr(cyl, "pts_ids") and self.show_one == True:
            gl.glPointSize(10)
            gl.glBegin(gl.GL_POINTS)
            gl.glColor3f(0.95, 0.9, 0.7)
            for pt_id in cyl.pts_ids:
                p = self.my_pcd.pc_data[pt_id]
                gl.glVertex3d(p[0], p[1], p[2])
            gl.glEnd()

        gl.glLineWidth(4)
        gl.glBegin( gl.GL_LINES )
        gl.glColor3f(1.0, 0.0, 0.0)
        l1 = cyl.pt_center + 0.5 * cyl.height * cyl.axis_vec
        l2 = cyl.pt_center - 0.5 * cyl.height * cyl.axis_vec
        gl.glVertex3d( l1[0], l1[1], l1[2] )
        gl.glVertex3d( l2[0], l2[1], l2[2] )

        gl.glColor3f(0.0, 1.0, 0.0)
        l1 = cyl.pt_center + cyl.radius * cyl.x_vec
        l2 = cyl.pt_center - cyl.radius * cyl.x_vec
        gl.glVertex3d( l1[0], l1[1], l1[2] )
        gl.glVertex3d( l2[0], l2[1], l2[2] )

        gl.glColor3f(0.0, 0.0, 1.0)
        l1 = cyl.pt_center + cyl.radius * cyl.y_vec
        l2 = cyl.pt_center - cyl.radius * cyl.y_vec
        gl.glVertex3d( l1[0], l1[1], l1[2] )
        gl.glVertex3d( l2[0], l2[1], l2[2] )
        gl.glEnd()

    def draw_bin_size(self, radius):
        gl.glLoadIdentity()
        scl = 2.0 / radius

        x_center = -0.95
        if self.show_closeup:
            x_center = 0.0

        # Actual bin size
        # Neighbor size is twice bin size
        if hasattr( self.my_pcd, "div" ):
            self.drawBox( x_center, 0, scl * self.my_pcd.div )
            self.drawCircle( x_center, 0, scl * 2 * self.my_pcd.div )

        # What to set the bin size to
        try:
            height = scl * self.gui.branch_height.value()
            self.drawBox(x_center, -0.1, scl * self.gui.smallest_branch_width.value(), height )
            self.drawBox(x_center, -0.1, scl * self.gui.largest_branch_width.value(), height)
            self.drawCircle(x_center, -0.1, scl * self.gui.connected_neighborhood_radius.value() )
        except (AttributeError):
            pass

    def draw_bins(self):
        if self.bin_gl_list != -1:
            gl.glCallList(self.bin_gl_list)
        if self.show_closeup:
            which_bin = self.gui.show_closeup_slider.value()
            pt_ids = self.my_pcd.bin_list[self.bin_mapping[which_bin][1]]
            gl.glPointSize(10)
            gl.glBegin( gl.GL_POINTS )
            gl.glColor3f(0.0, 1.0, 0.0)
            for p_id in pt_ids:
                p = self.my_pcd.pc_data[p_id]
                gl.glVertex3d( p[0], p[1], p[2] )
            gl.glEnd()


    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        pt_center = self.pt_center
        radius = self.radius
        try:
            which_one = self.gui.show_closeup_slider.value()
        except (AttributeError):
            which_one = 0

        if self.show_closeup:
            if self.show_bins:
                pt_center = self.my_pcd.pc_data[self.bin_mapping[which_one][0]]
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

        gl.glLoadIdentity()
        gl.glRotated(self.up_down, 1.0, 0.0, 0.0)
        gl.glRotated(self.turntable, 0.0, 1.0, 0.0)
        gl.glRotated(self.zRot, 0.0, 0.0, 1.0)
        gl.glScaled( 2/radius, 2/radius, 2/radius )
        gl.glTranslated(-pt_center[0], -pt_center[1], -pt_center[2])
        if self.show_bins:
            self.draw_bins()
        else:
            gl.glCallList(self.pcd_gl_list)
            if self.show_isolated is True:
                gl.glCallList(self.pcd_isolated_gl_list)

        if hasattr(self, "cyl"):
            if len(self.cyl.pts_ids) > 0:
                self.draw_cyl(self.cyl)

        if self.show_pca_cylinders == True:
            if self.show_one:
                if self.last_cyl != which_one:
                    print(self.cyl_cover.cyls_pca[which_one])
                    self.last_cyl = which_one
                self.selected_point = self.cyl_cover.cyls_pca[which_one].id
                self.draw_cyl(self.cyl_cover.cyls_pca[which_one])
            else:
                for c in self.cyl_cover.cyls_pca:
                    self.draw_cyl(c)

        if self.show_fitted_cylinders == True:
            if self.show_one:
                if self.last_cyl != which_one:
                    print(self.cyl_cover.cyls_fitted[which_one])
                    self.last_cyl = which_one
                self.selected_point = self.cyl_cover.cyls_fitted[which_one].id
                self.draw_cyl(self.cyl_cover.cyls_fitted[which_one])
            else:
                for c in self.cyl_cover.cyls_fitted:
                    self.draw_cyl(c)

        if hasattr(self, "selected_point"):
            gl.glPointSize(10)
            gl.glBegin( gl.GL_POINTS )
            gl.glColor3f(1.0,0.0,0.0)
            p = self.my_pcd.pc_data[self.selected_point]
            gl.glVertex3d( p[0], p[1], p[2] )
            gl.glEnd()

        self.draw_bin_size(radius)

    def resizeGL(self, width, height):
        side = min(width, height)
        if side < 0:
            return

        gl.glViewport((width - side) // 2, (height - side) // 2, side, side)

        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(-1.15, 1.15, -1.15, 1.15, -1.5, 1.5)
        gl.glMatrixMode(gl.GL_MODELVIEW)

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.setUpDownRotation(self.up_down + 4 * dy)
            self.setTurntableRotation(self.turntable + 4 * dx)

        self.lastPos = event.pos()

    def make_pcd_gl_list(self):
        self.pt_center = [0.5*(self.my_pcd.max_pt[i] + self.my_pcd.min_pt[i]) for i in range(0,3)]
        self.radius = 0
        for i in range(0,3):
            self.radius = max( self.radius, self.my_pcd.max_pt[i] - self.my_pcd.min_pt[i] )

        if self.pcd_gl_list == -1:
            self.pcd_gl_list = gl.glGenLists(1)

        gl.glNewList(self.pcd_gl_list, gl.GL_COMPILE)

        gl.glPointSize(2)
        gl.glBegin(gl.GL_POINTS)
        n_neigh = 20
        for i,p in enumerate(self.my_pcd.pc_data):
            if hasattr(self.my_pcd, "neighbors"):
                n_neigh = len( self.my_pcd.neighbors )
                if n_neigh  < 20:
                    gl.glColor3d( 0.8, 0.2, 0.0 )
                elif n_neigh < 30:
                    gl.glColor3d( 0.6, 0.2, 0.0 )
                else:
                    gl.glColor3d( 0.2, 0.8, 0.2 )
            else:
                gl.glColor3d( 0.2, 0.8, 0.2 )
            if n_neigh > 15:
                gl.glVertex3d(p[0], p[1], p[2])

        gl.glEnd()
        gl.glEndList()

        return self.pcd_gl_list

    def make_isolated_gl_list(self):
        if self.pcd_isolated_gl_list == -1:
            self.pcd_isolated_gl_list = gl.glGenLists(1)

        gl.glNewList(self.pcd_isolated_gl_list, gl.GL_COMPILE)

        gl.glPointSize(2)
        gl.glBegin(gl.GL_POINTS)
        for i,p in enumerate(self.my_pcd.pc_data):
            if hasattr(self.my_pcd, "neighbors"):
                n_neigh = len( self.my_pcd.neighbors )
                if n_neigh == 0:
                    gl.glColor3d( 0.8, 0.2, 0.0 )
                elif n_neigh < 5:
                    gl.glColor3d( 0.6, 0.2, 0.0 )
                else:
                    gl.glColor3d( 0.2, 0.8, 0.2 )

                if n_neigh < 15:
                    gl.glVertex3d(p[0], p[1], p[2])

        gl.glEnd()
        gl.glEndList()

        return self.pcd_isolated_gl_list

    def make_bin_gl_list(self):
        if self.bin_gl_list == -1:
            self.bin_gl_list = gl.glGenLists(1)

        gl.glNewList(self.bin_gl_list, gl.GL_COMPILE)

        gl.glPointSize(3)
        gl.glBegin(gl.GL_POINTS)
        for b in self.my_pcd.bin_list.values():
            d_col = max(0.1, min(1.0, len(b) / 15.0 ))
            gl.glColor3d(d_col, d_col, d_col)

            p = self.my_pcd.pc_data[b[0]]
            gl.glVertex3d(p[0], p[1], p[2])

        gl.glEnd()
        gl.glEndList()

        return self.bin_gl_list

    def normalizeAngle(self, angle):
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle

    def setColor(self, c):
        gl.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
