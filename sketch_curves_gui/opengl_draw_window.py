#!/usr/bin/env python3

from sketch_curves_gui.draw_images import DrawImages
from sketch_curves_gui.draw_curve_2d import DrawCurve2D
from sketch_curves_gui.draw_curve_3d import DrawCurve3D

import numpy as np
from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QOpenGLWidget)
import OpenGL.GL as GL

from utils.camera_projections import frame_at_z_near


class OopenGLDrawWindow(QOpenGLWidget):
    upDownRotationChanged = pyqtSignal(int)
    turntableRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)
    gl_inited = False

    def __init__(self, gui, parent=None, size_start=(2*640, 2*480)):
        super(OopenGLDrawWindow, self).__init__(parent)

        self.draw_images = DrawImages()
        self.draw_curve_2d = DrawCurve2D()
        self.draw_curve_3d = DrawCurve3D()

        self.draw_sketch_curve = True

        self.object = 0
        self.up_down = 0
        self.turntable = 0
        self.zRot = 0

        self.pt_center = np.array([0, 0, 0])

        self.selected_point = 0

        # Pointer back to sketch_curves_main_window
        self.gui = gui

        self.firstPos = QPoint()
        self.lastPos = QPoint()

        self.axis_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        self.aspect_ratio = 1.0
        self.im_size = (0, 0)
        self.size_start = size_start

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
        return QSize(self.size_start[0], self.size_start[1])

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

        GL.glShadeModel(GL.GL_FLAT)
        #  GL.glEnable(GL.GL_DEPTH_TEST)
        #  GL.glEnable(GL.GL_CULL_FACE)

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

    def set_2d_projection(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        aspect_ratio_window = self.height() / self.width()
        if self.draw_images.im_size[0] > 0:
            width_rgb_image = self.draw_images.im_size[0]
            height_rgb_image = self.draw_images.im_size[1]
            aspect_ratio_window = height_rgb_image / width_rgb_image
        GL.glOrtho(-1.0, 1.0, -aspect_ratio_window, aspect_ratio_window, -1.0, 1.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def draw_camera_frame_3d(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        if self.gui == None:
            return

        width_rgb_image = 640
        height_rgb_image = 480
        if self.draw_images.im_size[0] > 0:
            width_rgb_image = self.draw_images.im_size[0]
            height_rgb_image = self.draw_images.im_size[1]

        params = {"z_near": 1.0,
                  "z_far": 100.0,
                  "camera_width_angle": self.gui.horizontal_angle.value(),
                  "image_size": [width_rgb_image, height_rgb_image]}

        frame = frame_at_z_near(params)
        z_near = 1.0
        ang_width_half = 0.5 * np.pi * self.gui.horizontal_angle.value() / 180.0
        frame_width = z_near * np.tan(ang_width_half)
        frame_height = z_near * np.tan(ang_width_half)
        # rev = np.arctan2(frame_width, z_near)

        width_window = self.width()
        height_window = self.height()

        if width_window > height_window:
            # height will be set to 1, width to 1 +
            frame_width = (height_window / width_window) * frame_width
        else:
            # Scale height, keep width
            frame_height = (height_window / width_window) * frame_height

        GL.glFrustum(frame[0], frame[1], frame[2], frame[3], params['z_near'], params['z_far'])
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

        """Draw a frame to verify aspect ratio and camera param alignment"""
        pt_center = self.pt_center
        pt_center[2] = -1.0
        # if self.gui:
            # TODO Set pt_center[2] to middle of selected keyframe curve
            # if self.gui.fit_crv_3d:
                # pt_center[2] = self.gui.fit_crv_3d.crv_3d.pt_axis(0.5)[2]
                # pt_center[2] = -1.0
        scl_factor = 1
        if hasattr(self.gui, "zoom"):
            scl_factor = scl_factor / self.gui.zoom.value()

        GL.glLoadIdentity()

        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glLineWidth(6)
        GL.glBegin(GL.GL_LINE_LOOP)
        GL.glColor4d(1.0, 0.0, 1.0, 1.0)
        if width_rgb_image > height_rgb_image:
            x = 0.975 * np.tan(ang_width_half)
            y = 0.975 * np.tan(ang_width_half) * (height_rgb_image / width_rgb_image)
        else:
            x = 0.975 * np.tan(ang_width_half) * (width_rgb_image / height_rgb_image)
            y = 0.975 * np.tan(ang_width_half)
        z = - 1.0
        for p in ((-x, -y, z), (x, -y, z), (x , y, z), (-x, y, z)):
            GL.glVertex3d(p[0], p[1], p[2])
        GL.glEnd()

        # Rotate branch
        GL.glTranslated(pt_center[0], pt_center[1], pt_center[2])
        GL.glScaled(scl_factor, scl_factor, scl_factor)
        GL.glRotated(self.up_down, 1.0, 0.0, 0.0)
        GL.glRotated(self.turntable, 0.0, 1.0, 0.0)
        GL.glRotated(self.zRot, 0.0, 0.0, 1.0)
        GL.glTranslated(-pt_center[0], -pt_center[1], -pt_center[2])

    def paintGL(self):

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.set_2d_projection()
        self.draw_images.draw_image()

        GL.glShadeModel(GL.GL_FLAT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_TEXTURE_2D)
        for crv in self.gui.sketched_curves():
            if crv is not None:
                self.draw_curve_2d.draw_sketch(q_wind=self, sketched_curve=crv)
        for crv in self.gui.spline_curves():
            if crv is not None:
                self.draw_curve_2d.draw_backbone(crv)
                self.draw_curve_2d.draw_interior_rects(crv)
                self.draw_curve_2d.draw_edge_rects(crv)
                self.draw_curve_2d.draw_profile_curves(crv)

        if self.gui:
            try:
                pts_and_colors = self.gui.get_key_points()
                for pts, cols in pts_and_colors:
                    self.draw_curve_2d.draw_points(q_wind=self, pts=pts, col=cols)
            except AttributeError:
                pass
            try:
                self.draw_curve_2d.draw_vector(self.gui.get_vector())
            except AttributeError:
                pass

        self.draw_camera_frame_3d()
        self.draw_curve_3d.draw()

    @staticmethod
    def resizeGL(width, height):
        side = min(width, height)
        if side < 0:
            return

        GL.glViewport((width - side) // 2, (height - side) // 2, side, side)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def mousePressEvent(self, event):
        self.firstPos = event.pos()
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.set_up_down_rotation(self.up_down + 4 * dy)
            self.set_turntable_rotation(self.turntable + 4 * dx)

        self.lastPos = event.pos()

    def mouseReleaseEvent(self, event):
        """Either add a point to the backbone or a point to the crossbar
         Shift: Add a cross bar
         Cntr: Remove a point"""
        dx = event.x() - self.firstPos.x()
        dy = event.y() - self.firstPos.y()
        # Not a click
        if abs(dx) + abs(dy) > 5:
            print(f"Big {dx} {dy}")
            if event.modifiers() == Qt.ShiftModifier:
                if self.gui:
                    try:
                        self.gui.sketch_vector(dx, dy)
                    except AttributeError:
                        pass
            return
        
        if self.gui:
            if self.draw_sketch_curve:
                sc = self.gui.sketch_curve

                if event.modifiers() == Qt.ShiftModifier:
                    sc.add_crossbar_point(event.x(), event.y())
                elif event.modifiers() == Qt.ControlModifier:
                    sc.remove_point(event.x(), event.y())
                else:
                    sc.add_backbone_point(event.x(), event.y())
            else:
                try:
                    self.gui.key_point(event.x(), event.y())
                except AttributeError:
                    pass
        self.update()

    def normalize_angle(self, angle):
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360
        return angle

    def set_color(self, c):
        GL.glColor4f(c.redF(), c.greenF(), c.blueF(), c.alphaF())


if __name__ == '__main__':
    # THIS DOES NOT WORK - use Sketch_curvs_main_window
    pass
