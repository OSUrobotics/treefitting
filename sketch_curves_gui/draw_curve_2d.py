#!/usr/bin/env python3

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)
from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor
import OpenGL.GL as GL
import cv2
from ctypes import c_uint8

from bezier_cyl_3d_with_detail import BezierCyl3DWithDetail
from camera_projections import frame_at_z_near

import numpy as np


class DrawCurve2D():
    def __init__(self, gui):
        self.crv_gl_list = -1

        self.selected_point = 0

        self.firstPos = QPoint()
        self.lastPos = QPoint()

        # Number of points per segment
        self.n_pts_backbone = 8
        self.n_rects_edge = 6
        self.width_edge_rect = 0.2
        self.n_rects_interior = 6
        self.perc_interior_rect = 0.2
        self.show_backbone = True
        self.show_edge_rects = True
        self.show_interior_rects = True

        self.axis_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        self.aspect_ratio = 1.0
        self.im_size = (0, 0)

    def convert_pts(self, pts):
        pts[:, 0] = 2 * (pts[:, 0] / self.im_size[0] - 0.5)
        pts[:, 1] = -self.aspect_ratio * 2 * (pts[:, 1] / self.im_size[1] - 0.5)
        return pts

    def draw_backbone(self, crv):
        """ Draw a Bezier or bspline curve
        @param crv one of the two curve types"""

        if not self.show_backbone:
            return

        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(4)
        try:
            ts = np.linspace(0, crv.max_t(), self.n_pts_backbone * int(crv.max_t()))
            pts = crv.eval_crv(ts)
            edge_pts_left = crv.edge_pts(ts, 1.0)
            edge_pts_right = crv.edge_pts(ts, -1.0)

        except AttributeError:
            pts = crv.pt_axis(np.linspace(0, 1, self.n_pts_backbone))
            edge_pts_left = np.zeros((self.n_pts_backbone, 2))
            edge_pts_right = np.zeros((self.n_pts_backbone, 2))
            for i, t in enumerate(np.linspace(0, 1, self.n_pts_backbone)):
                edge_pts_left[i, :], edge_pts_right[i, :] = crv.edge_pts(t)

        GL.glBegin(GL.GL_LINE_STRIP)
        col_start = 0.5
        col_div = 0.5 / (len(pts) - 1.0)
        pts_backbone = self.convert_pts(pts)
        for p in pts_backbone:
            GL.glColor3d(col_start, col_start, col_start)
            GL.glVertex2d(p[0], p[1])
            col_start += col_div
        GL.glEnd()

        edge_pts_left = self.convert_pts(edge_pts_left)
        edge_pts_right = self.convert_pts(edge_pts_right)

        GL.glLineWidth(3)
        for pts in (edge_pts_left, edge_pts_right):
            col_start = 0.25
            col_div = 0.75 / (len(pts) - 1.0)
            GL.glBegin(GL.GL_LINE_STRIP)
            for p in pts:
                GL.glColor3d(col_start, col_start, col_start)
                GL.glVertex2d(p[0], p[1])
                col_start += col_div
            GL.glEnd()

    def draw_edge_rects(self, crv):
            """ Draw the edge rectangles
            @param crv one of the two curve types"""

        if not self.show_edge_rects:
            return

        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(4)

        try:
            ts = np.linspace(0, crv.max_t(), self.n_rects_edge * int(crv.max_t()))
            rects_left, rects_right = crv.
            pts = crv.eval_crv(ts)
            edge_pts_left = crv.edge_pts(ts, 1.0)
            edge_pts_right = crv.edge_pts(ts, -1.0)

        except AttributeError:

            rects, _ = crv.boundary_rects_image(int(self.im_size[0] / self.n_rects_edge), self.width_edge_rect)
            rects_left = rects[0:2:]
            rects_right = rects[1:2:]

            GL.glBegin(GL.GL_LINE_STRIP)
        GL.glLineWidth(2)
        if width_edge_rects > 0.0:
            rects, _ = crv.interior_rects_image(self.gui.step_size.value(), self.gui.width_inside.value())
            col_incr = 1.0 // len(rects)
            for i, r in enumerate(rects):
                GL.glColor3f(i * col_incr, 0.8, 0.8)
                GL.glBegin(GL.GL_LINE_LOOP)
                pts = self.convert_pts(r)
                for p in pts:
                    GL.glVertex2d(p[0], p[1])
                GL.glEnd()

        if self.gui.show_edge_rects_button.checkState():
            rects, _ = crv.boundary_rects_image(self.gui.step_size.value(), self.gui.width_edge.value())
            col_incr = 0.5 // len(rects)
            for i, r in enumerate(rects):
                GL.glColor3f(0.5 + i * col_incr, 0.3 + (i % 2) * 0.3, 0.5 + i * col_incr)
                GL.glBegin(GL.GL_LINE_LOOP)
                pts = self.convert_pts(r)
                for p in pts:
                    GL.glVertex2d(p[0], p[1])
                GL.glEnd()

        if self.gui.show_profiles_button.checkState():
            pts_reconstruct = np.zeros((len(self.gui.extract_crv.edge_stats["pixs_edge"]), 2))
            for i, pt_reconstruct in enumerate(self.gui.extract_crv.edge_stats["pixs_edge"]):
                pts_reconstruct[i, 0] = pt_reconstruct[0]
                pts_reconstruct[i, 1] = pt_reconstruct[1]
            pts = self.convert_pts(pts_reconstruct)

            GL.glPointSize(4.0)
            GL.glBegin(GL.GL_POINTS)
            GL.glColor3f(1.0, 0.5, 0.5)
            for pt in pts:
                GL.glVertex2d(pt[0], pt[1])
            GL.glEnd()

            # pixs_filtered or pixs_reconstruct
            b_do_profile_debug = False
            if b_do_profile_debug:
                pts_reconstruct = np.zeros((len(self.gui.extract_crv.edge_stats["pixs_filtered"]), 2))
                for i, pt_reconstruct in enumerate(self.gui.extract_crv.edge_stats["pixs_filtered"]):
                    pts_reconstruct[i, 0] = pt_reconstruct[0]
                    pts_reconstruct[i, 1] = pt_reconstruct[1]
                pts = self.convert_pts(pts_reconstruct)

                GL.glPointSize(2.0)
                GL.glBegin(GL.GL_POINTS)
                GL.glColor3f(1.0, 1.0, 0.5)
                for pt in pts:
                    GL.glVertex2d(pt[0], pt[1])
                GL.glEnd()

            for profile_crv, dir in zip([self.gui.extract_crv.left_curve, self.gui.extract_crv.right_curve],
                                        ['Left', 'Right']):
                col_incr = 0.5 // len(profile_crv)
                pts_reconstruct = np.zeros((len(profile_crv), 2))
                for i, pt in enumerate(profile_crv):
                    pt_reconstruct = self.gui.crv.bezier_crv_fit_to_edge.edge_offset_pt(pt[0], pt[1], dir)
                    pts_reconstruct[i, 0] = pt_reconstruct[0]
                    pts_reconstruct[i, 1] = pt_reconstruct[1]
                pts = self.convert_pts(pts_reconstruct)
                GL.glLineWidth(2.0)
                GL.glBegin(GL.GL_LINE_STRIP)
                for i, pt in enumerate(pts):
                    GL.glColor3f(0.5 + i * col_incr, 0.6, 0.5 + i * col_incr)
                    GL.glVertex2d(pt[0], pt[1])
                GL.glEnd()

        # GL.glBegin(GL.GL_LINE_LOOP)
        # GL.glColor3d(1.0, 1.0, 1.0)
        # GL.glVertex2d(-0.25, -0.25)
        # GL.glVertex2d( 0.25, -0.25)
        # GL.glVertex2d( 0.25,  0.25)
        # GL.glVertex2d(-0.25,  0.25)
        # GL.glEnd()

    def draw_sketch(self):
        """ The marks the user made"""
        if not self.gui or not self.gui.crv or not self.gui.sketch_curve:
            return
        qp = QPainter()
        qp.begin(self)
        pen_backbone = QPen(Qt.yellow, 3, Qt.SolidLine)
        pen_cross = QPen(Qt.blue, 4, Qt.SolidLine)
        pen_corner = QPen(Qt.white, 2, Qt.SolidLine)
        brush = QBrush(Qt.CrossPattern)
        qp.setPen(pen_backbone)
        qp.setBrush(brush)
        sc = self.gui.sketch_curve
        for pt in sc.backbone_pts:
            qp.drawLine(int(pt[0] - 5), int(pt[1]), int(pt[0] + 5), int(pt[1]))
            qp.drawLine(int(pt[0]), int(pt[1] - 5), int(pt[0]), int(pt[1] + 5))

        for pt1, pt2 in zip(sc.backbone_pts[0:-1], sc.backbone_pts[1:]):
            qp.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))

        qp.setPen(pen_cross)
        for pts in sc.cross_bars:
            for pt in pts:
                qp.drawLine(int(pt[0] - 3), int(pt[1]), int(pt[0] + 3), int(pt[1]))
                qp.drawLine(int(pt[0]), int(pt[1] - 3), int(pt[0]), int(pt[1] + 3))
            if len(pts) > 1:
                pt1 = pts[0]
                pt2 = pts[1]
                qp.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))

        qp.setPen(pen_corner)
        sc = self.gui.sketch_curve
        for pt in [self.gui.lower_left, self.gui.upper_right]:
            qp.drawLine(int(pt[0] - 5), int(pt[1]), int(pt[0] + 5), int(pt[1]))
            qp.drawLine(int(pt[0]), int(pt[1] - 5), int(pt[0]), int(pt[1] + 5))
        qp.end()

    def draw_camera_frame_3d(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()

        if self.gui == None:
            return

        width_rgb_image = 640
        height_rgb_image = 480
        if self.gui.crv:
            if self.gui.crv:
                width_rgb_image = self.gui.crv.image_rgb.shape[1]
                height_rgb_image = self.gui.crv.image_rgb.shape[0]

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
        if self.gui:
            if self.gui.fit_crv_3d:
                pt_center[2] = self.gui.fit_crv_3d.crv_3d.pt_axis(0.5)[2]
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
        for p in ((-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)):
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
        if self.gui:
            self.gui.set_corners()

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.set_2d_projection()
        self.draw_images()

        if self.gui.show_sketch_crv_button.checkState():
            if self.gui.crv_from_sketch:
                self.draw_crv_2d(self.gui.crv_from_sketch.sketch_crv)
        if self.gui.crv:
            if self.gui.show_mask_crv_button.checkState():
                self.draw_crv_2d(self.gui.crv.mask_crv.bezier_crv_fit_to_mask)
            if self.gui.show_edge_crv_button.checkState():
                self.draw_crv_2d(self.gui.crv.bezier_crv_fit_to_edge)

        GL.glShadeModel(GL.GL_FLAT)
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_TEXTURE_2D)
        self.draw_camera_frame_3d()
        if self.gui.fit_crv_3d:
            self.draw_crv_3d(self.gui.fit_crv_3d.crv_3d)

        if self.show and self.crv_gl_list is not None:
            GL.glCallList(self.crv_gl_list)

        self.draw_sketch()

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

        DrawSpline3D.gl_inited = True

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
            return

        if self.gui:
            sc = self.gui.sketch_curve

            if event.modifiers() == Qt.ShiftModifier:
                sc.add_crossbar_point(event.x(), event.y())
            elif event.modifiers() == Qt.ControlModifier:
                sc.remove_point(event.x(), event.y())
            else:
                sc.add_backbone_point(event.x(), event.y())

        self.update()

    def make_crv_gl_list(self):
        if not DrawSpline3D.gl_inited:
            return

        self.pt_center = [0.0, 0.0, 0.0]

        if self.crv_gl_list == -1:
            self.crv_gl_list = GL.glGenLists(1)

        GL.glNewList(self.crv_gl_list, GL.GL_COMPILE)

        for crv in self.crvs:
            crv.set_dims(self.gui.n_along.value(), self.gui.n_around.value())
            crv.make_mesh()
            self.draw_crv(crv)

        GL.glEndList()

        return self.crv_gl_list

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
    from Window_3D import Window_3D

    app = QApplication(sys.argv)
    window = Window_3D(DrawSpline3D)

    branch = BezierCyl3DWithDetail()

    branch.set_pts([506.5, 156.0, 0.0], [457.49999996771703, 478.9999900052037, 0.0], [521.5, 318.0, 0.0])
    branch.set_radii_and_junction(start_radius=10.5, end_radius=8.25, b_start_is_junction=True, b_end_is_bud=False)

    window.glWidget.crvs.append(branch)

    window.show()
    sys.exit(app.exec_())
