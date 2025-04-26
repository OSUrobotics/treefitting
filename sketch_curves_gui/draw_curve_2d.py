#!/usr/bin/env python3

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)
from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor
import OpenGL.GL as GL

from camera_projections import frame_at_z_near

import numpy as np


class DrawCurve2D():
    def __init__(self):
        self.crv_gl_list = -1

        self.selected_point = 0

        # Number of points per segment
        self.n_pts_backbone = 8
        self.n_rects_edge = 6
        self.width_edge_rect = 0.2
        self.n_rects_interior = 6
        self.perc_interior_rect = 0.2
        self.show_backbone = True
        self.show_edge_rects = True
        self.show_interior_rects = True
        self.show_profile_curves = False

        self.show_sketched_curve = True

        self.axis_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        self.aspect_ratio = 1.0
        self.lower_left = [0, 0]
        self.upper_right = [1, 1]
        self.im_size = (0, 0)

    def convert_pts(self, pts):
        pts[:, 0] = 2.0 * (pts[:, 0] / self.im_size[0] - 0.5)
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

        # Do the backbone/curve
        GL.glBegin(GL.GL_LINE_STRIP)
        col_start = 0.5
        col_div = 0.5 / (len(pts) - 1.0)
        pts_backbone = self.convert_pts(pts)
        for p in pts_backbone:
            GL.glColor3d(col_start, col_start, col_start)
            GL.glVertex2d(p[0], p[1])
            col_start += col_div
        GL.glEnd()

        # Do the edges
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
            rects, _ = crv.edge_rects(ts, self.width_edge_rect)
        except AttributeError:
            rects, _ = crv.boundary_rects_image(int(self.im_size[0] / self.n_rects_edge), self.width_edge_rect)

        rects_left = rects[0:2:]
        rects_right = rects[1:2:]

        GL.glLineWidth(2)

        col_incr = 1.0 // len(rects_left)
        for i, (rl, rr) in enumerate(zip(rects_left, rects_right)):
            for j, r in enumerate((rl, rr)):
                GL.glColor3f(0.5 + i * col_incr, 0.3 + j * 0.3, 0.5 + i * col_incr)
                GL.glBegin(GL.GL_LINE_LOOP)
                pts = self.convert_pts(r)
                for p in pts:
                    GL.glVertex2d(p[0], p[1])
                GL.glEnd()

    def draw_interior_rects(self, crv):
        """ Draw the interior rectangles
        @param crv one of the two curve types"""

        if not self.show_interior_rects:
            return

        GL.glDisable(GL.GL_LIGHTING)
        GL.glLineWidth(4)

        try:
            ts = np.linspace(0, crv.max_t(), self.n_rects_interior * int(crv.max_t()))
            rects, _ = crv.interior_rects(ts, self.perc_interior_rect)
        except AttributeError:
            rects, _ = crv.interior_rects(int(self.im_size[0] / self.n_rects_interior), self.perc_interior_rect)

        GL.glLineWidth(2)

        col_incr = 1.0 // len(rects)
        for i, r in enumerate(rects):
            GL.glColor3f(0.5 + i * col_incr, 0.6, 0.5 + i * col_incr)
            GL.glBegin(GL.GL_LINE_LOOP)
            pts = self.convert_pts(r)
            for p in pts:
                GL.glVertex2d(p[0], p[1])
            GL.glEnd()

    def draw_profile_curves(self, crv):
        """ Draw the profie curves made by extracting the pixels along the boundary"""

        if not self.show_profile_curves:
            return

        crv_pts = [] # self.gui.extract_crv.edge_stats["pixs_edge"]
        pts_reconstruct = np.zeros((len(crv_pts), 2))
        for i, pt_reconstruct in enumerate(crv_pts):
            pts_reconstruct[i, 0] = pt_reconstruct[0]
            pts_reconstruct[i, 1] = pt_reconstruct[1]
        pts = self.convert_pts(pts_reconstruct)

        GL.glPointSize(4.0)
        GL.glBegin(GL.GL_POINTS)
        GL.glColor3f(1.0, 0.5, 0.5)
        for pt in pts:
            GL.glVertex2d(pt[0], pt[1])
        GL.glEnd()

        """
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
        # GL.glEnd()"""

    def draw_sketch(self, q_wind, sketched_curve):
        """ The marks the user made"""

        if not self.show_sketched_curve:
            return

        qp = QPainter()
        qp.begin(q_wind)
        pen_backbone = QPen(Qt.yellow, 3, Qt.SolidLine)
        pen_cross = QPen(Qt.blue, 4, Qt.SolidLine)
        pen_corner = QPen(Qt.white, 2, Qt.SolidLine)
        brush = QBrush(Qt.CrossPattern)
        qp.setPen(pen_backbone)
        qp.setBrush(brush)

        for pt in sketched_curve.backbone_pts:
            qp.drawLine(int(pt[0] - 5), int(pt[1]), int(pt[0] + 5), int(pt[1]))
            qp.drawLine(int(pt[0]), int(pt[1] - 5), int(pt[0]), int(pt[1] + 5))

        for pt1, pt2 in zip(sketched_curve.backbone_pts[0:-1], sketched_curve.backbone_pts[1:]):
            qp.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))

        qp.setPen(pen_cross)
        for pts in sketched_curve.cross_bars:
            for pt in pts:
                qp.drawLine(int(pt[0] - 3), int(pt[1]), int(pt[0] + 3), int(pt[1]))
                qp.drawLine(int(pt[0]), int(pt[1] - 3), int(pt[0]), int(pt[1] + 3))
            if len(pts) > 1:
                pt1 = pts[0]
                pt2 = pts[1]
                qp.drawLine(int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1]))

        qp.setPen(pen_corner)
        for pt in [self.lower_left, self.upper_right]:
            qp.drawLine(int(pt[0] - 5), int(pt[1]), int(pt[0] + 5), int(pt[1]))
            qp.drawLine(int(pt[0]), int(pt[1] - 5), int(pt[0]), int(pt[1] + 5))
        qp.end()


if __name__ == '__main__':
    # THIS DOES NOT WORK - use Sketch_curvs_main_window
    pass
