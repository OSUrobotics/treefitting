import os
import sys
sys.path.insert(0, os.path.abspath('./'))
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../Image_based'))

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)
from PyQt5.QtGui import QPainter, QBrush, QPen, QFont, QColor
import OpenGL.GL as GL
import cv2
from ctypes import c_uint8

from bezier_cyl_3d_with_detail import BezierCyl3DWithDetail
import numpy as np


class DrawSpline3D(QOpenGLWidget):
    upDownRotationChanged = pyqtSignal(int)
    turntableRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)
    gl_inited = False

    def __init__(self, gui, parent=None):
        super(DrawSpline3D, self).__init__(parent)

        self.object = 0
        self.up_down = 0
        self.turntable = 0
        self.zRot = 00

        self.pt_center = np.array([0, 0, 0])

        self.crv_gl_list = -1
        self.image_gl_tex = []

        self.selected_point = 0

        # Pointer back to sketch_curves_main_window
        self.gui = gui
        self.crvs = []

        self.firstPos = QPoint()
        self.lastPos = QPoint()

        self.show = True

        self.axis_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]
        self.aspect_ratio = 1.0
        self.im_size = (0, 0)

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

        self.crv_gl_list = self.make_crv_gl_list()
        GL.glShadeModel(GL.GL_FLAT)
        #  GL.glEnable(GL.GL_DEPTH_TEST)
        #  GL.glEnable(GL.GL_CULL_FACE)

    def recalc_gl_ids(self):
        self.crv_gl_list = self.make_crv_gl_list()

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

    def draw_crv_3d(self, crv_3d):
        """ Render curve as 3D generalized cylinder
        @param branch_crv - the actual 3D cylinder, which has had make_mesh called
        """
        # GL.glEnable(GL.GL_DEPTH_TEST)
        if self.gui.show_3d_crv_axis_button.checkState():
            GL.glColor3f(0.75, 0.9, 0.95)
            GL.glLineWidth(5)
            GL.glBegin(GL.GL_LINE_STRIP)
            #GL.glVertex3d(0, 0, -1)
            for t in np.linspace(0, 1, 15):
                v = crv_3d.pt_axis(t)
                GL.glVertex3d(v[0], v[1], v[2])
            GL.glEnd()

        if self.gui.show_3d_crv_button.checkState():
            GL.glEnable(GL.GL_LIGHTING)
            GL.glEnable(GL.GL_DEPTH_TEST)
            ambient_light = 0.1 * np.ones((4,1), dtype=float)
            diffuse_light = 0.75 * np.ones((4,1), dtype=float)
            specular_light = diffuse_light * 0.5
            obj_col = diffuse_light * 0.5
            GL.glEnable(GL.GL_LIGHT0)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, ambient_light)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, diffuse_light)
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, specular_light)
            light_pos = np.ones((4,1), dtype=float)
            light_pos[0] = 2.0
            light_pos[1] = 5.0
            GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_pos)
            obj_col[0] = 0.75
            obj_col[1] = 0.1
            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, obj_col)
            GL.glColor3f(0.75, 0.5, 0.95)
            for it in range(0, crv_3d.n_along - 1):
                GL.glBegin(GL.GL_TRIANGLE_STRIP)
                # The first two vertices
                #  Alternate left, right
                for ir in range(0, crv_3d.n_around):
                    ir_next = (ir + 1) % crv_3d.n_around
                    v = crv_3d.vertex_locs[it, ir, :]
                    n = crv_3d.vertex_normals[it, ir, :]
                    GL.glVertex3d(v[0], v[1], v[2])
                    GL.glNormal3d(n[0], n[1], n[2])
                    v = crv_3d.vertex_locs[it + 1, ir_next, :]
                    n = crv_3d.vertex_normals[it + 1, ir_next, :]
                    GL.glVertex3d(v[0], v[1], v[2])
                    GL.glNormal3d(n[0], n[1], n[2])
                GL.glEnd()

    def bind_texture(self, rgb_image, mask_image, edge_image, flow_image, depth_image):
        print(f"Binding texture {rgb_image.shape} {edge_image.shape}")
        self.aspect_ratio = rgb_image.shape[0] / rgb_image.shape[1]
        self.im_size = (rgb_image.shape[1], rgb_image.shape[0])

        #im_check = np.ones((im_size, im_size, 3), dtype=np.uint8)
        #im_check[:,:] *= 64
        #im_check[:,:,0] *= 128
        #im_check[:,:,0] *= 192

        im_size = 512
        if rgb_image.shape[0] > 1024:
            im_size = 1024
        im_sq = cv2.resize(rgb_image, (im_size, im_size))

        im_sq_mask = cv2.cvtColor(cv2.resize(mask_image, (im_size, im_size)), cv2.COLOR_GRAY2RGB)
        im_sq_edge = cv2.cvtColor(cv2.resize(edge_image, (im_size, im_size)), cv2.COLOR_GRAY2RGB)
        if flow_image is not None:
            im_sq_flow = cv2.resize(flow_image, (im_size, im_size))
        else:
            im_sq_flow = None
        if depth_image is not None:
            im_sq_depth = cv2.resize(depth_image, (im_size, im_size))
        else:
            im_sq_depth = None

        im_sq_rgb_edge = im_sq // 2
        im_sq_rgb_mask = im_sq // 2
        im_sq_rgb_mask_edge = im_sq // 2
        for ch in (1, 2):
            im_sq_rgb_edge[:, :, ch] = im_sq_rgb_edge[:, :, ch] + im_sq_edge[:, :, ch] // 2
            im_sq_rgb_mask_edge[:, :, ch] = im_sq_rgb_mask_edge[:, :, ch] + im_sq_edge[:, :, ch] // 2
        im_sq_rgb_mask[:, :, 0] = im_sq_rgb_mask[:, :, 0] + im_sq_mask[:, :, 0] // 2
        im_sq_rgb_mask_edge[:, :, 0] = im_sq_rgb_mask_edge[:, :, 0] + im_sq_mask[:, :, 0] // 2

        if len(self.image_gl_tex) == 0:
            n_textures = 8
            self.image_gl_tex = GL.glGenTextures(n_textures)
        for i, im in enumerate([im_sq, im_sq_mask, im_sq_edge, im_sq_rgb_mask, im_sq_rgb_edge, im_sq_rgb_mask_edge, im_sq_flow, im_sq_depth]):
            if im is None:
                self.image_gl_tex[i] = -1
            else:
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_gl_tex[i])
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
                c_my_texture = (c_uint8 * im_size * im_size)() # copying under correct ctype format (likely clumsy)
                c_my_texture.value = im[:,:,:]
                GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, 3, im_size, im_size, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, c_my_texture.value)

    def set_2d_projection(self):
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        if self.height() > self.width():
            aspect_ratio_window = self.height() / self.width()
            GL.glOrtho(-1.0, 1.0, -aspect_ratio_window, aspect_ratio_window, -1.0, 1.0)
        else:
            aspect_ratio_window = self.width() / self.height()
            GL.glOrtho(-aspect_ratio_window, aspect_ratio_window, -1.0, 1.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()

    def draw_images(self):
        if len(self.image_gl_tex) == 0:
            return

        tex_indx = -1
        if self.gui.show_rgb_button.checkState():
            if self.gui.show_edge_button.checkState():
                if self.gui.show_mask_button.checkState():
                    tex_indx = 5
                else:
                    tex_indx = 4
            elif self.gui.show_mask_button.checkState():
                tex_indx = 3
            else:
                tex_indx = 0
        elif self.gui.show_edge_button.checkState():
            tex_indx = 2
        elif self.gui.show_mask_button.checkState():
            tex_indx = 1
        elif self.gui.show_opt_flow_button.checkState():
            tex_indx = 6
        elif self.gui.show_depth_button.checkState():
            tex_indx = 7

        if tex_indx == -1:
            return

        if self.image_gl_tex[tex_indx] != -1:
            GL.glTexEnvf(GL.GL_TEXTURE_ENV, GL.GL_TEXTURE_ENV_MODE, GL.GL_DECAL)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.image_gl_tex[tex_indx])
            GL.glEnable(GL.GL_TEXTURE_2D)

            quad_size_x = 1.0
            quad_size_y = self.aspect_ratio * quad_size_x
            GL.glBegin(GL.GL_QUADS)
            GL.glTexCoord2d(0.0, 1.0)
            GL.glVertex2f(-quad_size_x, -quad_size_y)
            GL.glTexCoord2d(1.0, 1.0)
            GL.glVertex2f(quad_size_x, -quad_size_y)
            GL.glTexCoord2d(1.0, 0.0)
            GL.glVertex2f(quad_size_x, quad_size_y)
            GL.glTexCoord2d(0.0, 0.0)
            GL.glVertex2f(-quad_size_x, quad_size_y)
            GL.glEnd()

    def convert_pts(self, pts):
        pts[:, 0] = 2 * (pts[:, 0] / self.im_size[0] - 0.5)
        pts[:, 1] = -self.aspect_ratio * 2 * (pts[:, 1] / self.im_size[1] - 0.5)
        return pts

    def draw_crv_2d(self):
        if not self.gui or not self.gui.crv:
            return

        crv = self.gui.crv.mask_crv.bezier_crv_fit_to_mask
        if self.gui.show_edge_button.checkState():
            crv = self.gui.crv.bezier_crv_fit_to_edge
        elif self.gui.crv_from_sketch:
            crv = self.gui.crv_from_sketch.sketch_crv

        if self.gui.show_backbone_button.checkState():
            n_pts_quad = 6
            pts = self.convert_pts(crv.pt_axis(np.linspace(0, 1, n_pts_quad)))

            GL.glDisable(GL.GL_LIGHTING)
            GL.glLineWidth(4)
            GL.glBegin(GL.GL_LINE_STRIP)
            col_start = 0.5
            col_div = 0.5 / (n_pts_quad - 1.0)
            for p in pts:
                GL.glColor3d(col_start, col_start, col_start)
                GL.glVertex2d(p[0], p[1])
                col_start += col_div
            GL.glEnd()

            edge_pts_left = np.zeros((n_pts_quad, 2))
            edge_pts_right = np.zeros((n_pts_quad, 2))
            for i, t in enumerate(np.linspace(0, 1, n_pts_quad)):
                edge_pts_left[i, :], edge_pts_right[i, :] = crv.edge_pts(t)
            edge_pts_left = self.convert_pts(edge_pts_left)
            edge_pts_right = self.convert_pts(edge_pts_right)

            GL.glLineWidth(3)
            for pts in (edge_pts_left, edge_pts_right):
                col_start = 0.25
                col_div = 0.75 / (n_pts_quad - 1.0)
                GL.glBegin(GL.GL_LINE_STRIP)
                for p in pts:
                    GL.glColor3d(col_start, col_start, col_start)
                    GL.glVertex2d(p[0], p[1])
                    col_start += col_div
                GL.glEnd()

        GL.glLineWidth(2)
        if self.gui.show_interior_rects_button.checkState():
            rects, _ = crv.interior_rects(self.gui.step_size.value(), self.gui.width_inside.value())
            col_incr = 1.0 // len(rects)
            for i, r in enumerate(rects):
                GL.glColor3f(i * col_incr, 0.8, 0.8)
                GL.glBegin(GL.GL_LINE_LOOP)
                pts = self.convert_pts(r)
                for p in pts:
                    GL.glVertex2d(p[0], p[1])
                GL.glEnd()

        if self.gui.show_edge_rects_button.checkState():
            rects, _ = crv.boundary_rects(self.gui.step_size.value(), self.gui.width_edge.value())
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

            for profile_crv, dir in zip([self.gui.extract_crv.left_curve, self.gui.extract_crv.right_curve], ['Left', 'Right']):
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


        #GL.glBegin(GL.GL_LINE_LOOP)
        #GL.glColor3d(1.0, 1.0, 1.0)
        #GL.glVertex2d(-0.25, -0.25)
        #GL.glVertex2d( 0.25, -0.25)
        #GL.glVertex2d( 0.25,  0.25)
        #GL.glVertex2d(-0.25,  0.25)
        #GL.glEnd()

    def draw_sketch(self):
        """ The marks the user made"""
        if not self.gui or not self.gui.crv:
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

        z_near = 0.1
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

        GL.glFrustum(-frame_width, frame_width, -frame_height, frame_height, z_near, 100.0)
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

        width_rgb_image = 640
        height_rgb_image = 640
        if self.gui.crv:
            if self.gui.crv:
                width_rgb_image = self.gui.crv.image_rgb.shape[1]
                height_rgb_image = self.gui.crv.image_rgb.shape[0]

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
        if self.gui:
            self.gui.set_corners()

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        self.set_2d_projection()
        self.draw_images()
        self.draw_crv_2d()

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
        GL.glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)

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
