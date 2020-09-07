import sys

from PyQt5.QtCore import pyqtSignal, QPoint, QSize, Qt
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QOpenGLWidget, QSlider,
                             QWidget)

import OpenGL.GL as GL
from OpenGL.GL import shaders
import OpenGL.GLUT as GLUT
from OpenGL.arrays import vbo

from Cylinder import Cylinder
from CylinderCover import CylinderCover
import numpy as np
import ctypes

from tree_model import TreeModel
import matplotlib.pyplot as plt
plt.ion()

SHADER_CODE = """
#version {ver}
    in vec4 position;
    uniform mat4 modelViewMat;
    void main() {
    gl_Position = modelViewMat * position;
}
"""

# SHADER_CODE = """
# #version {ver}
#     in vec4 position;
#     uniform mat4 modelViewMat;
#     void main() {
#     gl_Position = position;
# }
# """

FRAGMENT_CODE = """
#version {ver}
    void main(){
    gl_FragColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
    }
"""


class Window(QWidget):
    def __init__(self, pcd_file=None, cover_file=None):



        super(Window, self).__init__()

        self.glWidget = DrawPointCloud(self, pcd_file=pcd_file, cover_file=cover_file)

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

    def __init__(self, gui, parent=None, pcd_file=None, cover_file=None):
        super(DrawPointCloud, self).__init__(parent)

        self.shader_program = None
        self.vbo = None

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

        self.mesh_list = None
        self.skeleton_list = None

        self.selected_point = 0
        self.highlighted_points = []
        self.ignore_points = set()

        self.gui = gui
        self.cyl = Cylinder()

        self.cyl_cover = CylinderCover(pcd_file)
        if cover_file:
            with open(cover_file, "r") as fid:
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
        self.show_skeleton = False
        self.show_points = True
        self.last_cyl = -1

        self.tree = None
        self.skeleton = None
        self.mesh = np.zeros((0,3))

        self.axis_colors = [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]

        self.last_tf_data = None
        self.repair_mode = False
        self.repair_value = None

    def reset_model(self, pc=None):
        if pc is None:
            pc = self.my_pcd.points.copy()
        self.tree = TreeModel.from_point_cloud(pc)
        self.tree.load_superpoint_graph()
        self.skeleton = None
        self.mesh = np.zeros((0, 3))

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

        return info, GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)

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
        info_msg, shader_ver = self.get_opengl_info()
        print(info_msg)

        shader_ver = shader_ver.decode("utf8").replace('.', '')

        vertex_shader = shaders.compileShader(SHADER_CODE.replace('{ver}', shader_ver), GL.GL_VERTEX_SHADER)
        fragment_shader = shaders.compileShader(FRAGMENT_CODE.replace('{ver}', shader_ver), GL.GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.pcd_gl_list = self.make_pcd_gl_list()
        # self.pcd_isolated_gl_list = self.make_isolated_gl_list()
        self.bin_gl_list = self.make_bin_gl_list()
        GL.glShadeModel(GL.GL_SMOOTH)
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
                p = self.my_pcd[pt_id]
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
                p = self.my_pcd[p_id]
                GL.glVertex3d(p[0], p[1], p[2])
            GL.glEnd()

    def paintGL(self):
        # GL.glClearColor(1.0, 1.0, 1.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)


        pt_center = self.pt_center
        radius = self.radius
        try:
            which_one = self.gui.show_closeup_slider.value()
        except AttributeError:
            which_one = 0

        if self.show_closeup:
            if self.show_bins:
                pt_center = self.my_pcd[self.bin_mapping[which_one][0]]
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

        modelview_matrix = GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX).T
        proj_matrix = GL.glGetFloatv(GL.GL_PROJECTION_MATRIX).T
        self.last_tf_data = (modelview_matrix, proj_matrix)

        if self.show_points:
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
        GL.glEnable(GL.GL_DEPTH_TEST)
        if hasattr(self, "selected_point"):
            try:
                p = self.my_pcd[self.selected_point]
                GL.glPointSize(10)
                GL.glBegin(GL.GL_POINTS)
                GL.glColor3f(1.0, 0.0, 0.0)

                GL.glVertex3d(p[0], p[1], p[2])
                GL.glEnd()

            except IndexError:
                pass

        if self.mesh_list is not None:
            GL.glCallList(self.mesh_list)
        if self.skeleton_list and self.show_skeleton:
            GL.glCallList(self.skeleton_list)

        GL.glDisable(GL.GL_DEPTH_TEST)
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
        if not self.repair_mode or self.tree.thinned_tree is None:
            return

        click_x = self.lastPos.x()
        click_y = self.lastPos.y()

        modelview_matrix, proj_matrix = self.last_tf_data
        all_nodes = list(self.tree.superpoint_graph.nodes)
        all_pts = np.array([self.tree.superpoint_graph.nodes[n]['point'] for n in all_nodes])

        all_pts_homog = np.ones((len(all_pts), 4))
        all_pts_homog[:, :3] = all_pts

        clip_space_pts = proj_matrix @ modelview_matrix @ all_pts_homog.T
        ndc = (clip_space_pts / clip_space_pts[3]).T[:, :3]
        # ndc = ndc[(np.abs(ndc[:,:2]) <= 1).all(axis=1)]

        # Transformation to screen space
        x, y, w, h = GL.glGetInteger(GL.GL_VIEWPORT)
        # n, f = GL.glGetFloatv(GL.GL_DEPTH_RANGE)

        screen_xy = ndc[:,:2] * np.array([w/2, h/2]) + np.array([x + w/2, y + h/2])
        # Need to flip y coordinate due to pixel coordinates being defined from the top-left
        screen_xy[:,1] = h - screen_xy[:,1]

        # Convert the clicked point in PyQt land to the OpenGL viewport land
        geom = self.frameGeometry()
        pyqt_w = geom.width()
        pyqt_h = geom.height()

        click_x_vp = click_x * w / pyqt_w
        click_y_vp = click_y * h / pyqt_h

        click_vp = np.array([click_x_vp, click_y_vp])

        all_dists = np.linalg.norm(screen_xy - click_vp, axis=1)
        min_node_idx = all_dists.argmin()
        chosen_node = all_nodes[min_node_idx]
        min_dist = all_dists[min_node_idx]

        if min_dist > 10:
            print('No nearby node detected')
            return

        self.tree.thinned_tree.handle_repair(chosen_node, self.repair_value)
        self.tree.assign_edge_colors(iterate=False)
        self.make_pcd_gl_list()
        self.initialize_skeleton()
        self.update()

        #
        # print('Closest node was {} ({:.2f} pixels away)'.format(chosen_node, min_dist))

    def mouseMoveEvent(self, event):
        if self.repair_mode:
            return

        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.set_up_down_rotation(self.up_down + 4 * dy)
            self.set_turntable_rotation(self.turntable + 4 * dx)

        self.lastPos = event.pos()

    def make_pcd_gl_list(self):

        #draw_points()
        # This is where the point drawing happens!

        self.pt_center = [0.5*(self.my_pcd.max_pt[i] + self.my_pcd.min_pt[i]) for i in range(0, 3)]
        self.radius = max(0.00001, np.max(np.abs(self.my_pcd.max_pt - self.my_pcd.min_pt)))

        if self.pcd_gl_list == -1:
            self.pcd_gl_list = GL.glGenLists(1)

        GL.glNewList(self.pcd_gl_list, GL.GL_COMPILE)

        # if self.tree is None:
        #     return self.pcd_gl_list

        GL.glPointSize(2)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glBegin(GL.GL_POINTS)

        n_neigh = 20

        if self.tree is None:
            GL.glColor4d(0.0, 0.0, 0.0, 0.0)
            GL.glVertex3d(0.0, 0.0, 0.0)
        else:

            for (r, g, b, a), point_indexes in self.tree.get_pt_colors():
                print(r, g, b, a)
                if a < 0.3:
                    continue

                GL.glColor3d(r, g, b)
                for pt_index in point_indexes:
                    if pt_index in self.ignore_points:
                        continue
                    GL.glVertex3d(*self.tree.points[pt_index])

        #
        #
        # for i,p in enumerate(self.my_pcd.points):
        #     if hasattr(self.my_pcd, "neighbors"):
        #         n_neigh = len(self.my_pcd.neighbors)
        #         if n_neigh  < 20:
        #             GL.glColor3d(0.8, 0.2, 0.0)
        #         elif n_neigh < 30:
        #             GL.glColor3d(0.6, 0.2, 0.0)
        #         else:
        #             GL.glColor3d(0.2, 0.8, 0.2)
        #     else:
        #         if i in self.highlighted_points:
        #             GL.glColor3d(0.8, 0.2, 0.2)
        #         else:
        #             GL.glColor3d(0.2, 0.8, 0.8)
        #     if n_neigh > 15:
        #         GL.glVertex3d(p[0], p[1], p[2])

        GL.glEnd()
        GL.glDisable(GL.GL_BLEND)
        GL.glEndList()



        return self.pcd_gl_list

    def make_isolated_gl_list(self):
        if self.pcd_isolated_gl_list == -1:
            self.pcd_isolated_gl_list = GL.glGenLists(1)

        GL.glNewList(self.pcd_isolated_gl_list, GL.GL_COMPILE)

        GL.glPointSize(2)
        GL.glBegin(GL.GL_POINTS)
        for i, p in enumerate(self.my_pcd.points):
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

                p = self.my_pcd[b[0]]
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

    def compute_mesh(self):

        self.tree.assign_branch_radii()
        self.tree.create_mesh()
        self.initialize_mesh()

    def save_mesh(self, file_name):
        if not file_name.endswith('.obj'):
            file_name += '.obj'
        self.tree.output_mesh(file_name)


    def initialize_skeleton(self):

        if self.tree.superpoint_graph is None:
            print('Skeleton has not been initialized')
            return

        if self.skeleton_list is None:
            self.skeleton_list = GL.glGenLists(1)

        GL.glNewList(self.skeleton_list, GL.GL_COMPILE)


        GL.glPointSize(6)
        GL.glBegin(GL.GL_POINTS)
        for node in self.tree.superpoint_graph.nodes:
            color = self.tree.superpoint_graph.nodes[node].get('color', (0.0, 1.0, 0.0))
            if not color:
                continue
            GL.glColor3f(*color)
            GL.glVertex3f(*self.tree.superpoint_graph.nodes[node]['point'])
        GL.glEnd()

        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)
        for edge in self.tree.superpoint_graph.edges:
            edge_color = self.tree.superpoint_graph.edges[edge].get('color', False)
            if not edge_color:
                continue

            start = self.tree.superpoint_graph.nodes[edge[0]]['point']
            end = self.tree.superpoint_graph.nodes[edge[1]]['point']
            GL.glColor3f(*edge_color)
            GL.glVertex3f(*start)
            GL.glVertex3f(*end)

        GL.glEnd()
        GL.glEndList()

    def initialize_mesh(self):

        if self.mesh_list is None:
            self.mesh_list = GL.glGenLists(1)

        GL.glNewList(self.mesh_list, GL.GL_COMPILE)

        # GL.glPointSize(2)
        GL.glLineWidth(12)
        GL.glPointSize(5)
        GL.glBegin(GL.GL_TRIANGLES)


        GL.glColor4d(0.8, 0.3, 0.3, 0.5)

        vertices = self.tree.mesh['v']
        faces = self.tree.mesh['f']

        for face_pt in faces.reshape((-1,)):
            GL.glVertex3d(*vertices[face_pt])

        GL.glEnd()
        GL.glEndList()

    def magic(self):
        import ipdb
        ipdb.set_trace()

        pass


    # def update_highlighted_points(self, points):
    #     # Should be express as a set of indexes
    #     self.highlighted_points = points
    #     self.make_pcd_gl_list()

    # def random_select_radius(self, radius = 0.15):
    #     # Picks a random point in the point cloud and all points with a certain radius
    #     pt_indexes, ref_index = self.tree.query_random_radius(radius)
    #     self.update_highlighted_points(pt_indexes)
    #
    #     self.update()
    #     from exp_joint_detector import convert_pc_to_grid
    #     img = convert_pc_to_grid(self.tree.points[pt_indexes], self.tree.points[ref_index])
    #
    #     return img
    #
    #


if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
