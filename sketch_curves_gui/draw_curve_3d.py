#!/usr/bin/env python3

import numpy as np
import OpenGL.GL as GL


class DrawCurve3D():
    def __init__(self):

        self.n_along = 20
        self.n_around = 16

        self.crv_gl_list = -1

        self.show = True
        self.show_mesh = True

    def draw_crv_3d(self, crv_3d):
        """ Render curve as 3D generalized cylinder
        @param branch_crv - the actual 3D cylinder, which has had make_mesh called
        """
        # GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glColor3f(0.75, 0.9, 0.95)
        GL.glLineWidth(5)
        GL.glBegin(GL.GL_LINE_STRIP)
        # GL.glVertex3d(0, 0, -1)
        for t in np.linspace(0, 1, 15):
            v = crv_3d.pt_axis(t)
            GL.glVertex3d(v[0], v[1], v[2])
        GL.glEnd()

        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_DEPTH_TEST)
        ambient_light = 0.1 * np.ones((4, 1), dtype=float)
        diffuse_light = 0.75 * np.ones((4, 1), dtype=float)
        specular_light = diffuse_light * 0.5
        obj_col = diffuse_light * 0.5
        GL.glEnable(GL.GL_LIGHT0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, ambient_light)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, diffuse_light)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, specular_light)
        light_pos = np.ones((4, 1), dtype=float)
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

    def make_crv_gl_list(self, crvs):

        self.pt_center = [0.0, 0.0, 0.0]

        if self.crv_gl_list == -1:
            self.crv_gl_list = GL.glGenLists(1)

        GL.glNewList(self.crv_gl_list, GL.GL_COMPILE)

        for crv in crvs:
            crv.set_dims(self.n_along, self.n_around)
            crv.make_mesh()
            self.draw_crv_3d(crv)

        GL.glEndList()

        return self.crv_gl_list


if __name__ == '__main__':
    pass
