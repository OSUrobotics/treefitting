#!/usr/bin/env python3

# a 3D Bezier cylinder
#  - The 3D bezier curve
#  - Start and end radii
#
# Can:
#   - Evaluate points along the curve
#   - Make a cylinderical mesh
#   - Project self into an image

import numpy as np
from json import load, dump


class BezierCyl3D:

    def __init__(self, p1=(0.0, 0.0, 0.0), p2=(0.5, 0.75, 0.5), p3=(1.0, 1.0, 1.0), start_radius=10.0, end_radius=20.0):
        """ Initialize a 3D curve, built from a quadratic Bezier with radii
        @param p1 - start pt, x,y,z
        @param p2 - mid pt, x,y,z
        @param p3 - end pt x,y,z
        @param start_radius - starting radius
        @param end_radius - ending radius"""

        # Information about the current branch/trunk
        self.pt1 = np.array(p1)
        self.pt2 = np.array(p2)
        self.pt3 = np.array(p3)
        self.start_radii = start_radius
        self.end_radii = end_radius

        # Drawing/mesh creation parameters
        self.n_along = 10
        self.n_around = 64

        self.vertex_locs = np.zeros((self.n_along, self.n_around, 3))
        self.make_mesh()

    def copy(self, bezier_crv=None, b_compute_mesh=False):
        """Return a copy of self - mostly just copy, rather than =, the points/vertices
        @param bezier_crv - use this bezier curve versus making a new one
        @param b_compute_mesh - compute/copy the mesh, y/n
        @return: New curve"""
        if bezier_crv is None:
            bezier_crv = BezierCyl3D(np.copy(self.pt1), np.copy(self.pt2), np.copy(self.pt3), self.start_radii, self.end_radii)
        for k, v in self.__dict__:
            try:
                if v.size > 1:
                    pass
            except TypeError:
                setattr(bezier_crv, k, v)

        if b_compute_mesh:
            bezier_crv.vertex_locs = np.copy(self.vertex_locs)
        return bezier_crv

    def n_vertices(self):
        return self.n_along * self.n_around

    def set_dims(self, n_along=10, n_radial=64):
        self.n_along = n_along
        self.n_around = n_radial
        self.vertex_locs = np.zeros((self.n_along, self.n_around, 3))

    def set_pts(self, pt1, pt2, pt3):
        """ Turn into numpy array
        @param pt1 First point
        @param pt2 Mid point
        @param pt3 End point
        """
        self.pt1 = np.array(pt1)
        self.pt2 = np.array(pt2)
        self.pt3 = np.array(pt3)

    def set_pts_from_pt_tangent(self, pt1, vec1, pt3):
        """Set the points from a starting point/tangent
        @param pt1 - starting point
        @param vec1 - starting tangent
        @param pt3 - ending point"""
        # v = - 2 * p0 + 2 * p1
        # v/2 + p2 = p1
        mid_pt = np.array(pt1) + np.array(vec1) * 0.5
        self.set_pts(pt1, mid_pt, pt3)

    def set_radii(self, start_radius=1.0, end_radius=1.0):
        """ Set the radius of the branch
        @param start_radius - radius at pt1
        @param end_radius - radius at pt3"""
        self.start_radii = start_radius
        self.end_radii = end_radius

    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2 or 3d point"""
        pts_axis = np.array([self.pt1[i] * (1-t) ** 2 + 2 * (1-t) * t * self.pt2[i] + t ** 2 * self.pt3[i] for i in range(0, 3)])
        return pts_axis.transpose()
        # return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2

    def tangent_axis(self, t):
        """ Return the tangent vec
        @param t in 0, 1
        @return 3d vec"""
        vec_axis = [2 * t * (self.pt1[i] - 2.0 * self.pt2[i] + self.pt3[i]) - 2 * self.pt1[i] + 2 * self.pt2[i] for i in range(0, 3)]
        return np.array(vec_axis)

    def binormal_axis(self, t):
        """ Return the bi-normal vec, cross product of first and second derivative
        @param t in 0, 1
        @return 3d vec"""
        vec_tang = self.tangent_axis(t)
        vec_tang = vec_tang / np.linalg.norm(vec_tang)
        vec_second_deriv = np.array([2 * (self.pt1[i] - 2.0 * self.pt2[i] + self.pt3[i]) for i in range(0, 3)])

        vec_binormal = np.cross(vec_tang, vec_second_deriv)
        if np.isclose(np.linalg.norm(vec_second_deriv), 0.0):
            for i in range(0, 2):
                if not np.isclose(vec_tang[i], 0.0):
                    vec_binormal[i] = -vec_tang[(i + 1) % 3]
                    vec_binormal[(i + 1) % 3] = vec_tang[i]
                    vec_binormal[(i + 2) % 3] = 0.0
                    break

        return vec_binormal / np.linalg.norm(vec_binormal)

    def frenet_frame(self, t):
        """ Return the matrix that will take the point 0,0,0 to crv(t) with x axis along tangent, y along binormal
        @param t - t value
        @return 4x4 transformation matrix"""
        pt_center = self.pt_axis(t)
        vec_tang = self.tangent_axis(t)
        vec_tang = vec_tang / np.linalg.norm(vec_tang)
        vec_binormal = self.binormal_axis(t)
        vec_x = np.cross(vec_tang, vec_binormal)

        mat = np.identity(4)
        mat[0:3, 3] = pt_center[0:3]
        mat[0:3, 0] = vec_x.transpose()
        mat[0:3, 1] = vec_binormal.transpose()
        mat[0:3, 2] = vec_tang.transpose()

        return mat

    def _calc_radii(self):
        """ Calculate the radii along the branch
        @return a numpy array of radii"""
        radii = np.linspace(self.start_radii, self.end_radii, self.n_along)
        return radii

    def _calc_cyl_vertices(self):
        """Calculate the cylinder vertices"""
        pt = np.ones(shape=(4,))
        radii = self._calc_radii()

        for it, t in enumerate(np.linspace(0, 1.0, self.n_along)):
            mat = self.frenet_frame(t)
            pt[0] = 0
            pt[1] = 0
            pt[2] = 0
            for itheta, theta in enumerate(np.linspace(0, np.pi * 2.0, self.n_around, endpoint=False)):
                pt[0] = np.cos(theta) * radii[it]
                pt[1] = np.sin(theta) * radii[it]
                pt[2] = 0
                pt_on_crv = mat @ pt

                self.vertex_locs[it, itheta, :] = pt_on_crv[0:3].transpose()

    def make_mesh(self):
        """ Make a 3D generalized cylinder """
        self._calc_cyl_vertices()

    def write_mesh(self, fname):
        """Write out an obj file with the appropriate geometry
        @param fname - file name (should end in .obj"""
        with open(fname, "w") as fp:
            fp.write(f"# Branch\n")
            for it in range(0, self.n_along):
                for ir in range(0, self.n_around):
                    fp.write(f"v ")
                    fp.write(" ".join(["{:.6}"] * 3).format(*self.vertex_locs[it, ir, :]))
                    fp.write(f"\n")
            for it in range(0, self.n_along - 1):
                i_curr = it * self.n_around + 1
                i_next = (it+1) * self.n_around + 1
                for ir in range(0, self.n_around):
                    ir_next = (ir + 1) % self.n_around
                    fp.write(f"f {i_curr + ir} {i_next + ir_next} {i_curr + ir_next} \n")
                    fp.write(f"f {i_curr + ir} {i_next + ir} {i_next + ir_next} \n")

    def write_json(self, fname):
        """ Write out to json - does NOT write out the vertices for the mesh, use write_mesh for that
        @param fname - file name"""
        fix_nparray = []
        for k, v in self.__dict__.items():
            if k == "vertex_locs":
                continue
            try:
                if v.size == 3:
                    fix_nparray.append([k, v])
                    setattr(self, k, [float(x) for x in v])
            except AttributeError:
                pass

        with open(fname, "w") as f:
            dump(self.__dict__, f, indent=2)

        for fix in fix_nparray:
            setattr(self, fix[0], fix[1])

    @staticmethod
    def read_json(fname, bezier_crv=None, compute_mesh=True):
        """ Read back in from json file
        @param fname file name to read from
        @param bezier_crv - an existing bezier curve to put the data in
        @param compute_mesh - computer the mesh again, y/n
        @return the bezier curve"""
        with open(fname, 'r') as f:
            my_data = load(f)
            if not bezier_crv:
                bezier_crv = BezierCyl3D()
            for k, v in my_data.items():
                try:
                    if len(v) == 3:
                        setattr(bezier_crv, k, np.array(v))
                    else:
                        setattr(bezier_crv, k, v)
                except TypeError:
                    setattr(bezier_crv, k, v)
        bezier_crv.set_dims(bezier_crv.n_along, bezier_crv.n_around)
        if compute_mesh:
            bezier_crv.make_mesh()
        return bezier_crv


if __name__ == '__main__':
    import os

    __here__ = os.path.dirname(__file__)

    if not os.path.exists(f"{__here__}/data/DebugImages"):
        os.mkdir(f"{__here__}/data/DebugImages")
    branch = BezierCyl3D([506.5, 156.0, 0.0], [457.49999996771703, 478.9999900052037, 0.0], [521.5, 318.0, 0.0],
                         start_radius=10.5, end_radius=8.25)
    branch.make_mesh()
    branch.write_mesh(f"{__here__}/data/DebugImages/check_3d_bezier1.obj")

    branch = BezierCyl3D([-0.5, 0.0, 0.0], [0.0, 0.1, 0.05], [0.5, 0.0, 0.0], start_radius=0.5, end_radius=0.25)
    branch.make_mesh()
    branch.write_mesh(f"{__here__}/data/DebugImages/check_3d_bezier2.obj")

    branch.set_dims(n_along=30, n_radial=32)
    branch.make_mesh()
    branch.write_mesh(f"{__here__}/data/DebugImages/check_3d_bezier_more_vs.obj")
