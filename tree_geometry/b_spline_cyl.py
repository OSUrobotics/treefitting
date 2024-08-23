#!/usr/bin/env python3
"""
b_spline_cyl.py
Author: Luke Strohbehn
"""
import numpy as np

from .b_spline_curve import BSplineCurve

class BSplineCyl2D(BSplineCurve):
    def __init__(self):
        super().__init__()

        # Drawing/mesh creation parameters
        self.n_along = 10
        self.n_around = 64
        self.vertex_locs = np.zeros((self.n_along, self.n_around, 3))

        # Radii
        self.r0: float = 0.0
        self.r1: float = 0.0
        self.r2: float = 0.0
        return

    def radius(self, t):
        """Return radius at a point t along the spline"""
        return

    def norm_axis(self, t: float, dir: int):
        """
        Get the normal vector to the curve at parameter t 
        :param t: parameter
        :param dir: direction of normal, 0 for left, 1 for right
        :return: normal vector
        """
        vec_tang = self.tangent_axis(t)
        vec_length = np.sqrt(vec_tang[0] * vec_tang[0] + vec_tang[1] * vec_tang[1])
        if dir == 0:
            return np.array([-vec_tang[1] / vec_length, vec_tang[0] / vec_length])
        elif dir == 1:
            return np.array([vec_tang[1] / vec_length, -vec_tang[0] / vec_length])
        else:
            raise ValueError("Invalid direction for normal")
    
    def edge_pts(self, t):
        """ 
        Return the left and right edge of the tube as points
        :param t: parameter
        :return: 2d pts, left and right edge
        """
        pt = self.pt_axis(t)
        vec = self.tangent_axis(t)
        vec_step = self.radius(t) * vec / np.sqrt(vec[0] * vec[0] + vec[1] * vec[1])
        left_pt = [pt[0] - vec_step[1], pt[1] + vec_step[0]]
        right_pt = [pt[0] + vec_step[1], pt[1] - vec_step[0]]
        return left_pt, right_pt

    def edge_offset_pt(self, t, perc_in_out, direction):
        """ Go in/out of the edge point a given percentage
        @param t - t value along the curve (in range 0, 1)
        @param perc_in_out - if 1, get point on edge. If 0.5, get halfway to centerline. If 2.0 get 2 width
        @param direction - 'Left' is the left direction, 'Right' is the right direction
        @return numpy array x,y """
        pt_edge = self.pt_axis(t)
        vec_norm = self.norm_axis(t, direction)
        return pt_edge + vec_norm * (perc_in_out * self.radius(t))

    def binormal_axis(self, t):
        """ Return the bi-normal vec, cross product of first and second derivative
        @param t in 0, 1
        @return 3d vec"""
        vec_tang = self.tangent_axis(t)
        vec_tang = vec_tang / np.linalg.norm(vec_tang)
        vec_second_deriv = np.array([2 * (self.pt0[i] - 2.0 * self.pt1[i] + self.pt2[i]) for i in range(0, 3)])

        vec_binormal = np.cross(vec_tang, vec_second_deriv)
        if np.isclose(np.linalg.norm(vec_second_deriv), 0.0):
            for i in range(0, 2):
                if not np.isclose(vec_tang[i], 0.0):
                    vec_binormal[i] = -vec_tang[(i+1)%3]
                    vec_binormal[(i+1)%3] = vec_tang[i]
                    vec_binormal[(i+2)%3] = 0.0
                    break

        return vec_binormal / np.linalg.norm(vec_binormal)
    
    def _calc_radii(self):
        """ Calculate the radii along the branch
        @return a numpy array of radii"""
        return np.linspace(self.start_radii, self.end_radii, self.n_along)
        
    def _calc_cyl_vertices(self) -> None:
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
        return

    def make_mesh(self):
        """ Make a 3D generalized cylinder """
        return self._calc_cyl_vertices()

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
        return