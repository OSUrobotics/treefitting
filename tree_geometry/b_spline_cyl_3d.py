#!/usr/bin/env python3
"""
b_spline_cyl_3d.py
Author: Luke Strohbehn
"""

import numpy as np
from typing import Union, Any
from tree_geometry.b_spline_cyl import BSplineCyl


# Code for creating 3d meshes from bspline cylinders
class BSplineCyl3d(BSplineCyl):
    def __init__(self, ctrl_pts: Union[list[np.ndarray], np.ndarray], degree: str = "cubic",
                 radii: Union[float, list[float]] = 1.0) -> None:
        """BSpline with radii initialization
        :param ctrl_pts: control points, list of numpy array points of desired dimension
        :param degree: degree of spline, defaults to "cubic"
        :param radii: radii, either a single radii value for the whole curve or a list of radii values
        """
        super().__init__(ctrl_pts=ctrl_pts, degree=degree, radii=radii)

        # For controlling mesh generation
        self.n_along_per = 4 # Per segment of curve
        self.n_around = 32
        self.vertex_locs = None

    def set_mesh_dimensions(self, n_along_per=6, n_around=32):
        """ How many mesh vertices to create
        @param n_along_per - number per segment of the spine curve (around 4-6 is good)
        @param n_around - number of points around the tube, 16-64 is good"""
        self.n_along_per = n_along_per
        self.n_around = n_around
        self.vertex_locs = None
        self.vertex_normals = None

    def frenet_frame(self, t : float):
        """ Return the matrix that will take the point 0,0,0 to crv(t) with x axis along tangent, y along binormal
        @param t - t value between 0 and max_t
        @return 4x4 transformation matrix"""

        pt_center = self.eval_crv(t)
        vec_tang = self.eval_deriv(t)
        vec_tang = vec_tang / np.linalg.norm(vec_tang)
        vec_binormal = self.eval_norm(t)
        vec_x = np.cross(vec_tang, vec_binormal)

        mat = np.identity(4)
        mat[0:3, 3] = pt_center[0:3]
        mat[0:3, 0] = vec_x.transpose()
        mat[0:3, 1] = vec_binormal.transpose()
        mat[0:3, 2] = vec_tang.transpose()

        return mat

    def make_mesh(self) -> None:
        """Calculate the cylinder vertices"""
        pt = np.ones(shape=(4,))
        norm_vec = np.zeros(shape=(4,))
        n_total = self.n_along_per * self.n_segments()

        self.vertex_locs = np.zeros((n_total, self.n_around, 3))
        self.vertex_normals = np.zeros((n_total, self.n_around, 3))

        # TODO fix frenet frame twist
        for it, t in enumerate(np.linspace(0, self.max_t(), n_total)):
            mat = self.frenet_frame(t)
            radii = self.radius(t)
            for itheta, theta in enumerate(np.linspace(0, np.pi * 2.0, self.n_around, endpoint=False)):
                pt[0] = np.cos(theta) * radii
                pt[1] = np.sin(theta) * radii
                pt[2] = 0.0
                pt_on_crv = mat @ pt

                norm_vec[0] = np.cos(theta)
                norm_vec[1] = np.sin(theta)
                norm_on_srf = mat @ norm_vec

                self.vertex_locs[it, itheta, :] = pt_on_crv[0:3].transpose()
                self.vertex_normals[it, itheta, :] = norm_on_srf[0:3].transpose()

    def write_json(self):
        """Create a dictionary and return it"""
        my_dict = {"Name": "BSplineCyl3d",
                   "n_along_per": self.n_along_per,
                   "n_around": self.n_around,
                   "crv": super().write_json()}

        return my_dict

    def write_mesh(self, fname):
        """Write out an obj file with the appropriate geometry
        Assumes make_mesh has been called
        @param fname - file name (should end in .obj"""

        if self.vertex_locs == None:
            self.make_mesh()

        with open(fname, "w") as fp:
            fp.write(f"# Branch\n")
            n_along = self.vertex_locs.shape[0]
            for it in range(0, n_along):
                for ir in range(0, self.n_around):
                    fp.write(f"v ")
                    fp.write(" ".join(["{:.6}"] * 3).format(*self.vertex_locs[it, ir, :]))
                    fp.write(f"\n")
            for it in range(0, n_along - 1):
                i_curr = it * self.n_around + 1
                i_next = (it + 1) * self.n_around + 1
                for ir in range(0, self.n_around):
                    ir_next = (ir + 1) % self.n_around
                    fp.write(f"f {i_curr + ir} {i_next + ir_next} {i_curr + ir_next} \n")
                    fp.write(f"f {i_curr + ir} {i_next + ir} {i_next + ir_next} \n")


if __name__ == "__main__":
    branch = BSplineCyl3d([[506.5, 156.0, 0.0], [457.49999996771703, 478.9999900052037, 0.0], [521.5, 318.0, 0.0]],
                          degree='quadratic', radii=[10.5, 8.25])

    branch.make_mesh()
    branch.write_mesh("check_3d_bezier1.obj")
