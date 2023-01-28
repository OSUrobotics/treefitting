#!/usr/bin/env python3

import numpy as np
from json import load

class MakeTreeGeometry:
    _profiles = None
    _type_names = {"trunk", "sidebranch", "branch"}

    def __init__(self, data_dir):
        """ Read in profiles, if not already read in"""

        # Read in global parameters
        MakeTreeGeometry._read_profiles(data_dir)

        # Set per-branch/trunk params
        self.n_along = 10
        self.n_around = 64

        # Information about the current branch/trunk
        self.pt1 = [-0.5, 0.0, 0.0]
        self.pt2 = [0.0, 0.0, 0.0]
        self.pt3 = [0.5, 0.0, 0.0]
        self.start_radii = 0.5
        self.end_radii = 0.25
        self.start_is_junction = True

        self.vertex_locs = np.zeros((self.n_along, self.n_around, 3))


    @staticmethod
    def _read_profiles(data_dir):
        """ Read in all the profile curves for the various branch types"""
        if MakeTreeGeometry._profiles is not None:
            return

        MakeTreeGeometry._profiles = {}
        for t in MakeTreeGeometry._type_names:
            try:
                fname = data_dir + "/" + t + "_profiles.json"
                with open(fname, "r") as fp:
                    MakeTreeGeometry._profiles[t] = load(fp)
            except:
                pass

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
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3

    def set_pts_from_pt_tangent(self, pt1, vec1, pt3):
        """Set the points from a starting point/tangent
        @param pt1 - starting point
        @param vec1 - starting tangent
        @param pt3 - ending point"""
        # v = - 2 * p0 + 2 * p1
        # v/2 + p2 = p1
        mid_pt = np.array(pt1) + np.array(vec1) * 0.5
        self.set_pts(pt1, mid_pt, pt3)

    def set_radii(self, start_radius=1.0, end_radius=1.0, b_start_is_junction=False):
        """ Set the radius of the branch
        @param start_radius - radius at pt1
        @param end_radius - radius at pt3
        @param b_start_is_junction - is the start of the curve a junction? """
        self.start_radii = start_radius
        self.end_radii = end_radius
        self.start_is_junction = b_start_is_junction

    def pt_axis(self, t):
        """ Return a point along the bezier
        @param t in 0, 1
        @return 2 or 3d point"""
        pts_axis = np.array([self.pt1[i] * (1-t) ** 2 + 2 * (1-t) * t * self.pt2[i] + t ** 2 * self.pt3[i] for i in range(0, 3)])
        return pts_axis.transpose()
        #return self.p0 * (1-t) ** 2 + 2 * (1-t) * t * self.p1 + t ** 2 * self.p2

    def tangent_axis(self, t):
        """ Return the tangent vec
        @param t in 0, 1
        @return 3d vec"""
        vec_axis = [2 * t * (self.pt1[i] - 2.0 * self.pt2[i] + self.pt3[i]) - 2 * self.pt1[i] + 2 * self.pt2[i] for i in range(0, 3)]
        return  np.array(vec_axis)

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
                    vec_binormal[i] = -vec_tang[(i+1)%3]
                    vec_binormal[(i+1)%3] = vec_tang[i]
                    vec_binormal[(i+2)%3] = 0.0
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
        mat[0:3, 3] = -pt_center[0:3]
        mat[0:3, 0] = vec_x.transpose()
        mat[0:3, 1] = vec_binormal.transpose()
        mat[0:3, 2] = vec_tang.transpose()

        return mat

    def _calc_cyl_vertices(self):
        """Calculate the cylinder vertices"""
        pt = np.ones((4))
        radii = np.linspace(self.start_radii, self.end_radii, self.n_along)
        if self.start_is_junction:
            radii_exp = self.start_radii * 0.25 * np.exp(np.linspace(0, -10.0, self.n_along))
            radii = radii + radii_exp

        for it, t in enumerate(np.linspace(0, 1.0, self.n_along)):
            mat = self.frenet_frame(t)
            pt[0] = 0
            pt[1] = 0
            pt[2] = 0
            pt_on_crv = mat @ pt
            for itheta, theta in enumerate(np.linspace(0, np.pi * 2.0, self.n_around, endpoint=False)):
                pt[0] = np.cos(theta) * radii[it]
                pt[1] = np.sin(theta) * radii[it]
                pt[2] = 0
                pt_on_crv = mat @ pt

                self.vertex_locs[it, itheta, :] = pt_on_crv[0:3].transpose()

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
                    fp.write(f"f {i_curr + ir} {i_curr + ir_next} {i_next + ir_next} \n")
                    fp.write(f"f {i_curr + ir} {i_next + ir_next} {i_next + ir} \n")

    def _make_cyl(self, radius_start, radius_end, start_is_junction, profiles):
        """ Make a 3D generalized cylinder"""
        self.set_radii(start_radius=radius_start, end_radius=radius_end, b_start_is_junction=start_is_junction)
        self._calc_cyl_vertices()

    def make_branch_segment(self, pt1, pt2, pt3, radius_start, radius_end, start_is_junction):
        """ Output a 3D generalized cylinder"""
        self.set_pts(pt1, pt2, pt3)
        try:
            self._make_cyl(radius_start, radius_end, start_is_junction, MakeTreeGeometry._profiles["sidebranches"])
        except KeyError:
            self._make_cyl(radius_start, radius_end, start_is_junction, None)


if __name__ == '__main__':
    branch = MakeTreeGeometry("data")
    branch.make_branch_segment([-0.5, 0.0, 0.0], [0.0, 0.1, 0.05], [0.5, 0.0, 0.0], 0.5, 0.25, True)
    branch.write_mesh("data/cyl.obj")


