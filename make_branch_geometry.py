#!/usr/bin/env python3

import numpy as np
from json import load

import scipy.spatial.transform
from scipy.spatial.transform import Rotation as R

class MakeTreeGeometry:
    _profiles = None
    _type_names = {"trunk", "sidebranch", "branch"}
    _bud_shape = None

    def __init__(self, data_dir):
        """ Read in profiles, if not already read in"""

        # Read in global parameters
        MakeTreeGeometry._read_profiles(data_dir)
        MakeTreeGeometry._make_bud_shape()

        # Set per-branch/trunk params
        self.n_along = 10
        self.n_around = 64

        # Information about the current branch/trunk
        self.pt1 = [-0.5, 0.0, 0.0]
        self.pt2 = [0.0, 0.0, 0.0]
        self.pt3 = [0.5, 0.0, 0.0]
        self.start_radii = 0.5
        self.end_radii = 0.25
        self.start_is_junction = False
        self.end_is_bud = False
        self.start_bud = 0.7
        self.bud_angle = 0.8 * np.pi / 2
        self.bud_length = 0.1

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

    @staticmethod
    def _make_bud_shape():
        if MakeTreeGeometry._bud_shape is None:
            n_pts = 10
            MakeTreeGeometry._bud_shape = np.zeros((2, n_pts))
            MakeTreeGeometry._bud_shape[0, :] = np.linspace(0, 1.0, n_pts)
            MakeTreeGeometry._bud_shape[0, -2] = 0.5 * MakeTreeGeometry._bud_shape[0, -2] + 0.5 * MakeTreeGeometry._bud_shape[0, -1]
            MakeTreeGeometry._bud_shape[1, 0] = 1.0
            MakeTreeGeometry._bud_shape[1, 1] = 0.95
            MakeTreeGeometry._bud_shape[1, 2] = 1.05
            MakeTreeGeometry._bud_shape[1, 3] = 1.1
            MakeTreeGeometry._bud_shape[1, 4] = 1.05
            MakeTreeGeometry._bud_shape[1, 5] = 0.8
            MakeTreeGeometry._bud_shape[1, 6] = 0.7
            MakeTreeGeometry._bud_shape[1, 7] = 0.5
            MakeTreeGeometry._bud_shape[1, 8] = 0.3
            MakeTreeGeometry._bud_shape[1, 9] = 0.0

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

    def set_radii(self, start_radius=1.0, end_radius=1.0, b_start_is_junction=False, b_end_is_bud=False):
        """ Set the radius of the branch
        @param start_radius - radius at pt1
        @param end_radius - radius at pt3
        @param b_start_is_junction - is the start of the curve a junction?
        @param b_end_is_bud - is the end a bud? """
        self.start_radii = start_radius
        self.end_radii = end_radius
        self.start_is_junction = b_start_is_junction
        self.end_is_bud = b_end_is_bud

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
        mat[0:3, 3] = pt_center[0:3]
        mat[0:3, 0] = vec_x.transpose()
        mat[0:3, 1] = vec_binormal.transpose()
        mat[0:3, 2] = vec_tang.transpose()

        return mat

    def _calc_radii(self):
        """ Calculate the radii along the branch
        @return a numpy array of radii"""
        radii = np.linspace(self.start_radii, self.end_radii, self.n_along)
        if self.start_is_junction:
            radii_exp = self.start_radii * 0.25 * np.exp(np.linspace(0, -10.0, self.n_along))
            radii = radii + radii_exp

        if self.end_is_bud:
            i_start = int(np.floor(self.start_bud * self.n_along))
            i_total = self.n_along - i_start
            radii_bud = np.interp(np.linspace(0, 1, i_total), MakeTreeGeometry._bud_shape[0, :], MakeTreeGeometry._bud_shape[1, :])
            radii[i_start:] *= radii_bud
        return radii

    def _calc_cyl_vertices(self):
        """Calculate the cylinder vertices"""
        pt = np.ones((4))
        radii = self._calc_radii()

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
                    fp.write(f"f {i_curr + ir} {i_next + ir_next} {i_curr + ir_next} \n")
                    fp.write(f"f {i_curr + ir} {i_next + ir} {i_next + ir_next} \n")

    def _make_cyl(self, profiles):
        """ Make a 3D generalized cylinder
        @param profiles - variations to the radii """
        self._calc_cyl_vertices()

    def make_branch_segment(self, pt1, pt2, pt3, radius_start, radius_end, start_is_junction, end_is_bud):
        """ Output a 3D generalized cylinder"""
        self.set_pts(pt1, pt2, pt3)
        self.set_radii(start_radius=radius_start, end_radius=radius_end, b_start_is_junction=start_is_junction, b_end_is_bud=end_is_bud)
        try:
            self._make_cyl(MakeTreeGeometry._profiles["sidebranches"])
        except KeyError:
            self._make_cyl(None)

    def place_buds(self, locs):
        """ Position and orientation of buds,
        @param locs - t along, radius loc tuples in a list
        @
        @return [(pt1, pt2, pt3) """

        ts = np.linspace(0, 1, self.n_along)
        radii = self._calc_radii()

        pt = np.ones((4))
        zero_pt = np.ones((4))
        zero_pt[0:3] = 0.0
        vec = np.zeros((4))
        ret_list = []
        for loc in locs:
            mat = self.frenet_frame(loc[0])
            r = np.interp(loc[0], ts, radii)
            pt_on_crv = mat @ zero_pt
            pt[0] = np.cos(loc[1]) * r
            pt[1] = np.sin(loc[1]) * r
            pt[2] = 0
            pt_on_srf = mat @ pt
            vec[0] = np.cos(loc[1])
            vec[1] = np.sin(loc[1])
            vec[2] = 0
            vec_rotate = np.cross(vec[0:3], np.array([0, 0, 1]))
            vec_rotate = vec_rotate / np.linalg.norm(vec_rotate)
            mat_rotate_bud = R.from_rotvec(self.bud_angle * vec_rotate)
            # Note - newer versions use as_matrix
            mat_rot = mat_rotate_bud.as_dcm()
            vec[0:3] = mat_rot @ vec[0:3]
            vec_on_crv = mat @ vec
            vec_on_crv = vec_on_crv * (self.bud_length / np.linalg.norm(vec_on_crv))
            pt_end_bud = pt_on_srf + vec_on_crv
            pt_mid = 0.7 * pt_on_srf + 0.3 * pt_on_crv
            ret_list.append((pt_mid[0:3], pt_on_srf[0:3], pt_end_bud[0:3]))

        return ret_list


if __name__ == '__main__':
    branch = MakeTreeGeometry("data")

    branch.make_branch_segment([-0.5, 0.0, 0.0], [0.0, 0.1, 0.05], [0.5, 0.0, 0.0], radius_start=0.5, radius_end=0.25,
                               start_is_junction=True, end_is_bud=False)
    branch.write_mesh("data/cyl.obj")

    branch.set_dims(n_along=30, n_radial=32)
    branch.make_branch_segment([-0.5, 0.0, 0.0], [0.0, 0.1, 0.05], [0.5, 0.0, 0.0], radius_start=0.1, radius_end=0.075,
                               start_is_junction=False, end_is_bud=True)
    branch.write_mesh("data/cyl_bud.obj")

    bud_loc = branch.place_buds(((0.2, 0), (0.3, np.pi/4), (0.4, 3.0 * np.pi/4)))
    bud = MakeTreeGeometry("data")
    bud.start_bud = 0.2
    for i, b in enumerate(bud_loc):
        bud.make_branch_segment(b[0], b[1], b[2], radius_start=0.025, radius_end=0.03, start_is_junction=False, end_is_bud=True)
        bud.write_mesh(f"data/bud_{i}.obj")


