#!/usr/bin/env python3

# a 3D Bezier branch/trunk
#  Inherits from 3D bezier curve
#     - The 3D bezier curve
#     - Start and end radii
#  Adds
#    - placing buds/bud geometry
#    - profile curve - making the outside "bumpy"
#
# Can:
#   - Evaluate points along the curve
#   - Make a cylinderical mesh
#   - Project self into an image

import numpy as np
from json import load
from bezier_cyl_3d import BezierCyl3D
from scipy.spatial.transform import Rotation as R


class BezierCyl3DWithDetail(BezierCyl3D):
    _profiles = None
    _type_names = {"trunk", "sidebranch", "branch"}
    _bud_shape = None
    _data_dir = "./data/"

    def __init__(self, bezier_crv=None):
        """ Initialize a 3D curve, built from a quadratic Bezier with radii
        @param bezier_crv - 3D bezier curve to start with"""

        if bezier_crv:
            super(BezierCyl3DWithDetail, self).__init__()
            bezier_crv.copy(bezier_crv=self, b_compute_mesh=True)
        else:
            super(BezierCyl3DWithDetail, self).__init__()

        self.start_is_junction = False
        self.end_is_bud = False
        self.start_bud = 0.7
        self.bud_angle = 0.8 * np.pi / 2
        self.bud_length = 0.1

        # Read in global data/profiles
        BezierCyl3DWithDetail._read_profiles(self._data_dir)
        BezierCyl3DWithDetail._make_bud_shape()

    @staticmethod
    def _read_profiles(data_dir):
        """ Read in all the profile curves for the various branch types"""
        if BezierCyl3DWithDetail._profiles is not None:
            return

        BezierCyl3DWithDetail._profiles = {}
        for t in BezierCyl3DWithDetail._type_names:
            try:
                fname = data_dir + "/" + t + "_profiles.json"
                with open(fname, "r") as fp:
                    BezierCyl3DWithDetail._profiles[t] = load(fp)
            except FileNotFoundError:
                pass

    @staticmethod
    def _make_bud_shape():
        if BezierCyl3DWithDetail._bud_shape is None:
            n_pts = 10
            BezierCyl3DWithDetail._bud_shape = np.zeros((2, n_pts))
            BezierCyl3DWithDetail._bud_shape[0, :] = np.linspace(0, 1.0, n_pts)
            BezierCyl3DWithDetail._bud_shape[0, -2] = 0.5 * BezierCyl3DWithDetail._bud_shape[0, -2] + 0.5 * BezierCyl3DWithDetail._bud_shape[0, -1]
            BezierCyl3DWithDetail._bud_shape[1, 0] = 1.0
            BezierCyl3DWithDetail._bud_shape[1, 1] = 0.95
            BezierCyl3DWithDetail._bud_shape[1, 2] = 1.05
            BezierCyl3DWithDetail._bud_shape[1, 3] = 1.1
            BezierCyl3DWithDetail._bud_shape[1, 4] = 1.05
            BezierCyl3DWithDetail._bud_shape[1, 5] = 0.8
            BezierCyl3DWithDetail._bud_shape[1, 6] = 0.7
            BezierCyl3DWithDetail._bud_shape[1, 7] = 0.5
            BezierCyl3DWithDetail._bud_shape[1, 8] = 0.3
            BezierCyl3DWithDetail._bud_shape[1, 9] = 0.0

    def _make_cyl(self, profiles):
        """ Make a 3D generalized cylinder
        @param profiles - variations to the radii """
        if profiles:
            self._calc_cyl_vertices()
        else:
            self._calc_cyl_vertices()

    def set_radii_and_junction(self, start_radius=1.0, end_radius=1.0, b_start_is_junction=False, b_end_is_bud=False):
        """ Set the radius of the branch
        @param start_radius - radius at pt1
        @param end_radius - radius at pt3
        @param b_start_is_junction - is the start of the curve a junction?
        @param b_end_is_bud - is the end a bud? """
        self.set_radii(start_radius, end_radius)
        self.start_is_junction = b_start_is_junction
        self.end_is_bud = b_end_is_bud

    def make_branch_segment(self, pt1, pt2, pt3, radius_start, radius_end, start_is_junction, end_is_bud):
        """ Output a 3D generalized cylinder"""
        self.set_pts(pt1, pt2, pt3)
        self.set_radii_and_junction(start_radius=radius_start, end_radius=radius_end, b_start_is_junction=start_is_junction, b_end_is_bud=end_is_bud)
        try:
            self._make_cyl(BezierCyl3DWithDetail._profiles["sidebranches"])
        except KeyError:
            self._make_cyl(None)

    def place_buds(self, locs):
        """ Position and orientation of buds,
        @param locs - t along, radius loc tuples in a list
        @
        @return [(pt1, pt2, pt3) """

        ts = np.linspace(0, 1, self.n_along)
        radii = self._calc_radii()

        pt = np.ones((4,))
        zero_pt = np.ones((4,))
        zero_pt[0:3] = 0.0
        vec = np.zeros((4,))
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

    branch = BezierCyl3DWithDetail()

    branch.set_pts([506.5, 156.0, 0.0], [457.49999996771703, 478.9999900052037, 0.0], [521.5, 318.0, 0.0])
    branch.set_radii_and_junction(start_radius=10.5, end_radius=8.25, b_start_is_junction=True, b_end_is_bud=False)
    branch.write_mesh("data/jos.obj")

    branch.set_pts([-0.5, 0.0, 0.0], [0.0, 0.1, 0.05], [0.5, 0.0, 0.0])
    branch.set_radii_and_junction(start_radius=0.5, end_radius=0.25, b_start_is_junction=True, b_end_is_bud=False)
    branch.write_mesh("data/cyl.obj")

    branch.set_dims(n_along=30, n_radial=32)
    branch.set_pts([-0.5, 0.0, 0.0], [0.0, 0.1, 0.05], [0.5, 0.0, 0.0])
    branch.set_radii_and_junction(start_radius=0.1, end_radius=0.075, b_start_is_junction=False, b_end_is_bud=True)
    branch.write_mesh("data/cyl_bud.obj")

    bud_loc = branch.place_buds(((0.2, 0), (0.3, np.pi/4), (0.4, 3.0 * np.pi/4)))
    bud = BezierCyl3DWithDetail()
    bud.start_bud = 0.2
    for i, b in enumerate(bud_loc):
        bud.set_pts(b[0], b[1], b[2])
        bud.make_branch_segment(b[0], b[1], b[2], radius_start=0.025, radius_end=0.03, start_is_junction=False, end_is_bud=True)
        bud.write_mesh(f"data/bud_{i}.obj")
