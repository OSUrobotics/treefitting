#!/usr/bin/env python3

import numpy as np
from list_read_write import ReadWrite
from scipy.optimize import fmin
from scipy.spatial.transform import Rotation as R
from test_pts import radius_and_height


class PtsForFit(ReadWrite):
    def __init__(self):
        super(PtsForFit, self). __init__("PTSFORFIT")

        self.pt_center = np.array([0, 0, 0])
        self.id = -1
        self.pts = []
        self.pts_ids = []
        self.pca_vecs = np.identity(3)
        self.pca_vals = [1, 1, 1]
        self.pca_err = 1e6

    def set_fit_pts(self, in_pt_id, ids, all_pts):
        self.id = in_pt_id
        self.pts = np.zeros([len(ids), 3])
        self.pts_ids = []
        for index, pt_id in enumerate(ids):
            self.pts_ids.append(pt_id)
            self.pts[index] = all_pts[pt_id]

    @staticmethod
    def err_circle(params, radius_sq, pts_2d):
        err = 0.0
        for p in pts_2d:
            err += abs(np.power(p[0] - params[0], 2) + np.power(p[1] - params[1], 2) - radius_sq)
        return err / len(pts_2d)

    @staticmethod
    def calc_2d_circle_err(pt_center, radius, pts_2d):
        radius_sq = np.power(radius, 2)
        return PtsForFit.err_circle(pt_center, radius_sq, pts_2d)

    @staticmethod
    def fit_circle(pts_2d, b_ret_err=False):
        A = np.ones([len(pts_2d)+1, 4])
        b = np.zeros([len(pts_2d)+1, 1])
        b[len(pts_2d), 0] = 1
        for index, p in enumerate(pts_2d):
            A[index, 0] = p[0] * p[0] + p[1] * p[1]
            A[index, 1] = p[0]
            A[index, 2] = p[1]
            A[index, 3] = 1

        x, res, rank, _ = np.linalg.lstsq(A, b, rcond=-1)
        if rank < 3:
            return [[0, 0], 1, 1e30]

        a = x[0, 0]
        b1 = (x[1, 0] / a) / 2.0
        b2 = (x[2, 0] / a) / 2.0
        c = x[3, 0] / a
        r = c - b1 * b1 - b2 * b2

        pt_center = [-b1, -b2]
        radius = np.sqrt(-r)

        err = res
        if b_ret_err is True:
            err = PtsForFit.calc_2d_circle_err(pt_center, radius, pts_2d)

        return pt_center, radius, err

    @staticmethod
    def fit_circle_center(pts_2d, radius, b_ret_err=False):
        radius_sq = np.power(radius, 2)
        pt_center = fmin(PtsForFit.err_circle, np.array([0, 0]), args=(radius_sq, pts_2d), disp=False)

        err = 0.0
        if b_ret_err is True:
            err = PtsForFit.calc_2d_circle_err(pt_center, radius, pts_2d)

        return pt_center, radius, err

    def pca_ratio(self):
        return self.pca_vals[2] / self.pca_vals[1]

    def pca_second_ratio(self):
        return self.pca_vals[1] / self.pca_vals[0]

    def score_pca(self, pca_ratio=7, pca_min=1.6, pca_max=40):
        """
        Score pca ratios by expected values
        :param pca_ratio: Desired ratio
        :param pca_min: Minimum allowed ratio
        :param pca_max: Maximum allowed ratio
        :return: How much pca score varies from target
        """
        score = 0
        ratio = self.pca_ratio()
        if ratio < pca_ratio:
            score += np.power((ratio - pca_min) / (pca_ratio - pca_min), 2)
        else:
            score += np.power((ratio - pca_ratio) / (pca_max - pca_ratio), 2)

        return score

    def fit_pca(self):
        """
        PCA fit to the points
        :return: pca score
        """
        if len(self.pts) < 4:
            self.pca_vals = [100.0, 1.0, 1.0]
            return 10000.0

        self.pt_center = [np.mean(self.pts[:, index]) for index in range(0, 3)]
        shift_data = np.array(self.pts) - self.pt_center
        V = np.cov(np.transpose(shift_data))
        values, vectors = np.linalg.eigh(V)

        """
        m = np.mean(shift_data, axis=0)
        s = np.std(shift_data, axis=0)
        for i in range(0, 3):
            print("{0} min {1:.4f} max {2:.4f} mean {3:.4f} std {4:.4f}".format(i,
                                                                               min(shift_data[:, i]),
                                                                               max(shift_data[:, i]), m[i], s[i]))
        """

        self.pca_vecs = []
        self.pca_vals = []
        for val, vec in sorted(zip(values, vectors)):
            self.pca_vecs.append(vec)
            self.pca_vals.append(val)

        self.pca_err = self.score_pca()
        return self.pca_err

    # reconstruct axis from pt center and orientation
    @staticmethod
    def rotate_axis(euler_angs, ref_vecs, order="xyz"):
        """
         Rotate axes by euler angs in the given order
        :param euler_angs: One to three Euler angles
        :param ref_vecs: The coordinate frame to rotate
        :param order: Which axes to rotate around. Number of letters should match euler_angs
        :return:
        """
        rot_matrix = R.from_euler(order, euler_angs, degrees=False)

        try:
            ref_vecs = rot_matrix.as_dcm() @ ref_vecs
        except AttributeError:
            ref_vecs = rot_matrix.as_matrix() @ ref_vecs

        return ref_vecs

    @staticmethod
    def move_center_pt(move_xy, pt_c, ref_vecs):
        """
         Move the center pt in 3D using the two last vecs in ref_vecs
        :param move_xy: Move in x and y
        :param pt_c: Original point
        :param ref_vecs: Coordinate system z, x, y
        :return: Moved center point
        """
        x_add = ref_vecs[0] * move_xy[0]
        y_add = ref_vecs[1] * move_xy[1]
        return pt_c + x_add + y_add

    def proj_line_seg(self, pt_3d, axis_vec):
        """
        Project the point onto the vec
        :param pt_3d: Point to project
        :param axis_vec: Vector to project ontol
        :return: Distane along vec and distance to vec
        """
        t_on_line = np.dot(pt_3d - self.pt_center, axis_vec)
        pt_on_line = self.pt_center + axis_vec * t_on_line
        vec_to_line = pt_3d - pt_on_line
        #  check = np.dot(vec_to_line, self.axis_vec)
        #  if abs(check) > 0.0001:
        #      raise ValueError("Bad project line segment {0}".format(check))

        dist_to_line = np.linalg.norm(vec_to_line)
        return t_on_line, dist_to_line

    @staticmethod
    def check_vectors(vecs):
        """Check that the three vectors (represented as a matrix) are orthogonal"""
        should_be_identity = np.allclose(vecs.dot(vecs.T), np.identity(3, np.float))
        should_be_one = np.allclose(np.linalg.det(vecs), 1)

        if should_be_identity is False:
            print("Matrix not rotation: identity check failed")
            print(vecs)

        if should_be_one is False:
            print("Matrix not rotation: 1 determinate check failed")
            print(vecs)

        return should_be_identity and should_be_one

    def __str__(self):
        if hasattr(self, "id"):
            ret_str = "Id {0}, size {1}, center [".format(self.id, len(self.pts))
            for x in self.pt_center:
                ret_str = ret_str + "{0:.2f} ".format(x)
            ret_str = ret_str + "] axis ["
            for x in self.pca_vecs[0]:
                ret_str = ret_str + "{0:.2f} ".format(x)
        else:
            ret_str = "Empty"

        str1 = "  PCA ratios {0:.2f}, ".format(self.pca_ratio())
        str2 = "{0:.2f}\n".format(self.pca_second_ratio())
        ret_str = ret_str + str1 + str2

        return ret_str

    def read(self, fid, all_pts=None, b_check_header=True):
        if b_check_header:
            self.check_header(fid)

        b_in_pts = False
        pt_index = 0
        n_pts = 0
        b_found_footer = False
        for l in fid:
            if self.check_footer(l, b_assert=False):
                b_found_footer = True
                break

            if b_in_pts is False:
                method_name, vals = self.get_class_member(l)
                if method_name == "pts":
                    n_pts = int(vals[0])
                    self.pts = np.zeros([n_pts, 3])
                    b_in_pts = True
                elif len(vals) == 1:
                    setattr(self, method_name, vals[0])
                elif method_name == "pts_ids":
                    setattr(self, method_name, vals)
                elif len(vals) == 3:
                    val_as_ndarray = np.array(vals)
                    setattr(self, method_name, val_as_ndarray)
                else:
                    raise ValueError("Unknown Cylinder read {0} {1}".format(method_name, vals))
            else:
                vals = self.get_vals_only(l)
                self.pts[pt_index] = vals
                pt_index += 1
                if pt_index == n_pts:
                    b_in_pts = False

        if b_found_footer is False:
            raise ValueError("Bad Pts for Fit end read")

        if len(self.pts) == 0 and len(self.pts_ids) > 0:
            try:
                self.pts = np.ndarray([len(self.pts_ids), 3])
                for index, pt_id in enumerate(self.pts_ids):
                    self.pts[index] = all_pts[pt_id]
            except TypeError:
                pass

    def write(self, fid, write_pts=False):
        self.write_header(fid)
        self.write_class_members(fid, dir(self), PtsForFit, ["pts"])
        if write_pts is True:
            fid.write("pts {0}\n".format(len(self.pts_ids)))
            for p in self.pts:
                fid.write("{0}\n".format(p))
        self.write_footer(fid)

    # Debugging/self-check routines
    @staticmethod
    def make_circle(in_radius, in_theta=1.6, x_noise=0.00001, y_noise=0.000001):
        """
        Random half circle centered around (0, 0), theta -pi/2 to pi/2, noise in x and y dirs
        Uses uniform noise
        :param in_radius: radius to use
        :param in_theta: maximum theta to sample
        :param x_noise: Noise in x
        :param y_noise: Noise in y
        :return: radius and points
        """

        pts_2d = []
        for t in np.linspace(-in_theta, in_theta, 30):
            x = in_radius * np.cos(t) + np.random.uniform(low=-x_noise/2, high=x_noise/2)
            y = in_radius * np.sin(t) + np.random.uniform(low=-y_noise/2, high=y_noise/2)
            pts_2d.append([x, y, 0])

        return pts_2d

    @staticmethod
    def make_cylinder(in_rad_min=0.015, in_rad_max=0.09, in_height=0.28, in_theta=1.6,
                      offset=0.1, x_noise=0.00001, y_noise=0.000001,
                      b_random_height=False):
        """
        Random half circle centered around (1, -2, 0) +- offset, theta -pi/2 to pi/2, noise in x and y dirs
        Uses uniform noise
        :param in_rad_min: radius range minimum
        :param in_rad_max: radius range maximum
        :param in_height: height of cylinder
        :param in_theta: maximum theta to sample
        :param offset: amount to vary center/height by
        :param x_noise: Noise in x
        :param y_noise: Noise in y
        :param b_random_height: Use a random height rather than a structured set of rings
        :return:
        """
        off = offset / 2
        radius = np.random.uniform(low=in_rad_min, high=in_rad_max)
        xc = -1 + np.random.uniform(low=-off, high=off)
        yc = +2 + np.random.uniform(low=-off, high=off)
        zc = +0 + np.random.uniform(low=-off, high=off)
        h = in_height + np.random.uniform(low=-off/10, high=off/10)

        pts_ret = PtsForFit()
        pts_ret.pt_center = np.array([xc, yc, zc])
        pts_ret.pts = np.ndarray([30*10, 3])
        pts_ret.pca_vecs[0] = [1, 0, 0]
        pts_ret.pca_vecs[1] = [0, 1, 0]
        pts_ret.pca_vecs[2] = [0, 0, 1]
        i_count = 0
        for n_rings in range(0, 10):
            pts_2d = PtsForFit.make_circle(radius, in_theta, x_noise, y_noise)

            for p in pts_2d:
                if b_random_height:
                    z = zc + np.random.uniform(-h/2, h/2)
                else:
                    z = zc - h/2 + (n_rings/9.0) * (h) + np.random.uniform(low=-y_noise/2, high=y_noise/2)

                pts_ret.pts[i_count] = np.array([xc + p[0], yc + p[1], z])
                i_count += 1

        return pts_ret, radius, h

    def check_fit_circle(self, in_radius, eps=1e-4):
        """ Check the various fit equations """

        pt_center, radius, fit_radius_2d_err = self.fit_circle(self.pts[:, 0:2], b_ret_err=True)
        xerr = pt_center[0] - self.pt_center[0]
        yerr = pt_center[1] - self.pt_center[1]
        rerr = radius - in_radius

        if abs(xerr) > eps or abs(yerr) > eps or abs(rerr) > eps or fit_radius_2d_err > eps:
            raise ValueError("Bad circle fit {0} {1} {2} {3}".format(xerr, yerr, rerr, fit_radius_2d_err))

        # Center the points and then see if routine can find the correct center
        pts_center = np.array([np.mean(self.pts[:, index]) for index in range(0, 3)])
        pts_centered = self.pts[:, 0:1] - pts_center
        pt_center_ret, rad, err = self.fit_circle_center(pts_centered, in_radius, b_ret_err=True)
        x_c_err = pt_center_ret[0] - self.pt_center[0]
        y_c_err = pt_center_ret[1] - self.pt_center[1]

        if abs(x_c_err) > 100*eps or abs(y_c_err) > 2*eps or abs(err) > eps:
            print("Bad circle fit center {0} {1} {2}".format(x_c_err, y_c_err, err))

    def check_pca(self, eps=1e-4):
        fit_self = PtsForFit()
        fit_self.pts = self.pts
        fit_self.fit_pca()

        pt_center_err = fit_self.pt_center - self.pt_center
        yerr = pt_center_err[1]
        zerr = pt_center_err[2]
        z_ang_err = np.arccos(abs(np.dot(fit_self.pca_vecs[0], [0, 0, 1])))

        # Only y and z should be correct, along with axis in z direction
        if abs(yerr) > eps or abs(zerr) > eps or abs(z_ang_err) > np.pi/6:
            print("Bad circle fit pca {0} {1} {2}".format(yerr, zerr, z_ang_err))
            fit_self.fit_pca()

    @staticmethod
    def check_circle_fits(in_radius_min, in_radius_max):
        pts_check = PtsForFit()

        for theta in np.linspace(0.8 * np.pi/2, np.pi/2, 3):
            for noise in [0.00001, 0.0001, 0.001]:
                for r in np.linspace(in_radius_min, in_radius_max, 3):
                    print("Checking theta {0:0.3f} radius {1:0.2f} noise {2:0.6f}".format(theta, r, noise))
                    pts_check.pts = np.array(PtsForFit.make_circle(r, theta, noise * 2, noise))
                    pts_check.check_fit_circle(r, noise * 20)

    @staticmethod
    def get_pca_ratio_cylinder(in_radius_min, in_radius_max, in_height):
        pca_ratio = []
        pca_second_ratio = []
        for off in np.linspace(0, in_radius_min, 3):
            for theta in np.linspace(0.8 * np.pi/2, np.pi/2, 3):
                for noise in [0.00001, 0.0001, 0.001]:
                    cyl, _, _ = PtsForFit.make_cylinder(in_radius_min, in_radius_max, in_height, theta,
                                                        off, noise * 2, noise, b_random_height=True)

                    cyl.fit_pca()
                    pca_ratio.append(cyl.pca_ratio())
                    pca_second_ratio.append(cyl.pca_second_ratio())

        print("For radius {0:.2f}, {1:.2f}  height {2:.2f}".format(in_radius_min, in_radius_max, in_height))
        print("pca ratio min {0:.2f} max {1:.2f}, avg {2:.2f}".format(np.min(pca_ratio), np.max(pca_ratio), np.mean(pca_ratio)))
        print("pca second ratio min {0:.2f} max {1:.2f}, avg {2:.2f}".format(np.min(pca_second_ratio), np.max(pca_second_ratio), np.mean(pca_second_ratio)))


if __name__ == '__main__':
    radii = radius_and_height()

    for i in range(0, 10):
        PtsForFit.check_circle_fits(radii["radius_min"], radii["radius_max"])

    PtsForFit.get_pca_ratio_cylinder(radii["radius_min"], radii["radius_max"], 4 * radii["radius_min"])
    PtsForFit.get_pca_ratio_cylinder(radii["radius_min"], radii["radius_max"], 4 * radii["radius_max"])
