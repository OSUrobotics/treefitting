#!/usr/bin/env python3

import numpy as np
from scipy.optimize import fmin
# from test_pts import best_pts, bad_pts
from list_read_write import ReadWrite
from PtsForFit import PtsForFit
from pandas import DataFrame
from sklearn.cluster import KMeans

class BranchPt(ReadWrite):
    def __init__(self):
        super(BranchPt, self). __init__("BRANCHPT")

        self.data = PtsForFit()
        self.branch_vecs = []
        self.pts_in_branch = []
        self.branch_radius = []

        # For building branches - radius of sphere and where to clip
        self.sphere_radius = 0.01
        self.clip_radius = 0.015

        # From err_fit
        self.radius_err = 1e6
        self.percentage_in_err = 1e6
        self.percentage_not_branch_err = 1e6
        self.err = 1e6

        # Set in various optimization routines
        self.fit_branches_err = 1e6
        self.optimize_angs_err = 1e6

    def pt_center(self):
        return self.data.pt_center

    def pts(self):
        return self.data.pts

    def id(self):
        return self.data.id

    def set_fit_pts(self, in_pt_id, ids, all_pts):
        self.data.set_fit_pts(in_pt_id, ids, all_pts)

    def pca_ratio(self):
        return self.data.pca_ratio()

    def pca_second_ratio(self):
        return self.data.pca_second_ratio()

    def score_pca(self, pca_ratio=7, pca_min=1.5, pca_max=35):
        return self.data.score_pca(pca_ratio, pca_min, pca_max)

    def fit_pca(self, min_radius):
        """
        PCA fit to the points
        Sets axis/x y vecs
        :return: pca score
        """
        self.data.fit_pca()

        self.branch_vecs = []
        self.branch_radius = []
        self.pts_in_branch = []

        radii = []
        for p in self.data.pts:
            radii.append(np.linalg.norm(p - self.data.pt_center))
        self.sphere_radius = max(radii)
        self.clip_radius = max(min_radius, np.mean(radii) - np.std(radii))

        return self.data.pca_err

    def fit_branch(self, axis_vec, ids_covered, max_radius):
        """
        Project down into the plane and along the main vector, then fit a circle
        Assumes points that are "inside" the branch are already marked as covered
        Only adds points that are within max_radius of the axis

        :param: axis_vec The vector to try
        :param: ids_covered The ids that are already covered/too close
        :param: max_radius - don't use points outside of this radius
        :return: Returns the covered points, the radius of the fit, and the error of the fit
        """
        xy = []
        ids_in_branch = [False for i in range(0, len(self.data.pts))]
        for i, p in enumerate(self.data.pts):
            if ids_covered[i] is True:
                continue

            (t, d) = self.data.proj_line_seg(p, axis_vec)

            if d < max_radius:
                ids_in_branch[i] = True
                h = np.dot(d, axis_vec)
                vec_xy = d - self.axis_vec * t
                xy.append([np.dot(vec_xy, self.x_vec), np.dot(vec_xy, self.y_vec)])

        # Fit circle to projected points
        (_, radius_2d, fit_radius_2d_err) = self.data.fit_circle(xy, b_ret_err=True)

        return ids_in_branch, radius_2d, fit_radius_2d_err

    def fit_branches(self, min_radius, max_radius):
        ids_covered = []

        sphere = []
        for p in self.data.pts:
            vec = p - self.data.pt_center
            d = np.linalg.norm(vec)
            if d < self.clip_radius:
                ids_covered.append(True)
            else:
                ids_covered.append(False)
                vec = vec / d
                sphere.append(vec)

        sphere = np.array(sphere)
        sphere_centers = KMeans(n_clusters=4, random_state=0).fit(sphere)

        self.branch_vecs = []
        self.pts_in_branch = []
        self.branch_radius = []
        while np.count_nonzero(ids_covered) < len(self.data.pts):
            axis_vec = sphere_centers.clusters[0]
            axis_vec /= np.linalg.norm(axis_vec)

            ids_in_branch, radius_2d, fit_radius_2d_err = self.fit_branch(axis_vec, ids_covered, max_radius)
            if np.count_nonzero(ids_in_branch) < 10:
                continue
            if radius_2d < min_radius or radius_2d > max_radius:
                continue

            for i, v in enumerate(ids_in_branch):
                if v == True:
                    ids_covered[i] = True
            self.branch_vecs.append(axis_vec)
            self.pts_in_branch.append(ids_covered)
            self.branch_radius.append(radius_2d)

        if len(self.branch_vecs) < 3:
            return 1e6

        self.fit_branches_err = self.branch_pt_fit_err(min_radius, max_radius)
        return self.fit_branches_err

    @staticmethod
    def cyl_err(xs, cylinder, ref_vecs, radius_min, radius_max):
        rot_vecs = PtsForFit.rotate_axis(xs, ref_vecs, "xyz")

        err_proj = 0
        max_h = 0
        min_h = 0
        r_out = 0
        r_in = 0
        radius_span = radius_max - radius_min
        for p in cylinder.data.pts:
            (h, d) = cylinder.data.proj_line_seg(p, rot_vecs[2])
            err_proj += np.power((d - cylinder.radius) / radius_span, 2)
            if d < r_in:
                r_in = r_in + 1
            if d > r_out:
                r_out = r_out + 1

            max_h = max(max_h, h)
            min_h = min(min_h, h)

        err_proj = err_proj / len(cylinder.data.pts)
        proj_height = max_h - min_h
        err_h = 0
        if proj_height < cylinder.height:
            err_h = np.power(1.0 - proj_height / cylinder.height, 2)

        for x in xs:
            print("{0:0.4f} ".format(x), end='')
        print("{0:0.4f} {1:0.4f} r in {2} r out {2}".format(err_proj, err_h, r_in, r_out))
        return err_proj + err_h + r_in / len(cylinder.data.pts) + r_out / len(cylinder.data.pts)

    def branch_pt_fit_err(self, in_rad_min, in_rad_max):
        self.radius_err = 0

        count_not_in_branch = 0
        radius_span = in_rad_max - in_rad_min
        for p in self.data.pts:

            radii = []
            height = []
            for bi, b in enumerate(self.branch_vecs):
                (h, r) = self.data.proj_line_seg(p, self.axis_vec)
                radii.append(r)
                height.append(h)



            bin_height = int(min(max(0, np.floor((h + height_mid) / height_div)), 3))
            n_bins_height[bin_height] += 1

            # Error, 0 to 1 at radius cut-offs, make error inside radius cut_off worse than outside
            self.radius_err += np.power((r - self.radius) / radius_span, 2)
            if r < 0.5 * self.radius:
                count_inside += 1
            elif r > 2 * self.radius:
                count_outside += 1

        self.radius_err /= (len(self.data.pts) - count_outside - count_inside)

        self.percentage_in_err = count_inside / len(self.data.pts)
        self.percentage_out_err = count_outside / len(self.data.pts)
        self.height_distributon_err = (max(n_bins_height) - min(n_bins_height)) / len(self.data.pts)
        self.err = self.radius_err + self.percentage_in_err + self.percentage_out_err + self.height_distributon_err
        return self.err

    def optimize_ang(self, radius_min, radius_max):
        params = [0, 0, 0]  # rotation around x, y, z axis
        #  params = [0, 0, 0, 0, self.radius]
        ref_vecs = np.ndarray([3, 3])
        ref_vecs[0] = self.x_vec
        ref_vecs[1] = self.y_vec
        ref_vecs[2] = self.axis_vec

        radius_span = radius_max - radius_min
        radius_max = max(radius_max, self.radius + 0.2 * radius_span)
        radius_min = min(radius_min, self.radius - 0.2 * radius_span)
        new_params = fmin(Cylinder.cyl_err, params, args=(self, ref_vecs, radius_min, radius_max), disp=False)
        ret_vecs = PtsForFit.rotate_axis(new_params, ref_vecs, "xyz")
        self.x_vec = ret_vecs[0]
        self.y_vec = ret_vecs[1]
        self.axis_vec = ret_vecs[2]

        self.optimize_ang_err = self.err_fit(radius_min, radius_max)
        return self.optimize_ang_err

    def optimize_cyl(self, radius_min, radius_max):
        print("Beginning optimization: PCA-------------")
        self.fit_pca()
        print(self)
        for index in range(0, 1):
            print("Fit radius---------------")
            self.fit_radius()
            print(self)
            print("Angle-----------------")
            self.optimize_ang(radius_min, radius_max)
            print(self)
        print("Done\n\n")
        self.check()
        self.err = self.optimize_ang_err
        return self.err

    def check(self):
        return self.data.check_vectors()

    def __str__(self):
        if hasattr(self, "id"):
            str_id = "Id {0}, size {1}, err {2:.2f} center [".format(self.id(), len(self.data.pts), self.err)
            for x in self.data.pt_center:
                str_id = str_id + "{0:.2f} ".format(x)
            str_id = str_id + "] axis ["
            for x in self.axis_vec:
                str_id = str_id + "{0:.2f} ".format(x)
            ret_str = str_id + "] Height {0:.2f} radius {1:.2f}\n".format(self.height, self.radius)
        else:
            ret_str = "Empty cylinder"
        if hasattr(self, "pca_vals"):
            str1 = "  PCA ratios {0:.2f}, ".format(self.pca_ratio())
            str2 = "{0:.2f}\n".format(self.pca_second_ratio())
            ret_str = ret_str + str1 + str2

        ret_str += "  "
        for d in dir(self):
            if d not in dir(Cylinder) and "err" in d:
                if getattr(self, d) < 1e5:
                    ret_str = ret_str + " {0} {1:.3f}".format(d, getattr(self, d))

        return ret_str

    def read(self, fid, all_pts=None):
        self.check_header(fid)

        b_found_footer = False
        for l in fid:
            if self.check_footer(l, b_assert=False):
                b_found_footer = True
                break

            method_name, vals = self.get_class_member(l)
            if len(vals) == 0:
                if not l.startswith(self.data.header_name):
                    raise ValueError("Header incorrect on file read {0}".format(self.data.header_name))
                self.data.read(fid, all_pts, b_check_header=False)
            elif len(vals) == 1:
                setattr(self, method_name, vals[0])
            elif len(vals) == 3:
                val_as_ndarray = np.array(vals)
                setattr(self, method_name, val_as_ndarray)
            else:
                raise ValueError("Unknown Cylinder read {0} {1}".format(method_name, vals))

        if b_found_footer is False:
            raise ValueError("Bad Cylinder end read")

    def write(self, fid, write_pts=False):
        self.write_header(fid)
        self.write_class_members(fid, dir(self), Cylinder, ["data"])
        self.data.write(write_pts)
        self.write_footer(fid)

    @staticmethod
    def check_fit_cylinder(in_cyl, in_rad_min, in_rad_max, b_random_height, eps=1e-4):
        """ Check the various fit equations """

        z_eps = eps
        x_eps = 20 * eps
        y_eps = eps
        ang_eps = np.pi /6
        if b_random_height is True:
            z_eps = 0.015
            x_eps *= 5

        print("In cylinder")
        print(in_cyl)
        cyl_fit = Cylinder()
        cyl_fit.data.pts = in_cyl.data.pts

        cyl_fit.fit_pca()
        cyl_fit.err_fit(in_rad_min, in_rad_max)
        print("PCA")
        print(cyl_fit)
        pt_center_err = in_cyl.data.pt_center - cyl_fit.data.pt_center
        yerr = pt_center_err[1]
        zerr = pt_center_err[2]
        z_ang_err = np.arccos(abs(np.dot(cyl_fit.axis_vec, [0, 0, 1])))
        # Only y and c should be correct, along with axis in z direction
        if abs(yerr) > y_eps or abs(zerr) > z_eps or abs(z_ang_err) > ang_eps:
            print("Bad cylinder fit pca y {0:.6f} z {1:.6f} ang {2:.6f}".format(yerr, zerr, z_ang_err))
            cyl_fit.fit_pca()

        # should fix x
        cyl_fit.fit_radius()
        print("Fit radius")
        print(cyl_fit)
        pt_center_err = in_cyl.data.pt_center - cyl_fit.data.pt_center
        xerr = pt_center_err[0]
        yerr = pt_center_err[1]
        zerr = pt_center_err[2]
        rerr = cyl_fit.radius - in_cyl.radius
        if abs(xerr) > x_eps or abs(yerr) > y_eps or abs(zerr) > z_eps or abs(rerr) > eps:
            print("Bad cylinder fit radius x {0:.6f} y {1:.6f} z {2:.6f} r {3:.6f}".format(xerr, yerr, zerr, rerr))

        # Shouldn't break anything
        cyl_fit.optimize_ang(in_rad_min, in_rad_max)
        print("Optimize angle")
        print(cyl_fit)
        pt_center_err = in_cyl.data.pt_center - cyl_fit.data.pt_center
        xerr = pt_center_err[0]
        yerr = pt_center_err[1]
        zerr = pt_center_err[2]
        rerr = cyl_fit.radius - in_cyl.radius
        z_ang_err = np.arccos(abs(np.dot(cyl_fit.axis_vec, [0, 0, 1])))
        if abs(xerr) > x_eps or abs(yerr) > y_eps or abs(zerr) > z_eps or abs(rerr) > eps or abs(z_ang_err) > ang_eps:
            print("Bad circle fit angle x {0:.6f} y {1:.6f} z {2:.6f} r {3:.6f} ang {4:.6f}".format(xerr, yerr, zerr, rerr, z_ang_err))

        # should fix r
        cyl_fit.fit_radius()
        print("Fit radius")
        print(cyl_fit)
        pt_center_err = in_cyl.data.pt_center - cyl_fit.data.pt_center
        xerr = pt_center_err[0]
        yerr = pt_center_err[1]
        zerr = pt_center_err[2]
        rerr = cyl_fit.radius - in_cyl.radius
        z_ang_err = np.arccos(abs(np.dot(cyl_fit.axis_vec, [0, 0, 1])))
        if abs(xerr) > x_eps or abs(yerr) > y_eps or abs(zerr) > z_eps or abs(rerr) > eps:
            print("Bad circle fit radius 2 {0:.6f} {1:.6f} {2:.6f} {3:.6f}".format(xerr, yerr, zerr, rerr))

    @staticmethod
    def check_cylinder_fits(radius_min, radius_max, in_height):
        div = min(radius_max - radius_min, in_height) * 0.1
        cyl = Cylinder()
        cyl.axis_vec = np.array([0, 0, 1])
        cyl.x_vec = np.array([1, 0, 0])
        cyl.y_vec = np.array([0, 1, 0])

        print("Cylinder fit radius {0:0.2f} {1:0.2f} height {2:0.2f}".format(radius_min, radius_max, in_height))
        for off in np.linspace(0, div, 3):
            for theta in np.linspace(0.8 * np.pi/2, np.pi/2, 3):
                for noise in [0.00001, 0.0001, 0.001]:
                    for b in [True, False]:
                        print("Offset {0:.2f}, Theta {1:0.2f} noise {2:0.2f} random {3}".format(off, theta, noise, b))
                        cyl.data, cyl.radius, cyl.height = \
                            PtsForFit.make_cylinder(radius_min, radius_max, in_height=in_height,
                                                    in_theta=theta, offset=off,
                                                    x_noise=noise, y_noise=0.1*noise,
                                                    b_random_height=b)
                        Cylinder.check_fit_cylinder(cyl, radius_min, radius_max, b, eps=noise * 50)


if __name__ == '__main__':
    rad_min = 0.015  # Somewhere between 0.03 and 0.05
    rad_max = 0.09  # somewhere between 0.15 and 0.2
    height_min = 4 * rad_min
    height_max = 4 * rad_max

    """
    get_pca_ratio(rad_min, rad_max, height_min)
    get_pca_ratio(rad_min, rad_max, height_max)
    for i in range(0, 10):
        Cylinder.check_cylinder_fits(rad_min, rad_max, in_height=height_max)
    """

    cyl_pts = best_pts()
    cyl_pts.update(bad_pts())

    for cyl_id, label in cyl_pts.items():
        fname = "data/cyl_{0}.txt".format(cyl_id)
        cyl_check = Cylinder()
        with open(fname, "r") as f:
            cyl_check.read(f)

        print("Label: {0}".format(label))
        cyl_check.optimize_cyl(rad_min, rad_max)
        print(cyl_check)
        print("Done optimize\n\n")

        fname = "data/cyl_opt{0}.txt".format(cyl_id)
        with open(fname, "w") as f:
            cyl_check.write(f, write_pts=True)
