#!/usr/bin/env python3

import numpy as np
from scipy.optimize import fmin
from test_pts import Best_pts, Bad_pts
from list_read_write import ReadWrite


class Cylinder(ReadWrite):
    def __init__(self):
        super(Cylinder, self). __init__("CYLINDER")

        self.pt_center = np.array([0, 0, 0])
        self.axis_vec = np.array([0, 1, 0])
        self.x_vec = np.array([1, 0, 0])
        self.y_vec = np.array([0, 0, 1])
        self.height = 1
        self.radius = 0.1
        self.id = -1
        self.pts = []
        self.pts_ids = []
        self.pca_vals = [1, 1, 1]

        # From err_fit
        self.radius_err = 1e6
        self.percentage_in_err = 1e6
        self.height_distributon_err = 1e6
        self.err = 1e6

        # Set in various optimization routines
        self.pca_err = 1e6
        self.fit_radius_2d_err = 1e6
        self.optimize_ang_err = 1e6

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
        return Cylinder.err_circle(pt_center, radius_sq, pts_2d)

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
            err = Cylinder.calc_2d_circle_err(pt_center, radius, pts_2d)

        return pt_center, radius, err

    @staticmethod
    def fit_circle_center(pts_2d, radius, b_ret_err=False):
        radius_sq = np.power(radius, 2)
        pt_center = fmin(Cylinder.err_circle, np.array([0, 0]), args=(radius_sq, pts_2d), disp=False)

        err = 0.0
        if b_ret_err is True:
            err = Cylinder.calc_2d_circle_err(pt_center, radius, pts_2d)

        return pt_center, radius, err

    def fit_radius(self, min_radius, max_radius):
        """
        Project down into the plane and along the main vector, then fit a circle
        Sets the center point, vectors, height, and radius
        Clamps the radius between the given values
        :param min_radius: smallest radius of branch
        :param max_radius: largest radius of trunk
        :return: error of the fit
        """
        height = []
        xy = []
        for p in self.pts:
            d = p - self.pt_center
            height.append(np.dot(d, self.axis_vec))
            vec_xy = d - self.axis_vec * height[-1]
            xy.append([np.dot(vec_xy, self.x_vec), np.dot(vec_xy, self.y_vec)])

        # Fit circle to projected points
        (pt_center_2d, radius_2d, self.fit_radius_2d_err) = self.fit_circle(xy, b_ret_err=True)
        self.radius = radius_2d
        """
        # Don't set the radius bigger/smaller than target radii
        self.radius = min(max_radius, max(min_radius, radius_2d))
        self.height = max(height) - min(height)

        # re-fit the center if need be
        if self.radius != radius_2d:
            pt_center_2d, _, self.fit_radius_2d_err = self.fit_circle_center(xy, self.radius, b_ret_err=True)
        """
        # Move the center point to the middle height-wise and by the detected circle center
        median_height = np.mean(height)
        self.pt_center = self.pt_center + median_height * self.axis_vec
        self.pt_center = self.pt_center + pt_center_2d[0] * self.x_vec + pt_center_2d[1] * self.y_vec

        return self.fit_radius_2d_err

    def pca_ratio(self):
        return self.pca_vals[2] / self.pca_vals[1]

    def pca_second_ratio(self):
        return self.pca_vals[1] / self.pca_vals[0]

    def score_pca(self, pca_ratio=7):
        """
        Score pca ratios by expected values
           For radius 0.01, 0.09  height 0.36
           pca ratio min 2.50 max 34.41, avg 9.99
           For radius 0.01, 0.09  height 0.06
           pca ratio min 1.65 max 8.09, avg 5.09
        :param pca_ratio:
        :return: How much pca score varies from target
        """
        score = 0
        ratio = self.pca_ratio()
        pca_min = 1.6
        pca_max = 40
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
        values, vectors = np.linalg.eig(V)

        vecs = []
        self.pca_vals = []
        for y, x in sorted(zip(values, vectors)):
            vecs.append(x)
            self.pca_vals.append(y)

        self.axis_vec = vecs[2]
        self.x_vec = vecs[0]
        self.y_vec = vecs[1]

        #  if abs(self.axis_vec[1]) > 0.1:
        #      for i in range(0, 3):
        #          print("{0:.6f} {1:.6f} {2:.6f}".format(np.median(shift_data[:, i]), np.min(shift_data[:, i]), np.max(shift_data[:, i])))
        # Decent guess for the radius
        self.radius = np.sqrt(0.5 * (self.pca_vals[1] + self.pca_vals[0]))

        # Get height
        height = []
        for p in self.pts:
            d = p - self.pt_center
            height.append(np.dot(d, self.axis_vec))

        self.height = max(height) - min(height)

        self.pca_err = self.score_pca()
        return self.pca_err

    # reconstruct axis from pt center and orientation
    @staticmethod
    def reconstruct_pt_axis(xs, pt_c, ref_vecs):
        x_add = ref_vecs[0] * xs[0]
        y_add = ref_vecs[1] * xs[1]
        new_axis_vec = ref_vecs[2] + x_add + y_add
        new_axis_vec = (ref_vecs[2] + x_add + y_add) / np.linalg.norm(new_axis_vec)
        #  pt_center = pt_c + ref_vecs[0] * xs[2] + ref_vecs[1] * xs[3]
        #  return new_axis_vec, pt_center
        return new_axis_vec, pt_c

    @staticmethod
    def cyl_err(xs, cylinder, pt_c, ref_vecs, radius_min, radius_max):
        cylinder.axis_vec, cylinder.pt_center = Cylinder.reconstruct_pt_axis(xs, pt_c, ref_vecs)
        #  cylinder.radius = xs[4]
        err_proj = 0
        max_h = 0
        min_h = 0
        radius_span = 0.5 * (radius_max - radius_min)
        for p in cylinder.pts:
            (h, d) = cylinder.proj_line_seg(p)
            err_d = 1
            if radius_min < d < cylinder.radius:
                err_d = (d - radius_min) / radius_span
            elif cylinder.radius < d < radius_max:
                err_d = (radius_max - d) / radius_span

            err_proj += np.power(err_d, 2)
            max_h = max(max_h, h)
            min_h = min(min_h, h)

        err_proj = err_proj / len(cylinder.pts)
        proj_height = max_h - min_h
        err_h = 0
        if proj_height < cylinder.height:
            err_h = np.power(1.0 - proj_height / cylinder.height, 2)

        #  err_radius = 0
        #  if xs[4] < radius_min:
        #      err_radius = 100 * np.power((xs[4] - radius_min) / radius_min, 2)
        #  elif xs[4] > radius_max:
        #      err_radius = 10 * np.power((xs[4] - radius_max) / radius_min, 2)
        #  return err_proj + err_h + err_radius
        return err_proj + err_h

    def proj_line_seg(self, pt):
        t_on_line = np.dot(pt - self.pt_center, self.axis_vec)
        pt_on_line = self.pt_center + self.axis_vec * t_on_line
        vec_to_line = pt - pt_on_line
        #  check = np.dot(vec_to_line, self.axis_vec)
        #  if abs(check) > 0.0001:
        #      raise ValueError("Bad project line segment {0}".format(check))

        dist_to_line = np.linalg.norm(vec_to_line)
        return t_on_line, dist_to_line

    def err_fit(self, in_rad_min, in_rad_max):
        self.radius_err = 0

        n_bins_height = [0, 0, 0, 0]
        height_div = self.height / len(n_bins_height)
        height_mid = self.height / 2.0
        count_inside = 0
        count_outside = 0
        radius_span = in_rad_max - in_rad_min
        for p in self.pts:
            (h, r) = self.proj_line_seg(p)
            bin_height = int(min(max(0, np.floor((h + height_mid) / height_div)), 3))
            n_bins_height[bin_height] += 1

            # Error, 0 to 1 at radius cut-offs, make error inside radius cut_off worse than outside
            self.radius_err += np.power((r - self.radius) / radius_span, 2)
            if r < 0.5 * self.radius:
                count_inside += 1
            elif r > 2 * self.radius:
                count_outside += 1

        self.radius_err /= (len(self.pts) - count_outside - count_inside)

        self.percentage_in_err = count_inside / len(self.pts)
        self.percentage_out_err = count_outside / len(self.pts)
        self.height_distributon_err = (max(n_bins_height) - min(n_bins_height)) / len(self.pts)
        self.err = self.radius_err + self.percentage_in_err + self.percentage_out_err + self.height_distributon_err
        return self.err

    def optimize_ang(self, radius_min, radius_max):
        params = [0, 0]
        #  params = [0, 0, 0, 0, self.radius]
        p_c = self.pt_center
        ref_vecs = np.ndarray([3, 3])
        ref_vecs[0] = self.x_vec
        ref_vecs[1] = self.y_vec
        ref_vecs[2] = self.axis_vec

        new_params = fmin(Cylinder.cyl_err, params, args=(self, p_c, ref_vecs, radius_min, radius_max), disp=False)
        self.axis_vec, self.pt_center = Cylinder.reconstruct_pt_axis(new_params, p_c, ref_vecs)
        #  self.radius = new_params[4]

        # Fix x and y vec
        x_vec = self.x_vec - np.dot(self.axis_vec, self.x_vec) * self.axis_vec
        self.x_vec = x_vec / np.linalg.norm(self.x_vec)

        y_vec = self.y_vec - self.axis_vec * np.dot(self.axis_vec, self.y_vec)
        self.y_vec = y_vec / np.linalg.norm(self.y_vec)

        self.optimize_ang_err = self.err_fit(radius_min, radius_max)
        return self.optimize_ang_err

    def optimize_cyl(self, radius_min, radius_max):
        print("Beginning optimization: PCA-------------")
        self.fit_pca()
        print(self)
        for index in range(0, 1):
            print("Fit radius---------------")
            self.fit_radius(radius_min, radius_max)
            print(self)
            print("Angle-----------------")
            self.optimize_ang(radius_min, radius_max)
            print(self)
        print("Done\n\n")
        self.check()
        self.err = self.optimize_ang_err
        return self.err

    def check(self):
        b_ok = True
        check_x = np.dot(self.x_vec, self.axis_vec)
        check_y = np.dot(self.y_vec, self.axis_vec)
        check_z = np.dot(self.y_vec, self.x_vec)

        if abs(check_x) > 0.001 or abs(check_y) > 0.001 or abs(check_z) > 0.001:
            b_ok = False

        check_len_x = np.dot(self.x_vec, self.x_vec)
        check_len_y = np.dot(self.y_vec, self.y_vec)
        check_len_z = np.dot(self.axis_vec, self.axis_vec)

        if abs(1-check_len_x) > 0.001 or abs(1-check_len_y) > 0.001 or abs(1-check_len_z) > 0.001:
            b_ok = False

        return b_ok

    def __str__(self):
        if hasattr(self, "id"):
            str_id = "Id {0}, size {1}, err {2:.2f} center [".format(self.id, len(self.pts), self.err)
            for x in self.pt_center:
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
            raise ValueError("Bad Cylinder end read")

        if len(self.pts) == 0 and len(self.pts_ids) > 0:
            try:
                self.pts = np.ndarray([len(self.pts_ids), 3])
                for index, pt_id in enumerate(self.pts_ids):
                    self.pts[index] = all_pts[pt_id]
            except TypeError:
                pass

    def write(self, fid, write_pts=False):
        self.write_header(fid)
        self.write_class_members(fid, dir(self), Cylinder, ["pts"])
        if write_pts is True:
            fid.write("pts {0}\n".format(len(self.pts_ids)))
            for p in self.pts:
                fid.write("{0}\n".format(p))
        self.write_footer(fid)


def make_cyl(in_rad_min=0.015, in_rad_max=0.09, in_height=0.28, in_theta=1.6, offset=0.1, x_noise=0.00001, y_noise=0.000001):
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
    :return:
    """
    off = offset / 2
    xc = 1 + np.random.uniform(low=-off, high=off)
    yc = -2 + np.random.uniform(low=-off, high=off)
    zc = 0 + np.random.uniform(low=-off, high=off)
    r = in_rad_min + np.random.uniform(low=in_rad_min, high=in_rad_max)
    h = in_height + np.random.uniform(low=-off/10, high=off/10)

    pts_2d = []
    pts_3d = []
    for t in np.linspace(-in_theta, in_theta, 30):
        for zt in np.linspace(-h/2, h/2, 10):
            x = xc + r * np.cos(t) + np.random.uniform(low=-x_noise/2, high=x_noise/2)
            y = yc + r * np.sin(t) + np.random.uniform(low=-y_noise/2, high=y_noise/2)
            z = zc + zt + np.random.uniform(low=-y_noise/2, high=y_noise/2)
            pts_2d.append([x, y])
            pts_3d.append([x, y, z])

    cyl = Cylinder()
    cyl.pt_center = np.array([xc, yc, zc])
    cyl.radius = r
    cyl.height = h
    cyl.pts = np.array(pts_3d)
    cyl.axis_vec = np.array([0, 0, 1])
    cyl.x_vec = np.array([1, 0, 0])
    cyl.y_vec = np.array([0, 1, 0])
    print("Mean noise: ", end=' ')
    for i in range(0, 3):
        m = np.mean(cyl.pts[:, i]) - cyl.pt_center[i]
        print("{0:.5f} ".format(m), end=' ')

    print(" err: {0:0.2f}\n", cyl.err_fit(in_rad_min, in_rad_max))
    return cyl, np.array(pts_2d)


def check_fit_circ(in_cyl, pts_2d, in_rad_min, in_rad_max, eps=1e-4):
    """ Check the various fit equations """

    pt_center, radius, in_cyl.fit_radius_2d_err = Cylinder.fit_circle(pts_2d, b_ret_err=True)
    xerr = pt_center[0] - in_cyl.pt_center[0]
    yerr = pt_center[1] - in_cyl.pt_center[1]
    rerr = radius - in_cyl.radius

    if abs(xerr) > eps or abs(yerr) > eps or abs(rerr) > eps or in_cyl.fit_radius_2d_err > eps:
        raise ValueError("Bad circle fit {0} {1} {2} {3}".format(xerr, yerr, rerr, in_cyl.fit_radius_2d_err))

    pt_2d_center = [np.mean(pts_2d[:, index]) for index in range(0, 2)]
    pts_2d_centered = pts_2d - pt_2d_center
    pt_center, rad, err = Cylinder.fit_circle_center(pts_2d_centered, radius, b_ret_err=True)
    xerr = pt_center[0] + pt_2d_center[0] - in_cyl.pt_center[0]
    yerr = pt_center[1] + pt_2d_center[1] - in_cyl.pt_center[1]

    if abs(xerr) > eps or abs(yerr) > eps or abs(err) > eps:
        raise ValueError("Bad circle fit center {0} {1} {2}".format(xerr, yerr, err))

    print("In cylinder")
    print(in_cyl)
    cyl_fit = Cylinder()
    cyl_fit.pts = in_cyl.pts

    cyl_fit.fit_pca()
    cyl_fit.err_fit(in_rad_min, in_rad_max)
    print("PCA")
    print(cyl_fit)
    pt_center_err = in_cyl.pt_center - cyl_fit.pt_center
    yerr = pt_center_err[1]
    zerr = pt_center_err[2]
    z_ang_err = np.arccos(abs(np.dot(cyl_fit.axis_vec, [0, 0, 1])))
    # Only y and c should be correct, along with axis in z direction
    if abs(yerr) > eps or abs(zerr) > eps or abs(z_ang_err) > np.pi/6:
        print("Bad circle fit pca {0} {1} {2}".format(yerr, zerr, z_ang_err))
        cyl_fit.fit_pca()

    # should fix x
    cyl_fit.fit_radius(in_rad_min, in_rad_max)
    print("Fit radius")
    print(cyl_fit)
    pt_center_err = in_cyl.pt_center - cyl_fit.pt_center
    xerr = pt_center_err[0]
    yerr = pt_center_err[1]
    zerr = pt_center_err[2]
    rerr = radius - in_cyl.radius
    if abs(xerr) > eps*10 or abs(yerr) > eps or abs(zerr) > eps or abs(rerr) > eps:
        print("Bad circle fit radius {0} {1} {2} {3}".format(xerr, yerr, zerr, rerr))

    return
    # Shouldn't break anything
    cyl_fit.optimize_ang(in_rad_min, in_rad_max)
    print("Optimize angle")
    print(cyl_fit)
    pt_center_err = in_cyl.pt_center - cyl_fit.pt_center
    xerr = pt_center_err[0]
    yerr = pt_center_err[1]
    zerr = pt_center_err[2]
    rerr = radius - in_cyl.radius
    z_ang_err = np.arccos(abs(np.dot(cyl_fit.axis_vec, [0, 0, 1])))
    if abs(xerr) > eps or abs(yerr) > eps or abs(zerr) > eps*20 or abs(rerr) > eps or abs(z_ang_err) > np.pi/6:
        print("Bad circle fit angle {0} {1} {2} {3} {4}".format(xerr, yerr, zerr, rerr, z_ang_err))

    # should fix r
    cyl_fit.fit_radius(in_rad_min, in_rad_max)
    print("Fit radius")
    print(cyl_fit)
    pt_center_err = in_cyl.pt_center - cyl_fit.pt_center
    xerr = pt_center_err[0]
    yerr = pt_center_err[1]
    zerr = pt_center_err[2]
    rerr = radius - in_cyl.radius
    if abs(xerr) > eps or abs(yerr) > eps or abs(zerr) > eps*20 or abs(rerr) > eps:
        print("Bad circle fit radius 2 {0} {1} {2} {3}".format(xerr, yerr, zerr, rerr))


def check_circle_fits(radius_min, radius_max, in_height):
    div = min(radius_max - radius_min, in_height) * 0.1
    for off in np.linspace(0, div, 3):
        for theta in np.linspace(0.8 * np.pi/2, np.pi/2, 3):
            for noise in [0.00001, 0.0001, 0.001]:
                cyl, pts_2d = make_cyl(radius_min, radius_max, in_height=in_height, in_theta=theta,
                                       x_noise=noise, y_noise=0.1*noise)
                check_fit_circ(cyl, pts_2d, radius_min, radius_max, eps=noise * 50)


def get_pca_ratio(radius_min, radius_max, in_height):
    div = min(radius_max - radius_min, in_height) * 0.1
    pca_ratio = []
    pca_second_ratio = []
    for off in np.linspace(0, div, 3):
        for theta in np.linspace(0.8 * np.pi/2, np.pi/2, 3):
            for noise in [0.00001, 0.0001, 0.001]:
                cyl, pts_2d = make_cyl(radius_min, radius_max, in_height=in_height, in_theta=theta,
                                       x_noise=noise, y_noise=0.1*noise)

                cyl.fit_pca()
                pca_ratio.append(cyl.pca_ratio())
                pca_second_ratio.append(cyl.pca_second_ratio())

    print("For radius {0:.2f}, {1:.2f}  height {2:.2f}".format(radius_min, radius_max, in_height))
    print("pca ratio min {0:.2f} max {1:.2f}, avg {2:.2f}".format(np.min(pca_ratio), np.max(pca_ratio), np.mean(pca_ratio)))
    print("pca second ratio min {0:.2f} max {1:.2f}, avg {2:.2f}".format( np.min(pca_second_ratio), np.max(pca_second_ratio), np.mean(pca_second_ratio)))

if __name__ == '__main__':
    rad_min = 0.015  # Somewhere between 0.03 and 0.05
    rad_max = 0.09  # somewhere between 0.15 and 0.2
    height_min = 4 * rad_min
    height_max = 4 * rad_max

    #  get_pca_ratio(rad_min, rad_max, height_min)
    #  get_pca_ratio(rad_min, rad_max, height_max)
    for i in range(0, 10):
        check_circle_fits(rad_min, rad_max, in_height=height_max)

    cyl_pts = Best_pts()
    cyl_pts.update(Bad_pts())


    for cyl_id, label in cyl_pts.items():
        fname =  "data/cyl_{0}.txt".format(cyl_id)
        cyl = Cylinder
        with open(fname, "r") as f:
            cyl.read(f)
        print("Label: {0}".format(label))
        cyl.optimize_cyl(rad_min, rad_max)
        print(cyl)
        print("Done optimize\n\n")

        fname =  "data/cyl_opt{0}.txt".format(cyl_id)
        with open(fname, "w") as f:
            cyl.write(f, write_pts=True)

