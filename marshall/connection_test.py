# THIS IS THE FILE YOU RUN
import cv2
import math
import params as p
import camera as c
import numpy as np
import curve_fitting as cf
import matplotlib.pyplot as plt
from skimage import io, color
from curve_fitting import BezierBasedDetection, Bezier


class Depths_Average:

    def __init__(self, depth_img, bin_msk):
        self.my_mask = np.asarray(color.rgb2gray(cv2.imread(bin_msk)))
        self.my_depth = np.asarray(cv2.imread(depth_img))
        self.my_depth2 = np.asarray(io.imread(depth_img))
        self.lined_mask = None
        self.radii = None
        self.curve_pts = None

    def get_radii(self):
        detector = cf.BezierBasedDetection(self.my_mask, use_medial_axis=True)
        curve = detector.fit()
        radius_interpolator = detector.get_radius_interpolator_on_path()
        ds = np.linspace(0, 1, 11)
        self.radii = radius_interpolator(ds)
        self.curve_pts, ts = curve.eval_by_arclen(ds, normalized=True)
        self.lined_mask = cv2.polylines(self.my_depth, [self.curve_pts.reshape((-1, 1, 2)).astype(int)], False,
                                        (0, 200, 200), 1)

        return self.radii, self.curve_pts, curve, ts

    def find_average(self):
        depths = []
        depths_average = []

        for i in range(len(self.curve_pts) - 1):
            right_bottom_pt = (
            self.curve_pts[i][0].astype(int) + self.radii[i].astype(int), self.curve_pts[i][1].astype(int))
            left_bottom_point = (
            self.curve_pts[i][0].astype(int) - self.radii[i].astype(int), self.curve_pts[i][1].astype(int))
            right_top_point = (
            self.curve_pts[i + 1][0].astype(int) + self.radii[i + 1].astype(int), self.curve_pts[i + 1][1].astype(int))
            left_top_pt = (
            self.curve_pts[i + 1][0].astype(int) - self.radii[i + 1].astype(int), self.curve_pts[i + 1][1].astype(int))

            arr = np.array([right_bottom_pt, left_bottom_point, left_top_pt, right_top_point])
            img = cv2.drawContours(self.my_depth.copy(), [arr.astype(int)], -1, color=255, thickness=-1)
            pts = np.where(img == 255)

            depths.append(self.my_depth2[pts[0], pts[1]] / 1000)  # LOOK INTO THIS MORE
            depths_average.append(np.mean(depths[i]))

        # For visualizing the radii:
        # for i in range(len(self.curve_pts)):
        # cv2.circle(self.lined_mask, self.curve_pts[i].astype(int), self.radii[i].astype(int), (0, 0, 255), thickness=1)

        return depths_average, self.radii


class Determine_Match:

    def return_closest(self, curve_pts, end_pt_1, end_pt_2):
        distance_arr = []
        prev_dist = math.inf
        for i in range(len(curve_pts)):
            dist1 = math.dist(curve_pts[i], end_pt_1)
            dist2 = math.dist(curve_pts[i], end_pt_2)

            if dist1 <= dist2:
                distance_arr.append(dist1)
                var1 = end_pt_1

                if prev_dist >= dist1:
                    var = var1
                    ind = i
                    dist = dist1
            else:
                distance_arr.append(dist2)
                var2 = end_pt_2
                dist = dist2

                if prev_dist >= dist2:
                    var = var2
                    ind = i
                    dist = dist2

            prev_dist = dist

            return ind, var, dist

    def depth_scoring(self, closest_depth, end_depth):

        return abs(closest_depth - end_depth)

    def check_angle_match(self, vec_1, vec_2):

        angle = np.arccos(
            np.clip(np.dot((vec_1 / np.linalg.norm(vec_1)), (vec_2 / np.linalg.norm(vec_2))), -1.0, 1.0)) * (
                            180 / np.pi)

        if angle < p.angle_limit:
            return 1

        elif angle > p.angle_limit:
            return 0

        else:
            return 0.5


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def manage_arrs(curve_2d, depth_arr):
    split_curve = np.split(curve_2d, 2, 1)
    _x = np.delete(split_curve[0], -1)
    _y = np.delete(split_curve[1], -1)
    _z = np.asarray(depth_arr)
    full_arr = np.array([_x, _y, _z]).T
    return _x, _y, _z, full_arr


if __name__ == '__main__':
    """
    For testing purposes, build your own arrays:
    branch1 = [0.5, 0.5, -0.5]
    x1 = np.linspace(0, 0.5, 5)
    y1 = np.linspace(0, 0.5, 5) 
    z1 = np.linspace(0.5, -0.5, 5)
    
    branch2=[0, 0, 2]
    x2 = np.linspace(0, 0, 5)
    y2 = np.linspace(0, 0, 5)
    z2 = np.linspace(0, 2, 5)
    follower_curve_pts = np.array([x2, y2, z2]).T
    leader_curve = np.array([x1, y1, z1]).T
    """

    camera = c.PinholeCameraModel()
    camera.from_npz(p.cam)
    assert isinstance(camera, c.PinholeCameraModel)

    d = Depths_Average(p.depth_img, p.mask_img)
    rad, curve_pts, curve, ts = d.get_radii()
    depth, num_pix = d.find_average()

    d2 = Depths_Average(p.depth_img, p.follower_msk)
    rad2, curve_pts2, curve2, ts2 = d2.get_radii()
    depth2, num_pix2 = d2.find_average()

    x1, y1, z1, leader_pts = manage_arrs(curve_pts, depth)
    x2, y2, z2, follower_pts = manage_arrs(curve_pts2, depth2)

    dm = Determine_Match()
    end1 = follower_pts[0]
    end2 = follower_pts[-1]

    index, end_pt, minimum_distance = dm.return_closest(leader_pts, end1, end2)
    depth_score = dm.depth_scoring(z1[index], end_pt[2])
    leader_tan = curve.tangent(ts[index])
    follower_tan = curve2.tangent(ts2[index])

    angle_score = dm.check_angle_match(leader_tan, follower_tan)
    total_score = angle_score + depth_score + minimum_distance

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    print(total_score)

    ax.plot(x1, y1, z1)
    ax.plot(x2, y2, z2)
    ax.scatter(*np.asarray(end1).T)
    ax.scatter(*np.asarray(end2).T)
    set_axes_equal(ax)
    plt.show()
