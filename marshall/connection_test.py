import cv2
import math
import params as p
import camera as c
import numpy as np
import curve_fitting as cf
import matplotlib.pyplot as plt
from skimage import io, color, img_as_bool
from curve_fitting import BezierBasedDetection, Bezier


class Depths_Average:
    """
    Parameters:
        depth_img: Currently uses a frame from Envy orchard, the depth image of a single tree.
        bin_msk: Binary mask of a tree, corresponds with depth data.
    selected_end_pointiables:
        my_mask: reads binary mask
        my_depth: reads depth image using skimage, which gets the depth values of the image
        lined_mask: commented, used for visualization
        radii: radii of branch at evaluated points along curve
        curve_pts: evaluated points along Bezier curve
        
    """

    def __init__(self, depth_img, bin_msk):
        self.my_mask = np.asarray(img_as_bool(cv2.resize(color.rgb2gray(cv2.imread(bin_msk).astype('uint8')),(640,576))))
        self.my_mask2 = np.asarray(cv2.resize(color.rgb2gray(cv2.imread(bin_msk).astype('uint8')),(640,576)))
        #self.my_depth = np.asarray(io.imread(depth_img))
        npimg = np.fromfile(depth_img, dtype=np.uint16)
        imageSize = (640,576)
        self.my_depth = npimg.reshape(imageSize)
        self.lined_mask = None
        self.radii = None
        self.curve_pts = None

    def get_radii(self):
        """
        Uses curve_fitting to get Bezier curve by first fitting a curve and then determining radii along evaluated points.
        Currently evaluates for 11 points. Returns radii, points along the curve, the curve object, and the "t" values.
        """
        detector = cf.BezierBasedDetection(self.my_mask, use_medial_axis=True)
        curve = detector.fit()
        radius_interpolator = detector.get_radius_interpolator_on_path()
        ds = np.linspace(0, 1, 11)
        self.radii = radius_interpolator(ds)
        self.curve_pts, ts = curve.eval_by_arclen(ds, normalized=True)
        #self.lined_mask = cv2.polylines(self.my_mask2, [self.curve_pts.reshape((-1, 1, 2)).astype(int)], False, (0, 200, 200), 1)
        
        return self.radii, self.curve_pts, curve, ts

    def find_average(self):
        """
        Finds the average depth across a rectangular contour. Returns array of average depths and pixel width of tree.
        TODO: Figure out the correct depth conversion by testing the camera's parameters
        """
        depths = []
        depths_average = []

        for i in range(len(self.curve_pts) - 1):
            right_bottom_pt = (
            self.curve_pts[i][0].astype(int) + (self.radii[i]/2).astype(int), self.curve_pts[i][1].astype(int))
            left_bottom_point = (
            self.curve_pts[i][0].astype(int) - (self.radii[i]/2).astype(int), self.curve_pts[i][1].astype(int))
            right_top_point = (
            self.curve_pts[i + 1][0].astype(int) + (self.radii[i + 1]/2).astype(int), self.curve_pts[i + 1][1].astype(int))
            left_top_pt = (
            self.curve_pts[i + 1][0].astype(int) - (self.radii[i + 1]/2).astype(int), self.curve_pts[i + 1][1].astype(int))

            arr = np.array([right_bottom_pt, left_bottom_point, left_top_pt, right_top_point])
            img = cv2.cvtColor((self.my_mask2.copy().astype('uint8')), cv2.COLOR_GRAY2BGR)
            img = cv2.drawContours(img, [arr.astype(int)], -1, color=(0,0,255), thickness=-1)
            
            pts = np.asarray(np.where(img == 255))
            
            arry=self.my_depth[pts[0],pts[1]]
            
            arr = arry[np.where(arry!=0)]
            
            depths.append(arr[~np.isnan(arr).any(axis=0)])
            ar2 = (np.mean(depths[i]))*0.0001    #need to figure out the correct conversion between depth data and meters
            if np.isnan(ar2):    #replaces empty array values with 0, might be a deeper problem--> look more into this later
                depths_average.append(p.nan_value)
            else:
                depths_average.append(ar2)
            # For visualizing the radii:
            #img = img.astype(np.uint8)
            
            #cv2.waitKey(0)
            #cv2.imshow('im', img)
        #for i in range(len(self.curve_pts)):
            #cv2.circle(self.lined_mask, self.curve_pts[i].astype(int), self.radii[i].astype(int), (0, 0, 255), thickness=1)
        #cv2.imshow("i", self.lined_mask.astype(np.uint8))
        
        return depths_average, self.radii.astype(int)*2


class Determine_Match:
    """
    Determines if a side branch matches a leader branch.
    """
    
    def return_closest(self, curve_pts, end_pt_1, end_pt_2):
        """
        Returns the closest distance between evaluated points on leader curve and end points of side branch curve.
        Parameters:
            curve_pts: all points evaluated on leader curve
            end_pt_1: an end point on side branch curve
            end_pt_2: other end point on side branch curve
        Returns:
            ind: index of matching points
            selected_end_point: the matching end point
            dist: minimum distance between selected points 
        """
        distance_arr = []
        prev_dist = math.inf
        for i in range(len(curve_pts)):
            dist1 = math.dist(curve_pts[i], end_pt_1)
            dist2 = math.dist(curve_pts[i], end_pt_2)

            if dist1 <= dist2:
                distance_arr.append(dist1)
                selected_end_point1 = end_pt_1

                if prev_dist >= dist1:
                    selected_end_point = selected_end_point1
                    ind = i
                    dist = dist1
            else:
                distance_arr.append(dist2)
                selected_end_point2 = end_pt_2
                dist = dist2

                if prev_dist >= dist2:
                    selected_end_point = selected_end_point2
                    ind = i
                    dist = dist2

            prev_dist = dist

        return ind, selected_end_point, dist

    def depth_scoring(self, closest_depth, end_depth):
        """
        Parameters:
            closest_depth: depth of closest point on leader curve
            end_depth: depth of selected end point
        Returns:
            the difference between the two (the difference in depth)
        """
        return abs(closest_depth - end_depth)


    def check_angle_match(self, vec_1, vec_2):
        """
        Checks if the angle falls within the given angle parameters.
        Parameters:
            vec_1: a tangent line to one of the selected points
            vec_2: a tangent line to the other selected point
        Returns:
            a score based on if the angle between the two vectors falls in an acceptable range. Currently set to arbitrary numbers.
        """
        angle = np.arccos(
            np.clip(np.dot((vec_1 / np.linalg.norm(vec_1)), (vec_2 / np.linalg.norm(vec_2))), -1.0, 1.0)) * (
                            180 / np.pi)
        print(angle)
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
    
    for i in range(len(depth)):
        real_width = camera.getDeltaX(num_pix[i], depth[i])
        print(f'With this camera, {num_pix[i]} pixels at a depth of {depth[i]} m is {real_width:.4f} m')
    for i in range(len(depth2)):
        real_width = camera.getDeltaX(num_pix2[i], depth2[i])
        print(f'With this camera, {num_pix2[i]} pixels at a depth of {depth2[i]} m is {real_width:.4f} m')
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

    print("Total: ", total_score)
    print("Angle Score: ", angle_score)
    print("Min Dist: ", minimum_distance)
    print("Depth Difference: ", depth_score)
    
    ax.scatter(x1, y1, z1)
    ax.scatter(x2, y2, z2)
    ax.plot(x1, y1, z1)
    ax.plot(x2, y2, z2)
    ax.scatter(*np.asarray(end1).T)
    ax.scatter(*np.asarray(end2).T)
    ax.axis('equal')
    #set_axes_equal(ax)
    plt.show()

