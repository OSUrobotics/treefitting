import cv2
import math
import params as p
import camera as c
import numpy as np
import sys
from pyk4a import PyK4APlayback
import curve_fitting as cf
import matplotlib.pyplot as plt
from skimage import io, color, img_as_bool
from curve_fitting import BezierBasedDetection, Bezier
#from generate_3d import Generate_3D
sys.path.insert(1, '../')
import bezier_cyl_3d as bc3

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

    def __init__(self, bin_msk):
        self.my_mask = np.asarray(img_as_bool(cv2.resize(color.rgb2gray(cv2.imread(bin_msk).astype('uint8')),(640,576))))
        self.my_mask2 = np.asarray(cv2.resize(color.rgb2gray(cv2.imread(bin_msk).astype('uint8')),(640,576)))
        k4a = PyK4APlayback(p.video_file)    #specify video file you wish to open in params
        k4a.open()
        capture = k4a.get_next_capture()
        self.my_depth = capture.depth
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
        ds = np.linspace(0, 1, p.curve_ammt)
        self.radii = radius_interpolator(ds)
        self.curve_pts, ts = curve.eval_by_arclen(ds, normalized=True)
        
        return self.radii, self.curve_pts, curve, ts

    def find_average(self):
        """
        Finds the average depth across a rectangular contour. Returns array of average depths and pixel width of tree.
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
            
            depths.append(arr[~np.isnan(arr).any(axis=0) and arr<10000])
            
            ar2 = (np.mean(depths[i]))*0.0001    #need to figure out the correct conversion between depth data and meters
            if np.isnan(ar2):    #replaces empty array values with 0
                depths_average.append(p.nan_value)
            else:
                depths_average.append(ar2)
            
        return depths_average, self.radii


class Determine_Match:
    """
    Determines if a side branch matches a leader branch.
    """
    def __init__(self):
        self.angle_limit = p.angle_limit
        
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
        for i, point in enumerate(curve_pts):
            dist1 = math.dist(point, end_pt_1)
            dist2 = math.dist(point, end_pt_2)

            if dist1 <= dist2:
                distance_arr.append(dist1)
                if prev_dist >= dist1:
                    ind, selected_end_point, dist = self.get_values(end_pt_1, i, dist1)
                    
            else:
                distance_arr.append(dist2)
                if prev_dist >= dist2:
                    ind, selected_end_point, dist = self.get_values(end_pt_2, i, dist2)

            prev_dist = dist

        return ind, selected_end_point, dist
        
        
    def get_values(self, end_point, ind, dist):
        """
        returns index, end_point, and dist after conditions of return_closest are satisfied
        """
        return ind, end_point, dist
            
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
        
        if angle < self.angle_limit:
            return 1

        elif angle > self.angle_limit:
            return 0

        else:
            return 0.5


class Build_3D:
    """
    Constructs 3D cylinders for a given curve
    Parameters:
        curve_points: given points along a Bezier curve
        radii: radii at curve_points
    """
    def __init__(self, curve_points, radii):
        self.radii = radii
        self.curve_points = curve_points

    def set_real_measurements(self):
        """
        Takes curve_points and converts to size in meters, scales up for visibility
        """
        camera = c.PinholeCameraModel()
        camera.from_npz(p.cam)
        assert isinstance(camera, c.PinholeCameraModel)
        for i in range(len(self.curve_points)):
            self.curve_points[i][0] = camera.getDeltaX(self.curve_points[i][0], self.curve_points[i][2])*p.scale_factor
            self.curve_points[i][1] = camera.getDeltaY(self.curve_points[i][1], self.curve_points[i][2])*p.scale_factor
            self.radii[i] = camera.getDeltaX(self.radii[i], self.curve_points[i][2])*p.scale_factor
                       
    def build(self):
        """
        Passes in beginning, middle, and end curve points and beginning and end radii to a class that builds a 3D mesh for each cylinder
        """
        self.set_real_measurements()
        for i in range(len(self.curve_points)-2):
            b = bc3.BezierCyl3D(self.curve_points[i], self.curve_points[i+1], self.curve_points[i+2], self.radii[i], self.radii[i+2])
            b.make_mesh()
            b.write_mesh(f"curve{i}.obj")

            
def manage_arrs(curve_2d, depth_arr):
    """
    Splits curve_2D and depth_arr to their components, as well as merging them
    Parameters:
        curve_2d: 2D Bezier points in x and y from a mask
        depth_arr: depth at those points
    """
    split_curve = np.split(curve_2d, 2, 1)
    _x = np.delete(split_curve[0], -1)
    _y = np.delete(split_curve[1], -1)
    _z = np.asarray(depth_arr)
    full_arr = np.array([_x, _y, _z]).T
    
    return _x, _y, _z, full_arr


if __name__ == '__main__':

    d = Depths_Average(p.mask_img)
    rad, curve_pts, curve, ts = d.get_radii()
    depth, num_pix = d.find_average()

    d2 = Depths_Average(p.follower_msk)
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
    
    b = Build_3D(leader_pts, rad)
    b.build()
    
    #b2 = Build_3D(follower_pts, rad2)
    #b2.build()
    
    #g = Generate_3D(leader_pts, x1, y1, z1, rad)
    #g.create_mesh()
    
    #g2 = Generate_3D(follower_pts, x2, y2, z2, num_pix2)
    #g2.create_mesh()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    print("Total: ", total_score)
    print("Angle Score: ", angle_score)
    print("Min Dist: ", minimum_distance)
    print("Depth Difference: ", depth_score)
    
    #Uncomment below if you wish to visualize both curves
    """
    ax.scatter(x1, y1, z1)
    ax.scatter(x2, y2, z2)
    
    ax.plot(x1, y1, z1)
    ax.plot(x2, y2, z2)
    ax.scatter(*np.asarray(end1).T)
    ax.scatter(*np.asarray(end2).T)
    ax.axis('equal')
    plt.show()
    """
