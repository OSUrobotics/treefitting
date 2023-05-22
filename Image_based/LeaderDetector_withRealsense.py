#!/usr/bin/env python3

# Read in one masked image, the flow image, and the two rgbd images and
#  a) Find the most likely mask
#  b) Fit a bezier to that mask
#    b.1) use PCA to find the center estimate of the mask at 3 locations
#    b.2) Fit the bezier to the horizontal slices, assuming mask is correct
#    b.3) Fit the bezier to the edges

import numpy as np
from glob import glob
import csv
import cv2
import json
from os.path import exists
from cyl_fit_2d import Quad
from line_seg_2d import draw_line, draw_box, draw_cross, LineSeg2D
from scipy.cluster.vq import kmeans, whiten, vq
import pyrealsense2 as rs
import tempfile
import os
from branch_geometry_cindy import MakeTreeGeometry
from ransac_circle import RANSAC
import matplotlib.pyplot as plt
from circle_fit import taubinSVD
from annonation.json_dict import json_dict
from trunk_width_estimator import TrunkWidthEstimator
class LeaderDetector:
    image_type = {"Mask", "Flow", "RGB1", "RGB2", "Edge", "RGB_Stats", "Mask_Stats", "Edge_debug"}

    _width = 0
    _height = 0

    _x_grid = None
    _y_grid = None

    @staticmethod
    def _init_grid_(in_im):
        """ INitialize width, height, xgrid, etc so we don't have to keep re-making it
        :param in_im: Input image
        """
        if LeaderDetector._width == in_im.shape[1] and LeaderDetector._height == in_im.shape[0]:
            return
        LeaderDetector._width = in_im.shape[1]
        LeaderDetector._height = in_im.shape[0]

        LeaderDetector._x_grid, LeaderDetector._y_grid = np.meshgrid(np.linspace(0.5, LeaderDetector._width - 0.5, LeaderDetector._width), np.linspace(0.5,  LeaderDetector._height -  0.5,  LeaderDetector._height))

    def __init__(self, path, image_name, b_output_debug=True, b_recalc=False):
        """ Read in the image, mask image, flow image, 2 rgb images
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @param b_recalc: Force recalculate the result, y/n"""

        self.path_debug = path + "DebugImages/"
        path_calculated = path + "CalculatedData/"

        self.name = image_name
        # Read in all images that have name_ and are not debugging images
        self.read_images(path, image_name)
        # Split the mask into connected components, each of which might be a vertical leader
    def find_four_corners_v2(self, mask, quad):
        #find corners of the mask closest to the curve
        #find the top left corner
        #find the top right corner
        #find the bottom left corner
        #find the bottom right corner
        #return the four corners
        bound_x, bound_y = np.where(mask == 1)
        #generate points on the curve with length of bound_x
        curve_left = np.zeros((len(bound_x),2))
        curve_right = np.zeros((len(bound_x),2))
        i=0
        for t in np.linspace(0.1,0.9,len(bound_x)):
            pix_edge = np.ceil(quad.edge_pts(t))
            curve_left[i,:] = pix_edge[0]
            curve_right[i,:] = pix_edge[1]
            i+=1

        #search for the top left corner closest to the points on left curve
        #first occurence is top left
        #last occurence is top right
        dists_left_y = np.linalg.norm(np.array([bound_y,bound_x]).T - curve_left, axis=1)
        dists_right_y = np.linalg.norm(np.array([bound_y,bound_x]).T - curve_right, axis=1)
        top_left = []
        top_right = []
        bottom_left = []
        bottom_right = []
        # find top left
        for i in range(len(bound_x)):
            if bound_x[i] == np.ceil(curve_left[np.argmin(dists_left_y)][1]): #or bound_y[i] == np.floor(curve_left[np.argmin(dists_left_y)][1]):
                top_left = [bound_y[i],bound_x[i]]
                break
        for i in range(len(bound_x)-1,0,-1):
            if bound_x[i] == np.ceil(curve_left[np.argmin(dists_left_y)][1]): #or  bound_y[i] == np.floor(curve_left[np.argmin(dists_left_y)][1]):
                top_right = [bound_y[i],bound_x[i]]
                break
        #search for the bottom left corner closest to the points on left curve
        for i in range(len(bound_x)):
            if bound_x[i] == np.ceil(curve_left[np.argmax(dists_left_y)][1]): #or bound_x[i] == np.floor(curve_left[np.argmax(dists_left_y)][1]):
                bottom_left = [bound_y[i],bound_x[i]]
                break
        for i in range(len(bound_x)-1,0,-1):
            if bound_x[i] == np.ceil(curve_left[np.argmax(dists_left_y)][1]): #or bound_x[i] == np.floor(curve_left[np.argmax(dists_left_y)][1]):
                bottom_right = [bound_y[i],bound_x[i]]
                break
        return top_left, top_right, bottom_left, bottom_right

    def find_four_corners_v1(self, mask):
        #in the image x is y and y is x
        #find the four corners of the mask
        #find the top left corner
        #find the top right corner
        #find the bottom left corner
        #find the bottom right corner
        #return the four corners
        bound_x, bound_y = np.where(mask == 1)
        bound_low_y = (bound_x)[-500]
        bound_high_y = (bound_x)[500]
        #search for the top left corner
        # #first occurence is top left
        # last occurence is top right
        for i in range(len(bound_x)):
            if bound_x[i] == bound_high_y:
                top_left = [bound_y[i],bound_x[i]]
                break
        for i in range(len(bound_x)-1,0,-1):
            if bound_x[i] == bound_high_y:
                top_right = [bound_y[i],bound_x[i]]
                break
        #search for the bottom left corner
        #first occurence is bottom left
        #last occurence is bottom right
        for i in range(len(bound_x)):
            if bound_x[i] == bound_low_y:
                bottom_left = [bound_y[i],bound_x[i]]
                break
        for i in range(len(bound_x)-1,0,-1):
            if bound_x[i] == bound_low_y:
                bottom_right = [bound_y[i],bound_x[i]]
                break
        return top_left, top_right, bottom_left, bottom_right

    def fit_circle_to_3d_points(self, points):
        """using ransac fit a circle to the points"""

        # fit circle to every z_slice
        x_data = points[:, 0]
        y_data = points[:, 1]
        z_data = points[:, 2]
        # plt.ion()
        # plt.show()
        running_mean_raduis = []

        # fig = plt.figure()
        # ax1 = fig.add_subplot(1, 2, 1)
        # ax2 = fig.add_subplot(1, 2, 2, projection='3d')


        # ax.scatter(x_data_slice, y_data_slice, z_slice)
        for z_slice in np.unique(z_data):
            x_data_slice = x_data[z_data == z_slice]
            y_data_slice = y_data[z_data == z_slice]
            # print(len(x_data_slice)," ",z_slice)
            if(len(x_data_slice)>300):
                try:
                    # plt.cla()
                    # fit circle to every z_slice
                    xc, yc, r1, sigma = taubinSVD(list(zip(x_data_slice, y_data_slice)))

                    # ransac = RANSAC(x_data_slice, y_data_slice, 50)
                    # ransac.execute_ransac()
                    # a, b, r = ransac.best_model[0], ransac.best_model[1], ransac.best_model[2]

                    # show result
                    # circle = plt.Circle((xc, yc), radius=r1, color='r', fc='y', fill=False)
                    # ax1.add_patch(circle)
                    # ax1.scatter(x_data_slice, y_data_slice, s=1)
                    # ax1.axis('scaled')
                    # ax2.scatter(x_data_slice, y_data_slice, z_slice)
                    # ax2.set_xlabel('$X$')
                    # ax2.set_ylabel('$Y$')
                    # ax2.set_zlabel('$Z$')
                    # plt.pause(0.1)
                    # plt.draw()
                    running_mean_raduis.append(r1)
                    #plot in 3d for debugging

                except np.linalg.LinAlgError:
                    print("Singular Matrix")
                    continue

        median_radius = np.median(running_mean_raduis)
        # running_mean_raduis = running_mean_raduis/len(np.unique(z_data))
        print("running_mean_raduis: ",np.mean(running_mean_raduis), " median_radius: ",median_radius)
        return median_radius

    def read_images(self, path, image_name, b_output_debug=True, b_recalc=False):
        """ Read in all of the mask, rgb, flow images
        If Edge image does not exist, create it
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @returns dictionary of images with image_type as keywords """

        images = {}
        # search_path = f"{path}{image_name}_*.npy"
        # fnames = glob(search_path)
        # if fnames is None:
        #     raise ValueError(f"No files in directory {search_path}")
        # Setup:
        f2 = '/home/josyula/Documents/DataAndModels/3AprilTrees_D435i/20230403_145650.bag'
        path_calculated = path + "CalculatedData/"

        pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device_from_file(f2)
        profile = pipe.start(cfg)
        branch = MakeTreeGeometry("data")
        radmaxs = []
        radmins = []
        rad_picos = []
        radius_from_ransac = []
        for i in range(1000):
            frameset = pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()
            color = np.asanyarray(color_frame.get_data())
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
            ts = frameset.get_timestamp()
            image_name = str(ts).split('.')[0]
            print(image_name)
            # if image_name == str(1680559045964):

            colorizer = rs.colorizer()
            align = rs.align(rs.stream.color)
            frameset = align.process(frameset)
            depth_frame = frameset.get_depth_frame()
            # Update color and depth frames:
            # colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            path_temp = tempfile.mkdtemp("masks")
            search_path = f"{path}{image_name}_*.npy"
            fnames = glob(search_path)
            i=0
            for n in fnames:
                images[image_name+"_np_mask"+"_{0}".format(i)] = np.load(n)
                images["RGB0"] = color
                images[image_name+"_np_mask"+"_{0}".format(i)] = self.depth_mask(images["RGB0"], images[image_name+"_np_mask"+"_{0}".format(i)], depth_frame, intrinsics)
                cv2.imwrite(path+"/RGB"+"/{0}.jpg".format(image_name), images["RGB0"])
                cv2.imwrite(path_temp+"/mask.jpg", images[image_name+"_np_mask"+"_{0}".format(i)]*255)
                images["Mask"] = cv2.imread(path_temp+"/mask.jpg")
                images["Mask"] = cv2.cvtColor(images["Mask"], cv2.COLOR_BGR2GRAY)
                images["Edge"] = cv2.Canny(images["Mask"], 50, 150, apertureSize=3)
                # bound_x , bound_y = np.where(images[image_name+"_np_mask"+"_{0}".format(i)].astype(int)==1) # np.where(images["Edge"] == 255)
                # bound_low_x1 = min(bound_y)
                # bound_low_x2 = min(bound_y)
                # bound_low_y1 = min(bound_x[bound_y==bound_low_x1])
                # bound_low_y2 = max(bound_x[bound_y==bound_low_x2])
                #
                # bound_high_x1 = max(bound_y)
                # bound_high_x2 = max(bound_y)
                # bound_high_y1 = min(bound_x[bound_y==bound_high_x1])
                # bound_high_y2 = max(bound_x[bound_y==bound_high_x2])

                cv2.imwrite(path + image_name + "_edge.png", images["Edge"])
                self.images = images
                self.name = image_name
                self.vertical_leader_masks = self.split_mask(self.images["Mask"], b_one_mask=True,
                                                             b_debug=b_output_debug)
                self.vertical_leader_stats = []
                self.vertical_leader_quads = []
                #extract 3d points for all points in the depth_mask


                all_3d_points = []
                for k in range(images[image_name+"_np_mask"+"_{0}".format(i)].shape[0]):
                    for l in range(images[image_name+"_np_mask"+"_{0}".format(i)].shape[1]):
                        if images[image_name+"_np_mask"+"_{0}".format(i)][k,l] == 1:
                            depth = depth_frame.get_distance(l,k)
                            if depth != 0:
                                pixel = rs.rs2_deproject_pixel_to_point(intrinsics, [l,k], depth)
                                all_3d_points.append(pixel)
                all_3d_points = np.array(all_3d_points)
                r_ransac = self.fit_circle_to_3d_points(all_3d_points)
                radius_from_ransac.append(r_ransac)

                # For each component of the mask image, calculate or read in statistics (center, eigen vectors)
                print("Calculating stats")
                for j, mask in enumerate(self.vertical_leader_masks):
                    fname_stats = path_calculated + self.name + f"_mask_{j}.json"
                    if b_recalc or not exists(fname_stats):
                        stats_dict = self.stats_image(self.images["Mask"], mask)
                        for k, v in stats_dict.items():
                            try:
                                if v.size == 2:
                                    stats_dict[k] = [v[0], v[1]]
                            except:
                                pass
                        # If this fails, make a CalculatedData and DebugImages folder in the data/forcindy folder
                        with open(fname_stats, 'w') as f:
                            json.dump(stats_dict, f)
                    elif exists(fname_stats):
                        with open(fname_stats, 'r') as f:
                            stats_dict = json.load(f)

                    for k, v in stats_dict.items():
                        try:
                            if len(v) == 2:
                                stats_dict[k] = np.array([v[0], v[1]])
                        except:
                            pass
                    self.vertical_leader_stats.append(stats_dict)

                # For each of the masks, see if we have reasonable stats
                #   Save points in debug image
                if b_output_debug:
                    self.images["Mask_Stats"] = np.copy(self.images["Mask"])
                    self.images["RGB_Stats"] = np.copy(self.images["RGB0"])
                    for i, stats in enumerate(self.vertical_leader_stats):
                        self.images["Mask_Stats"] = self.images["Mask"] / 2
                        try:
                            p1 = stats["lower_left"]
                            p2 = stats["upper_right"]
                            self.images["Mask_Stats"][self.vertical_leader_masks[i]] = 255
                            draw_line(self.images["RGB_Stats"], p1, p2, (128, 128, 128), 2)
                            draw_line(self.images["Mask_Stats"], p1, p2, (128, 128, 128), 1)

                            pc = ["center"]
                            draw_cross(self.images["RGB_Stats"], pc, (128, 128, 128), 1, 2)
                            draw_cross(self.images["Mask_Stats"], pc, (180, 180, 128), 1, 3)

                        except:
                            pass

                        cv2.imwrite(self.path_debug + image_name + "_" + f"{i}_mask_points.png",
                                    self.images["Mask_Stats"])
                    # cv2.imwrite(self.path_debug + image_name + "_" + "rgb_points.png", self.images["RGB_Stats"])

                # Fit a quad to each vertical leader
                print("Fitting quads")
                for j, stats in enumerate(self.vertical_leader_stats):
                    print(f"  {image_name}, mask {j}")
                    image_mask = np.zeros(self.images["Mask"].shape, dtype=self.images["Mask"].dtype)
                    fname_quad = path_calculated + self.name + "_" + image_name + f"_{j}_quad.json"
                    fname_params = path_calculated + self.name + "_" + image_name + f"_{j}_quad_params.json"
                    quad = None
                    if exists(fname_quad) and not b_recalc:
                        quad = Quad([0, 0], [1, 1], 1)
                        quad.read_json(fname_quad)
                        with open(fname_params, 'r') as f:
                            params = json.load(f)
                    else:
                        image_mask[self.vertical_leader_masks[i]] = 255
                        quad, params = self.fit_quad(image_mask, pts=stats, b_output_debug=b_output_debug, quad_name=i)
                        quad.write_json(fname_quad)
                        with open(fname_params, 'w') as f:
                            json.dump(params, f)
                    self.vertical_leader_quads.append(quad)

                    if b_output_debug:
                        # Draw the edge and original image with the fitted quad and rects
                        im_covert_back = cv2.cvtColor(self.images["Edge"], cv2.COLOR_GRAY2RGB)
                        im_orig_debug = np.copy(self.images["RGB0"])

                        # Draw the original, the edges, and the depth mask with the fitted quad
                        quad.draw_quad(im_orig_debug)
                        if quad.is_wire():
                            draw_cross(im_orig_debug, quad.p0, (255, 0, 0), thickness=2, length=10)
                            draw_cross(im_orig_debug, quad.p2, (255, 0, 0), thickness=2, length=10)
                        else:
                            quad.draw_boundary(im_orig_debug, 10)
                            quad.draw_edge_rects(im_covert_back, step_size=params["step_size"],
                                                 perc_width=params["width"])

                        im_both = np.hstack([im_orig_debug, im_covert_back])
                        cv2.imshow("Original and edge and depth", im_both)
                        cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_quad.png", im_both)

                        quad.draw_quad(self.images["RGB_Stats"])
                    test = []
                    test_l = []
                    test_r = []
                    for t in np.arange(0, 1, 0.1 ):
                        pix1 = np.ceil(quad.pt_axis(t)).clip(0, max=np.array([720,1280])).astype(int)
                        point1 = rs.rs2_deproject_pixel_to_point(intrinsics, pix1, depth_frame.get_distance(*pix1))
                        # print(point1)
                        pix_edge = np.ceil(quad.edge_pts(t)).clip(0, max=np.array([720,1280])).astype(int)
                        pixl = [pix_edge[0][0], pix_edge[0][1]]
                        pixr = [pix_edge[1][0], pix_edge[1][1]]
                        pointl = rs.rs2_deproject_pixel_to_point(intrinsics, pixl, depth_frame.get_distance(*pixl))
                        pointr = rs.rs2_deproject_pixel_to_point(intrinsics, pixr, depth_frame.get_distance(*pixr))
                        test_l.append(pointl)
                        test_r.append(pointr)
                        test.append(point1)

                        if any(np.isnan(point1)):
                            break
                    if len(test) != 10:
                        break
                # for quad in self.vertical_leader_quads:
                #     score = self.score_quad(quad)
                #     print(f"Score {score}")
                # get indices on the ledft and right test_l, test_r where the points are not zero
                test_l = np.array(test_l)
                test_r = np.array(test_r)
                test = np.array(test)
                ind1 = np.where(test_l[:, 0] != 0)
                ind2 = np.where(test_r[:, 0] != 0)
                set_ind = set(ind1[0]).intersection(set(ind2[0]))
                ind = np.array(list(set_ind))
                rad_min = np.linalg.norm(test_l[ind[0]] - test_r[ind[0]])
                rad_max = np.linalg.norm(test_l[ind[-1]] - test_r[ind[-1]])
                points_on_axis = [test[0], test[int(len(test)/2)], test[-1]]
                ##alternate radius calculation
                #3d coordinates of bound_low
                mask_b = images[image_name + "_np_mask" + "_{0}".format(i)]
                top_left, top_right, bottom_left, bottom_right = self.find_four_corners_v1(mask_b)
                # top_left, top_right, bottom_left, bottom_right = self.find_four_corners_v2(mask_b, quad)
                try:
                    bound_low_left = rs.rs2_deproject_pixel_to_point(intrinsics, [bottom_left[0], bottom_left[1]], depth_frame.get_distance(bottom_left[0], bottom_left[1]))
                    bound_low_right = rs.rs2_deproject_pixel_to_point(intrinsics, [bottom_right[0], bottom_right[1]], depth_frame.get_distance(bottom_right[0], bottom_right[1]))
                    bound_high_left = rs.rs2_deproject_pixel_to_point(intrinsics, [top_left[0], top_left[1]], depth_frame.get_distance( top_left[0], top_left[1]))
                    bound_high_right = rs.rs2_deproject_pixel_to_point(intrinsics, [top_right[0], top_right[1]], depth_frame.get_distance(top_right[0], top_right[1]))

                    rad_min_alt = np.linalg.norm(np.array(bound_low_left) - np.array(bound_low_right))
                    rad_max_alt = np.linalg.norm(np.array(bound_high_left) - np.array(bound_high_right))
                    rad_min = min(rad_min, rad_min_alt)
                    rad_max = min(rad_max, rad_max_alt)
                    print(f"rad_min {rad_min}, rad_max {rad_max}")
                    radmins.append(rad_min)
                    radmaxs.append(rad_max)
                    depths = self.get_depths(images["RGB0"], depth_frame)
                    tw = TrunkWidthEstimator()
                    rad_pico = tw.get_width(images["RGB0"], depths, images[image_name + "_np_mask" + "_{0}".format(i)] )
                    print(f"rad_pico {rad_pico}")
                    rad_picos.append(rad_pico)
                    #WRITE BRANCH TO FILE
                    branch.make_branch_segment(points_on_axis[0], points_on_axis[1],
                                               points_on_axis[2],
                                               radius_start=rad_min, radius_end=rad_max,
                                               start_is_junction=True, end_is_bud=False)
                    tw = TrunkWidthEstimator()
                    vertex_locs = branch.write_mesh(self.path_debug + "_" + image_name + f"_{i}_branch.obj")
                    #project the vertices onto the image and draw them
                    vertex_2d = []
                    for v in vertex_locs:
                        for v1 in v:
                            pix = rs.rs2_project_point_to_pixel(intrinsics, v1)
                            pix = np.array(pix).clip(0, max=np.array([720,1280])).astype(int)
                            cv2.circle(self.images["RGB_Stats"], (int(pix[0]), int(pix[1])), 5, (0, 0, 255), -1)
                            vertex_2d.append([int(pix[0]), int(pix[1])])
                    # cv2.imshow("RGB_Stats", self.images["RGB_Stats"])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    cv2.imwrite(self.path_debug + self.name + "_" + image_name + f"_{i}_branch.png", self.images["RGB_Stats"])
                    os.remove(path_temp+"/mask.jpg")
                    #MAKE ANNONATION
                    self.make_image_annotation(vertex_2d, image_name, path, class_id=i)
                    i += 1
                except:
                    import sys
                    print("Error in branch detection - no depth data for branch ", image_name)
                    print("error type: ", sys.exc_info()[0])
                    # os.remove(path_temp+"/mask.jpg")
                    continue
        pipe.stop()
        radmins = np.array(radmins)
        radmaxs = np.array(radmaxs)
        rad_picos = np.array(rad_picos)
        avg_rad_min = np.mean(radmins)
        avg_rad_max = np.mean(radmaxs)
        avg_rad_pico = np.mean(rad_picos)
        print(f"avg_rad_min {avg_rad_min}, avg_rad_max {avg_rad_max}")
        np.save(self.path_debug + self.name + "_radmins.npy", radmins)
        np.save(self.path_debug + self.name + "_radmaxs.npy", radmaxs)
        radactual = 15.34
        #plot error bar with radmin
        import seaborn as sns
        import pandas as pd
        #make dataframe
        df = pd.DataFrame({'radmin': radmins, 'radmax': radmaxs, 'rad_pico': rad_picos})
        sns.set()
        #histogram of radmin
        sns.distplot(df['radmin'], bins=3, kde=False, rug=True)

        plt.hist(radmins, bins=3, color='b', label='radmin')
        plt.errorbar(np.arange(len(radmins)), radmins, yerr=radactual-radmins, fmt='o')

        #plot bar graph of radii vs actual radii
        plt.bar(np.arange(len(radmins)), radmins, width=0.2, color='b', align='center', label='radmin')
        #plot histogram of radii
        plt.hist(radmins, bins=10, color='b', label='radmin')
        #plot actual as x-axis
        plt.plot([0, len(radmins)], [radactual, radactual], color='r', label='actual')
        return images
    def make_image_annotation(self, vertex_locs, image_name, path, class_id=0):
        #make coco annotation from vertices on boundary of branch
        #get vertices on boundary of branch
        classes = ["leader", "side_branch"]
        i = class_id
        annotations = []
        img_mask = np.zeros((720, 1280))
        for v in vertex_locs:
            img_mask[v[1], v[0]] = 1

        img_mask += self.images[image_name + "_np_mask" + "_{0}".format(i)]
        img_mask = np.clip(img_mask, 0, 1)
        img_mask= img_mask*255
        img_mask= img_mask.astype(np.uint8)

        #reduce noise in mask
        kernel = np.ones((3, 3), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=1)
        # cv2.imshow("eroded", img_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #dilate the mask to close gaps
        kernel = np.ones((5, 5), np.uint8)
        img_mask = cv2.dilate(img_mask, kernel, iterations=10)

        cv2.imwrite(self.path_debug + image_name + f"_{i}seg_mask.jpg", img_mask)
        # cv2.imshow("dilated", img_mask)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        #build annonation from image mask
        segmentation = []
        for i in range(720):
            for j in range(1280):
                if img_mask[i, j] == 255:
                    segmentation+=[j, i]

        segmentation = [segmentation]
        #find bounding box using opencv
        _, contours, _ = cv2.findContours(img_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rect = cv2.minAreaRect(contours[0])
        (x, y), (w, h), a = rect

        # box = cv2.boxPoints(rect)
        # box = np.intp(box)  # turn into ints
        # box = np.clip(box, 0, 719)
        # box = np.clip(box, 0, 1279)

        bbox = [int(x), int(y), int(w), int(h)]



        # cv2.imshow("RGB_Stats", self.images["RGB_Stats"])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        image_dict = {
            "id": 0,
            "license": 1,
            "file_name": "",
            "height": 720,
            "width": 1280,
            "date_captured": "2022-07-20T19:39:26+00:00",
            "annotated": "true",
            "category_ids": [0,1],
            "metadata": {}
        }

        image_dict.update({'file_name': image_name, 'id': image_name, "path": path+"/RGB"+"/{0}.jpg".format(image_name)})

        annotation = {
            "segmentation": segmentation,
            "area": w*h,
            "iscrowd": 0,
            "image_id": image_name,
            "bbox": bbox,
            "category_id": i,
            "id": i
        }

        json_dict["images"] = image_dict
        json_dict["annotations"] = annotation
        json_dict["categories"] = {"id": i, "name": classes[i], "supercategory": "branch"}

        with open(self.path_debug + image_name + "_annotation.json", 'w') as outfile:
            json.dump(json_dict, outfile)



    def depth_mask(self, image, mask, depth_frame, intrinsics):
        depth = []
        points = np.where(mask == 1)
        points = zip(points[0], points[1])
        mask_depth = np.zeros_like(mask)
        for point in points:
            if depth_frame.get_distance(point[1], point[0]) != 0:
                depth.append(depth_frame.get_distance(point[1], point[0]))
                mask_depth[point[0], point[1]] = 255
        return mask_depth

    def get_depths(self, image, depth_frame):
        depths = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if depth_frame.get_distance(j, i) != 0:
                    depths[i, j] = depth_frame.get_distance(j, i)
        return depths

    def split_mask(self, in_im_mask, b_one_mask=True, b_debug=False):
        """Split the mask image up into connected components, discarding anything really small
        @param in_im_mask - the mask image
        @param b_debug - print out mask labeled image
        @return a list of boolean indices for each component"""
        output = cv2.connectedComponentsWithStats(in_im_mask)
        labels = output[1]
        stats = output[2]

        ret_masks = []
        i_widest = 0
        i_area = 0
        for i, stat in enumerate(stats):
            if np.sum(in_im_mask[labels == i]) == 0:
                continue

            if stat[cv2.CC_STAT_WIDTH] < 5:
                continue
            if stat[cv2.CC_STAT_HEIGHT] < 0.5 * in_im_mask.shape[1]:
                continue
            if i_area < stat[cv2.CC_STAT_AREA]:
                i_widest = len(ret_masks)
                i_area = stat[cv2.CC_STAT_AREA]
            ret_masks.append(labels == i)

        if b_debug:
            labels = 128 + labels * (120 // output[0])
            cv2.imwrite(self.path_debug + self.name + "_" + "labels.png", labels)

        try:
            if b_one_mask:
                return [ret_masks[i_widest]]
        except:
            pass
        return ret_masks

    def stats_image(self, in_im, pixs_in_mask):
        """ Add statistics (bounding box, left right, orientation, radius] to image
        Note: Could probably do this without transposing image, but       @param im image
        @returns stats as a dictionary of values"""

        LeaderDetector._init_grid_(in_im)

        xs = LeaderDetector._x_grid[pixs_in_mask]
        ys = LeaderDetector._y_grid[pixs_in_mask]

        stats = {}
        stats["x_min"] = np.min(xs)
        stats["y_min"] = np.min(ys)
        stats["x_max"] = np.max(xs)
        stats["y_max"] = np.max(ys)
        stats["x_span"] = stats["x_max"] - stats["x_min"]
        stats["y_span"] = stats["y_max"] - stats["y_min"]

        avg_width = 0.0
        count_width = 0
        if stats["x_span"] > stats["y_span"]:
            stats["Direction"] = "left_right"
            stats["Length"] = stats["x_span"]
            for r in range(0, LeaderDetector._width):
                if sum(pixs_in_mask[:, r]) > 0:
                    avg_width += sum(pixs_in_mask[:, r] > 0)
                    count_width += 1
        else:
            stats["Direction"] = "up_down"
            stats["Length"] = stats["y_span"]
            for c in range(0, LeaderDetector._height):
                if sum(pixs_in_mask[c, :]) > 0:
                    avg_width += sum(pixs_in_mask[c, :] > 0)
                    count_width += 1
        stats["width"] = avg_width / count_width
        stats["center"] = np.array([np.mean(xs), np.mean(ys)])

        x_matrix = np.zeros([2, xs.shape[0]])
        x_matrix[0, :] = xs.transpose() - stats["center"][0]
        x_matrix[1, :] = ys.transpose() - stats["center"][1]
        covariance_matrix = np.cov(x_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        if eigen_values[0] < eigen_values[1]:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[1, :]
        else:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[0, :]
        eigen_ratio = stats["EigenValues"][1] / stats["EigenValues"][0]
        stats["EigenVector"][1] *= -1
        stats["EigenRatio"] = eigen_ratio
        stats["lower_left"] = stats["center"] - stats["EigenVector"] * (stats["Length"] * 0.5)
        stats["upper_right"] = stats["center"] + stats["EigenVector"] * (stats["Length"] * 0.5)
        # print(stats)
        # print(f"Eigen ratio {eigen_ratio}")
        return stats

    def fit_quad(self, im_mask, pts, b_output_debug=True, quad_name=0):
        """ Fit a quad to the mask, edge image
        @param im_mask - the image mask
        @param pts - the stats from the stats call
        @param b_output_debug - output mask with quad at the intermediate step
        @returns fitted quad"""

        # Fit a quad to the trunk
        pt_lower_left = pts['center']
        vec_len = pts["Length"] * 0.4
        while pt_lower_left[0] > 2 + pts['x_min'] and pt_lower_left[1] > 2 + pts['y_min']:
            pt_lower_left = pts["center"] - pts["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        pt_upper_right = pts['center']
        vec_len = pts["Length"] * 0.4
        while pt_upper_right[0] < -2 + pts['x_max'] and pt_upper_right[1] < -2 + pts['y_max']:
            pt_upper_right = pts["center"] + pts["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        quad = Quad(pt_lower_left, pt_upper_right, 0.5 * pts['width'])

        # Current parameters for the vertical leader
        params = {"step_size": int(quad.radius_2d * 1.5), "width_mask": 1.4, "width": 0.25}

        # Iteratively move the quad to the center of the mask
        for i in range(0, 5):
            res = quad.adjust_quad_by_mask(im_mask,
                                           step_size=params["step_size"], perc_width=params["width_mask"],
                                           axs=None)
            # print(f"Res {res}")

        if b_output_debug:
            im_debug = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
            quad.draw_quad(im_debug)
            quad.draw_boundary(im_debug)
            cv2.imwrite(self.path_debug + self.name + "_" + self.name + f"_{quad_name}_quad_fit_mask.png", im_debug)

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = quad.adjust_quad_by_hough_edges(self.images["Edge"], step_size=params["step_size"], perc_width=params["width"], axs=None)
            # print(f"Res Hough {ret}")

        return quad, params

    def flow_mask(self, quad):
        """ Use the fitted quad and the original mask to extract a better mask from the flow image
        @param quad - the quad we've fitted so far
        @return im_mask - a better image mask"""
        im_flow = self.images["Flow"]

        im_inside = quad.interior_rects_mask((im_flow.shape[0], im_flow.shape[1]), step_size=30, perc_width=0.5)
        im_inside = im_inside.reshape((im_flow.shape[0] * im_flow.shape[1]))
        im_flow_reshape = im_flow.reshape((im_flow.shape[0] * im_flow.shape[1], 3))
        n_inside = np.count_nonzero(im_inside)
        n_total = im_flow.shape[0] * im_flow.shape[1]
        im_flow_whiten = whiten(im_flow_reshape)
        color_centers = kmeans(im_flow_whiten, 4)

        pixel_labels = vq(im_flow_whiten, color_centers[0])
        label_count = [(np.count_nonzero(np.logical_and(pixel_labels[0] == i, im_inside == True)), i) for i in range(0, 4)]
        label_count.sort()

        im_mask_labels = np.zeros(im_inside.shape, dtype=im_flow.dtype)
        im_mask = np.zeros(im_inside.shape, dtype=im_flow.dtype)
        n_div = 125 // 3
        for i, label in enumerate(label_count):
            im_mask_labels[np.logical_and(pixel_labels[0] == label[1], im_inside == True)] = 125 + int(i * n_div)
            im_mask_labels[np.logical_and(pixel_labels[0] == label[1], im_inside == False)] = int(i * n_div)
        im_mask[pixel_labels[0] == label_count[-1][1]] = 255
        return im_mask.reshape((im_flow.shape[0], im_flow.shape[1])), im_mask_labels.reshape((im_flow.shape[0], im_flow.shape[1]))

    def score_quad(self, quad):
        """ See if the quad makes sense over the optical flow image
        @quad - the quad
        """

        # Two checks: one, are the depth/optical fow values largely consistent under the quad center
        #  Are there boundaries in the optical flow image where the edge of the quad is?
        im_flow_mask = cv2.cvtColor(self.images["Flow"], cv2.COLOR_BGR2GRAY)
        perc_consistant, stats_slice = quad.check_interior_depth(im_flow_mask)

        diff = 0
        for i in range(1, len(stats_slice)):
            diff_slices = np.abs(stats_slice[i]["Median"] - stats_slice[i-1]["Median"])
            if diff_slices > 20:
                print(f"Warning: Depth values not consistant from slice {self.name} {i} {stats_slice}")
            diff += diff_slices
        if perc_consistant < 0.9:
            print(f"Warning: not consistant {self.name} {stats_slice}")
        return perc_consistant, diff / (len(stats_slice) - 1)


if __name__ == '__main__':
    # # path = "./data/predictions/"
    path = "/home/josyula/Documents/DataAndModels/3AprilTrees_D435i/numpy_masks/"
    # #path = "./forcindy/"
    # # for im_i in range(0, 49):
    name = "1"
    # print(name)
    bp = LeaderDetector(path, name, b_output_debug=True, b_recalc=True)



####
# images[image_name + "_np_mask" + "_{0}".format(i)] = np.load(n)
# cv2.imwrite(path_temp + "/mask.jpg", images[image_name + "_np_mask" + "_{0}".format(i)] * 255)
# images[image_name + "_Mask"] = cv2.imread(path_temp + "/mask.jpg")
# images[image_name + "_RGB0"] = color
# images[image_name + "_Edge"] = cv2.Canny(images[image_name + "_Mask"], 50, 150, apertureSize=3)
# cv2.imwrite(path + image_name + "_edge.png", images[image_name + "_Edge"])
