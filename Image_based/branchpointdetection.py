#!/usr/bin/env python3

# Read in masked images and estimate points where a side branch joins a leader (trunk)

import numpy as np
from glob import glob
import csv
import cv2
import json
from os.path import exists
from cyl_fit_2d import Quad
from line_seg_2d import draw_line, draw_box, draw_cross, LineSeg2D
from scipy.cluster.vq import kmeans, whiten, vq

class BranchPointDetection:
    def __init__(self, path, image_name, b_output_debug=True, b_recalc=False):
        """ Detect possible branch points where side branch touches trunk
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @param b_recalc: Force recalculate the result, y/n"""

        self.path_debug = path + "DebugImages/"
        path_calculated = path + "CalculatedData/"

        self.name = image_name
        # Read in all images that have name_ and are not debugging images
        self.images, self.images_single = self.read_images(path, image_name)
        # Keep the point/intersection info
        self.image_stats = {}

        # Vertical leaders and side branches off of them
        self.trunks = []
        self.sidebranches = []

        # For the trunks/side branches, calculate or read in statistics (center, eigen vectors)
        print("Calculating stats")
        for im in self.images:
            fname_stats = path_calculated + self.name + "_" + im["name"] + ".json"
            if b_recalc or not exists(fname_stats):
                stats_dict = self.stats_image(im["image"])
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
            im["stats"] = stats_dict

            print(f"  {im['name']}")
            if "trunk" in im["name"]:
                self.trunks.append(im)
            elif "sidebranch" in im["name"]:
                self.sidebranches.append(im)

        # For each of the trunk/branches, see if we have reasonable upper left/lower right points
        #   Save points in debug image
        if b_output_debug:
            self.images_single["marked points"] = np.copy(self.images_single["masked"])
            for im in self.images:
                try:
                    p1 = im["stats"]["lower_left"]
                    p2 = im["stats"]["upper_right"]
                    draw_line(im["image"], p1, p2, (128, 128, 128), 2)
                    draw_line(self.images_single["marked points"], p1, p2, (128, 128, 128), 1)

                    pc = im["stats"]["center"]
                    draw_cross(im["image"], pc, (128, 128, 128), 1, 2)
                    draw_cross(self.images_single["marked points"], pc, (180, 180, 128), 1, 3)

                    cv2.imwrite(self.path_debug + image_name + "_" + im["name"] + "_points.png", im["image"])
                except:
                    pass

            cv2.imwrite(self.path_debug + image_name + "_" + "_marked_points.png", self.images_single["marked points"])

        # Fit a quad to each branch
        print("Fitting quads")
        for im in self.images:
            print(f"  {im['name']}")
            fname_quad = path_calculated + self.name + "_" + im["name"] + "_quad.json"
            fname_params = path_calculated + self.name + "_" + im["name"] + "_quad_params.json"
            if exists(fname_quad) and not b_recalc:
                im["quad"] = Quad([0, 0], [1,1], 1)
                im["quad"].read_json(fname_quad)
                with open(fname_params, 'r') as f:
                    params = json.load(f)
            else:
                im["quad"], params = self.fit_quad(im)
                im["quad"].write_json(fname_quad)
                with open(fname_params, 'w') as f:
                    json.dump(params, f)

            if b_output_debug:
                # Draw the edge and original image with the fitted quad and rects
                im_covert_back = cv2.cvtColor(self.images_single["edge"], cv2.COLOR_GRAY2RGB)
                im_orig_debug = np.copy(self.images_single["orig"])

                # Draw the original, the edges, and the depth mask with the fitted quad
                im["quad"].draw_quad(im_orig_debug)
                if im["quad"].is_wire():
                    draw_cross(im_orig_debug, im["quad"].p0, (255, 0, 0), thickness=2, length=10)
                    draw_cross(im_orig_debug, im["quad"].p2, (255, 0, 0), thickness=2, length=10)
                else:
                    im["quad"].draw_boundary(im_orig_debug, 10)
                    im["quad"].draw_edge_rects(im_covert_back, step_size=params["step_size"], perc_width=params["width"])

                im_both = np.hstack([im_orig_debug, im_covert_back])
                cv2.imshow("Original and edge and depth", im_both)
                cv2.imwrite(self.path_debug + self.name + "_" + im["name"] + "_quad.png", im_both)

                im["quad"].draw_quad(self.images_single["marked points"])

        # Use the flow image to make a better mask
        print("Quad in flow mask")
        for im in self.images:
            print(f"  {im['name']}")

            fname_quad_flow_mask = path_calculated + self.name + "_" + im["name"] + "_quad_flow_mask.png"
            if exists(fname_quad_flow_mask) and not b_recalc:
                im["flow_mask"] = cv2.cvtColor(cv2.imread(fname_quad_flow_mask), cv2.COLOR_BGR2GRAY)
            else:
                im["flow_mask"], im_flow_mask_labels = self.flow_mask(im)
                if b_output_debug:
                    cv2.imwrite(self.path_debug + self.name + "_" + im["name"] + "_quad_flow_labels.png", im_flow_mask_labels)
                cv2.imwrite(fname_quad_flow_mask, im["flow_mask"])

            fname_quad_flow = path_calculated + self.name + "_" + im["name"] + "_quad_flow.json"
            fname_params_flow = path_calculated + self.name + "_" + im["name"] + "_quad_params_flow.json"
            if exists(fname_quad_flow) and not b_recalc:
                im["quad_flow"] = Quad([0, 0], [1,1], 1)
                im["quad_flow"].read_json(fname_quad_flow)
                with open(fname_params_flow, 'r') as f:
                    params = json.load(f)
            else:
                im["quad_flow"], params = self.fit_quad_flow(im)
                im["quad_flow"].write_json(fname_quad_flow)
                with open(fname_params_flow, 'w') as f:
                    json.dump(params, f)

            if b_output_debug:
                # Draw the edge and original image with the fitted quad and rects
                im_covert_back = cv2.cvtColor(self.images_single["edge"], cv2.COLOR_GRAY2RGB)
                im_orig_debug = np.copy(self.images_single["orig"])

                # Draw the original, the edges, and the depth mask with the fitted quad
                im["quad_flow"].draw_quad(im_orig_debug)
                if im["quad_flow"].is_wire():
                    draw_cross(im_orig_debug, im["quad_flow"].p0, (255, 0, 0), thickness=2, length=10)
                    draw_cross(im_orig_debug, im["quad_flow"].p2, (255, 0, 0), thickness=2, length=10)
                else:
                    im["quad_flow"].draw_boundary(im_orig_debug, 10)
                    im["quad_flow"].draw_edge_rects(im_covert_back, step_size=params["step_size"], perc_width=params["width"])

                im_both = np.hstack([im_orig_debug, im_covert_back])
                cv2.imwrite(self.path_debug + self.name + "_" + im["name"] + "_quad_flow.png", im_both)

                im["quad_flow"].draw_quad(self.images_single["marked points"])

        # Now look for branch points
        fname_branch_pts = path_calculated + self.name + "_branches.csv"
        if b_recalc or not exists(fname_branch_pts):
            self.branch_points = []
            for im_trunk in self.trunks:
                for im_branch in self.sidebranches:
                    if not im_branch["quad"].is_wire():
                        bp = self.find_branch_point(im_trunk, im_branch)
                        if bp is not None:
                            self.branch_points.append(bp)

            with open(fname_branch_pts, 'w') as f:
                csv_file = csv.writer(f)
                data_row = ["x", "y", "vx", "vy"]
                csv_file.writerow(data_row)
                for p, v in self.branch_points:
                    data_row = [p[0], p[1], v[0], v[1]]
                    csv_file.writerow(data_row)
        else:
            bp = np.loadtxt(fname_branch_pts, delimiter=",", skiprows=1)
            self.branch_points = []
            try:
                for r in bp:
                    p = np.array([r[0], r[1]])
                    v = np.array([r[2], r[3]])
                    self.branch_points.append([p, v])
            except IndexError:
                p = np.array([bp[0], bp[1]])
                v = np.array([bp[2], bp[3]])
                self.branch_points.append([p, v])

        if b_output_debug:
            for p, v in self.branch_points:
                draw_box(self.images_single["marked points"], p, (254, 128, 254), 6)
                draw_line(self.images_single["marked points"], p, p + v, (128, 254, 254), 1)

            cv2.imwrite(self.path_debug + image_name + "_" + "_marked_joins_points.png", self.images_single["marked points"])

    def read_images(self, path, image_name):
        """ Read in all of the trunk/sidebranch/edge/depth/orig images, labeled by name
        If Edge image does not exist, create it
        Store all of the trunk/sidebranch images in the im data structure
        Store the others as just images
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @returns list of image, name pairs, as dictionaries (for branches), and a dictionary of images for others"""

        images = []
        search_path = f"{path}{image_name}_*.png"
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No files in directory {search_path}")

        images_single = {}
        images_single["orig"] = cv2.imread(f"{path}{image_name}.png")

        trunk_count = 0
        branch_count = 0
        for n in fnames:
            im = {}
            if "points" in n:
                continue
            elif "trunk" in n:
                im["name"] = "trunk" + str(trunk_count)
                im["image"] = cv2.cvtColor(cv2.imread(n), cv2.COLOR_BGR2GRAY)
                images.append(im)
                trunk_count += 1
            elif "branch" in n:
                im["name"] = "sidebranch" + str(branch_count)
                im["image"] = cv2.cvtColor(cv2.imread(n), cv2.COLOR_BGR2GRAY)
                images.append(im)
                branch_count += 1
            elif "edge" in n:
                im_edge_color = cv2.imread(n)
                images_single["edge"] = cv2.cvtColor(im_edge_color, cv2.COLOR_BGR2GRAY)
            elif "depth" in n:
                images_single["depth"] = cv2.imread(n)
            elif "flow" in n:
                images_single["flow"] = cv2.imread(n)
            elif "masked" in n:
                images_single["masked"] = cv2.imread(n)

        if "edge" not in images_single:
            im_gray = cv2.cvtColor(images_single["orig"], cv2.COLOR_BGR2GRAY)
            images_single["edge"] = cv2.Canny(im_gray, 50, 150, apertureSize=3)
            cv2.imwrite(path + image_name + "_edge.png", images_single["edge"])

        return images, images_single

    def stats_image(self, in_im):
        """ Add statistics (bounding box, left right, orientation, radius] to image
        Note: Could probably do this without transposing image, but...
        @param im image
        @returns stats as a dictionary of values"""
        im = in_im.transpose()

        width = im.shape[0]
        height = im.shape[1]

        y_grid, x_grid = np.meshgrid(np.linspace(0.5, height - 0.5, height), np.linspace(0.5, width -  0.5, width))

        xs = x_grid[im > 0]
        ys = y_grid[im > 0]

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
            for r in range(0, width):
                if sum(im[r, :]) > 0:
                    avg_width += sum(im[r, :] > 0)
                    count_width += 1
        else:
            stats["Direction"] = "up_down"
            stats["Length"] = stats["y_span"]
            for c in range(0, height):
                if sum(im[:, c]) > 0:
                    avg_width += sum(im[:, c] > 0)
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
        print(stats)
        print(f"Eigen ratio {eigen_ratio}")
        return stats

    def fit_quad(self, im, b_output_debug=True):
        """ Fit a quad to the mask, edge image
        @param im - the image and the stats
        @param b_output_debug - output mask with quad at the intermediate step
        @returns fitted quad"""

        # For the vertical leader...
        pts = im["stats"]

        # Fit a quad to the trunk
        quad = Quad(pts['lower_left'], pts['upper_right'], 0.5 * pts['width'])

        # Current parameters for the vertical leader
        params = {"step_size": int(quad.radius_2d * 1.5), "width_mask": 1.4, "width": 0.3}

        # Iteratively move the quad to the center of the mask
        for i in range(0, 5):
            res = quad.adjust_quad_by_mask(im["image"],
                                           step_size=params["step_size"], perc_width=params["width_mask"],
                                           axs=None)
            print(f"Res {res}")

        if b_output_debug:
            im_debug = cv2.cvtColor(im["image"], cv2.COLOR_GRAY2RGB)
            quad.draw_quad(im_debug)
            quad.draw_boundary(im_debug)
            cv2.imwrite(self.path_debug + self.name + "_" + im["name"] + "_quad_fit_mask.png", im_debug)

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = quad.adjust_quad_by_hough_edges(self.images_single["edge"], step_size=params["step_size"], perc_width=params["width"], axs=None)
            print(f"Res Hough {ret}")

        return quad, params

    def fit_quad_flow(self, im, b_output_debug=True):
        """ Fit a quad to the mask, edge image
        @param im - the image and the stats
        @param b_output_debug - output mask with quad at the intermediate step
        @returns fitted quad"""

        # Fit a quad to the trunk
        quad = Quad(im["quad"].p0, im["quad"].p2, im["quad"].radius_2d, mid_pt=im["quad"].p1)

        # Current parameters for the vertical leader
        params = {"step_size": int(quad.radius_2d * 1.5), "width_mask": 1.4, "width": 0.3}

        # Iteratively move the quad to the center of the mask
        for i in range(0, 5):
            res = quad.adjust_quad_by_mask(im["flow_mask"],
                                           step_size=params["step_size"], perc_width=params["width_mask"],
                                           axs=None)
            print(f"Res {res}")

        if b_output_debug:
            im_debug = cv2.cvtColor(im["flow_mask"], cv2.COLOR_GRAY2RGB)
            quad.draw_quad(im_debug)
            quad.draw_boundary(im_debug)
            cv2.imwrite(self.path_debug + self.name + "_" + im["name"] + "_quad_fit_mask_flow.png", im_debug)

        # Now do the hough transform - first draw the hough transform edges
        for i in range(0, 5):
            ret = quad.adjust_quad_by_hough_edges(self.images_single["edge"], step_size=params["step_size"], perc_width=params["width"], axs=None)
            print(f"Res Hough {ret}")

        return quad, params

    def flow_mask(self, im):
        """ Use the fitted quad and the original mask to extract a better mask from the flow image
        @param im - image data structure
        @param quad - the quad we've fitted so far
        @return im_mask - a better image mask"""
        im_flow = self.images_single["flow"]

        im_inside = im["quad"].interior_rects_mask((im_flow.shape[0], im_flow.shape[1]), step_size=30, perc_width=0.5)
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

    def find_branch_point(self, im_trunk, im_sidebranch):
        """ See if it makes sense to connect trunk to side branch
        @param im_trunk Trunk image and stats
        @param im_sidebranch Side branch image and stats
        @returns x,y location in image if connection, zero otherwise"""

        stats_trunk = im_trunk["stats"]
        stats_branch = im_sidebranch["stats"]

        if stats_trunk["EigenRatio"] < 50:
            print(f"Not a clean trunk {im_trunk['name']} {stats_trunk['EigenRatio']}")
            return None

        for end in ["lower_left", "upper_right"]:
            xy = stats_branch[end]

            l2 = np.sum((stats_trunk["upper_right"]-stats_trunk["lower_left"])**2)
            if abs(l2) < 0.0001:
                continue

            #The line extending the segment is parameterized as p1 + t (p2 - p1).
            #The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2

            #if you need the point to project on line extention connecting p1 and p2
            t = np.sum((xy - stats_trunk["lower_left"]) * (stats_trunk["upper_right"] - stats_trunk["lower_left"])) / l2

            #if you need to ignore if p3 does not project onto line segment
            if t > 1 or t < 0:
                print("   Not on trunk")
                continue

            l1 = LineSeg2D(stats_trunk["lower_left"], stats_trunk["upper_right"])
            l2 = LineSeg2D(stats_branch["lower_left"], stats_branch["upper_right"])
            pt_trunk = LineSeg2D.intersection(l1, l2)
            pt_trunk_proj = stats_trunk["lower_left"] + t * (stats_trunk["upper_right"] - stats_trunk["lower_left"])
            if pt_trunk is None:
                pt_trunk = stats_trunk["lower_left"] + t * (stats_trunk["upper_right"] - stats_trunk["lower_left"])

            dist_to_trunk = np.sqrt(np.sum((pt_trunk - xy)**2))
            print(f"Trunk {im_trunk['name']} branch {im_sidebranch['name']} dist {dist_to_trunk}, {stats_trunk['width']}")
            vec_to_trunk = xy - pt_trunk
            if 0.25 * stats_trunk["width"] < dist_to_trunk < 1.75 * stats_trunk["width"]:
                if "lower_left" in end:
                    if vec_to_trunk[0] * stats_branch["EigenVector"][0] + vec_to_trunk[1] * stats_branch["EigenVector"][1] > 0:
                        print("  lower left")
                        pt_join = pt_trunk + stats_branch["EigenVector"] * stats_trunk["width"] * 0.5
                        return pt_join, stats_branch["EigenVector"]
                    else:
                        print("   Pointing wrong way")
                else:
                    if vec_to_trunk[0] * stats_branch["EigenVector"][0] + vec_to_trunk[1] * stats_branch["EigenVector"][1] < 0:
                        print("  upper right")
                        pt_join = pt_trunk + stats_branch["EigenVector"] * stats_trunk["width"] * -0.5
                        return pt_join, -stats_branch["EigenVector"]
                    else:
                        print("   Pointing wrong way")

        print("")
        return None


if __name__ == '__main__':
    path = "./data/forcindy/"
    #path = "./forcindy/"
    for im_i in range(0, 18):
        name = str(im_i)
        print(name)
        bp = BranchPointDetection(path, name, b_output_debug=True)
