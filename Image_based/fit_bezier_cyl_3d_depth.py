#!/usr/bin/env python3

# Read in the depth image and the fitted bezier curve.
# Assumes one fitted 2d curve
#   Extract depth values
#     - Average values along spline cross section
#     - Returns 3D curve

import numpy as np
import cv2
import json
from os.path import exists
from FileNames import FileNames
from bezier_cyl_3d import BezierCyl3D
from fit_bezier_cyl_2d_edge import FitBezierCyl2DEdge
from split_masks import convert_jet_to_grey
from camera_projections import frustrum_matrix, from_image_to_box


class FitBezierCyl3dDepth:
    def __init__(self, fname_depth_image, fname_depth_data, crv_2d, params=None, fname_calculated=None, fname_debug=None, b_recalc=False):
        """ Read in the depth image or data (data preferred), grab the depth data under the 2d curve, then promote to 3d
        @param fname_depth_image: Depth image name (used if no depth data csv file)
        @param fname_depth_data: Depth data as a .csv file (assumes depth image and csv file same size/aspect ratio)
        @param crv_2d: 2d bezier curve
        @param params: Parameters for filtering the depth image - how finely to sample along the edge and how much to believe edge
           perc_width_depth - percent of width to use, should be 0.1 to 0.85
           perc_along_depth - take median of pixels from a perc of curve, should be 0.1 to 0.3
           camera_width_angle - angle in degrees, 45 for intel d45, etc
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _stats.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc. Set to None if no debug
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the stats - this also reads the image in
        self.crv_2d = crv_2d
        # Keep extracted depth values
        self.depth_values = []

        # Get depth image
        b_has_image = exists(fname_depth_image)
        b_has_data = exists(fname_depth_data)
        if b_has_image:
            self.depth_image = cv2.imread(fname_depth_image)
        if b_has_data:
            self.depth_data = np.loadtxt(fname_depth_data, dtype="float", delimiter=",")

        if b_has_image and b_has_data:
            assert self.depth_image.shape[0] == self.depth_data.shape[0]
            assert self.depth_image.shape[1] == self.depth_data.shape[1]
        elif b_has_data:
            self.depth_image = np.zeros((self.depth_data.shape[0], self.depth_data.shape[1]), dtype=np.uint8)
            d_min = np.min(np.min(self.depth_data))
            d_max = np.max(np.max(self.depth_data))
            self.depth_image = np.uint8(255 * (self.depth_data - d_min) / (d_max - d_min))
        elif b_has_image:        
            if len(self.depth_image.shape) == 3:
                # data * alpha + beta, beta = 0   convert to unsigned int
                # most maxed out 65535
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                self.depth_data = convert_jet_to_grey(self.depth_image) / 255.0
                self.depth_data = self.depth_data / 0.03
                #self.depth_image = mask_image_depth
                #self.depth_image = cv2.cvtColor(mask_image_depth, cv2.COLOR_BGR2GRAY)

        # Create the file names for the calculated data that we'll store (initial curve, curve fit to mask, parameters)
        if fname_calculated:
            self.fname_depth_stats = fname_calculated + "_depth_stats.json"
            self.fname_params = fname_calculated + "_depth_params.json"
            self.fname_crv_3d = fname_calculated + "_crv_3d.json"

        # Copy params used mask and add the new ones
        self.params = {}
        if params is not None:
            for k in params.keys():
                self.params[k] = params[k]
        if "perc_width_depth" not in self.params:
            self.params["perc_width_depth"] = 0.65
        if "perc_along_depth" not in self.params:
            self.params["perc_along_depth"] = 0.1
        if "camera_width_angle" not in self.params:
            self.params["camera_width_angle"] = 50

        # Get the raw edge data
        if b_recalc or not fname_calculated or not exists(self.fname_depth_stats):
            # Recalculate and write
            self.depth_stats = FitBezierCyl3dDepth.full_depth_stats(self.depth_data,
                                                                    self.crv_2d,
                                                                    self.params)

            # Write out the 3d bezier curve
            if fname_calculated:
                with open(self.fname_depth_stats, 'w') as f:
                    json.dump(self.depth_stats, f, indent=" ")
                with open(self.fname_params, 'w') as f:
                    json.dump(self.params, f, indent=" ")
        else:
            # Read in the stored data
            with open(self.fname_depth_stats, 'r') as f:
                self.depth_stats = json.load(f)
            with open(self.fname_params, 'r') as f:
                self.params = json.load(f)

        # Now use the params to filter the raw edge location data - produces the left, right edge curves
        if b_recalc or not fname_calculated or not exists(self.fname_crv_3d):
            # Recalculate and write
            self.crv_3d = FitBezierCyl3dDepth.curve_from_stats(self.depth_stats, self.crv_2d, self.params)
            if fname_calculated:
                with open(self.fname_crv_3d, 'w') as f:
                    self.crv_3d.write_json(self.fname_crv_3d)
        else:
            # Read in the reconstructed curve
            self.crv_3d = BezierCyl3D.read_json(self.fname_crv_3d, None, True)

        if fname_debug:
            # Draw the mask with the initial and fitted curve
            self.crv_3d.make_mesh()
            self.crv_3d.write_mesh(fname_debug + ".obj")
            print("To do")

    @staticmethod
    def full_depth_stats(depth_data, crv_2d, params):
        """ Get the best pixel offset (if any) for each point/pixel along the edge
        @param depth_data - the depth data as a numpy array
        @param crv_2d - the 2d curve
        @param params - parameters for conversion
        @return t, stats for depth, spaced n apart"""

        # Fuzzy rectangles along the boundary
        n_pixs = int(crv_2d.curve_length() * params["perc_along_depth"])
        rects, _ = crv_2d.interior_rects(step_size=n_pixs, perc_width=params["perc_width_depth"])

        ts = np.linspace(0, 1, len(rects) + 1)

        # Size of the rectangle(s) to cutout is based on the step size and the radius
        height = int(crv_2d.radius(0.5))
        width = n_pixs

        ret_stats = {"n_segs": len(ts) - 1,
                     "image_size": (depth_data.shape[1], depth_data.shape[0]),
                     "ts":[],
                     "z_at_center":[],
                     "radius_3d":[],
                     "divs": [],
                     "depth_divs":[],
                     "depth_values":[],
                     "r_at_depth":[],
                     "t_at_depth":[]}
        im_cutouts = []
        ts_image = []
        rs_image = []
        trans_back = []

        n_total_pixs = width * height
        divs = (0, n_total_pixs // 4, n_total_pixs // 2, 3 * n_total_pixs // 4, n_total_pixs-1)
        ret_stats["divs"] = divs

        rs_seg = np.linspace(-params["perc_width_depth"], params["perc_width_depth"], height)
        for i_rect, r in enumerate(rects):
            # Cutout the image for the boundary rectangle
            #   Note this will be a height x width numpy array
            im_warp, tform3_back = crv_2d.image_cutout(depth_data, r, step_size=width, height=height)
            im_cutouts.append(im_warp)
            trans_back.append(trans_back)

            ret_stats["ts"].append((ts[i_rect], ts[i_rect+1]))

            depth_unsorted = np.reshape(im_warp[:, :], (n_total_pixs))
            depth_sort = np.sort(depth_unsorted)

            ret_stats["depth_divs"].append([depth_sort[d] for d in divs])
            ts_seg = np.linspace(ts[i_rect], ts[i_rect + 1], width)
            t_image = np.ones((height, width))
            for c in range(0, height):
                t_image[c, :] = ts_seg
            r_image = np.ones((height, width))
            for r in range(0, width):
                r_image[:, r] = rs_seg * crv_2d.radius(ts_seg[r])

            # if radius value is correct, and curve centered, this would be the z value and radius
            pix_max = int(n_total_pixs * .95)
            depth_at_center = depth_sort[pix_max]
            rad_2d = crv_2d.radius(ts_seg[width // 2])
            ang_subtend_degrees = params["camera_width_angle"] * (2 * rad_2d) / depth_data.shape[1]
            ang_subtend_radians = np.pi * ang_subtend_degrees / 180.0
            radius_3d = 0.5 * depth_at_center * np.tan(ang_subtend_radians)
            z_at_center = depth_at_center - radius_3d

            rad_clip_min = z_at_center
            ret_stats["z_at_center"].append(z_at_center)
            ret_stats["radius_3d"].append(radius_3d)

            for r in range(0, width):
                for c in range(1, height):
                    if im_warp[c, r] > rad_clip_min:
                        ret_stats["depth_values"].append(im_warp[c, r])
                        ret_stats["r_at_depth"].append(t_image[c, r])
                        ret_stats["t_at_depth"].append(r_image[c, r])

            """   
            for r in range(0, width):
                for c in range(1, height):
                    p1_in = np.transpose(np.array([r, c, 1.0]))
                    p1_back = tform3_back @ p1_in
                    pt_spine = crv_2d.pt_axis(ts_seg[r])
                    vec_norm = crv_2d.norm_axis(ts_seg[r], "left")
                    perc_along = np.dot(p1_back[:2] - pt_spine, vec_norm)
                    h_perc = perc_along / crv_2d.radius(ts_seg[r])
            """
        return ret_stats

    @staticmethod
    def curve_from_stats(stats_depth, crv_2d, params):
        """
        From the raw stats, create a set of evenly-spaced t values
        @param stats_depth: The stats from full_depth_stats
        @param crv_2d - the 2d curve
        @param params: max edge pixel value, step_size, perc to search, and n pts to reconstruct
        @return: 3d curve
        """

        params['image_size'] = stats_depth['image_size']
        mat = frustrum_matrix(params)
        mat_inv = np.linalg.inv(mat)

        pts = []
        image_width = stats_depth["image_size"][0]
        image_height = stats_depth["image_size"][1]

        cam_width_ang_half = 0.5 * params['camera_width_angle']
        cam_height_ang_half = 0.5 * params['camera_width_angle'] * stats_depth['image_size'][1] / stats_depth['image_size'][0]
        print(f"cam x ang {cam_width_ang_half * 2} cam y ang {cam_height_ang_half * 2} {image_width}, {image_height}")

        pt_z_origin = np.ones(shape=(4,))
        pt_post_proj = np.ones(shape=(4,))
        pt_z_origin[0] = 0.0
        pt_z_origin[1] = 0.0

        ts_pts = [0, 0.5, 1.0]
        for t in ts_pts:
            if t < stats_depth["ts"][0][1]:
                z_at_center = stats_depth["z_at_center"][0]
            elif t > stats_depth["ts"][-1][0]:
                z_at_center = stats_depth["z_at_center"][-1]
            else:
                for i, ts in enumerate(stats_depth["ts"]):
                    if ts[0] <= t <= ts[1]:
                        z_at_center = stats_depth["z_at_center"][i]

            # Project the point 0, 0, d into the frustum box to get w, d'
            pt_z_origin[2] = -z_at_center
            pt_z_proj = mat @ pt_z_origin

            # Convert point in image coordinates to frustum box post project
            pt_crv_2d = crv_2d.pt_axis(t)
            pt_proj_box = from_image_to_box(params, pt_crv_2d)

            # Now use the w to get the point pre-divide
            pt_post_proj[0] = pt_proj_box[0] * pt_z_proj[3]
            pt_post_proj[1] = pt_proj_box[1] * pt_z_proj[3]
            pt_post_proj[2] = pt_z_proj[2]
            pt_post_proj[3] = pt_z_proj[3]

            # Now undo the projection
            pt_in_space = mat_inv @ pt_post_proj

            # Check result
            pts.append(pt_in_space[0:3])

        #for i in range(0, stats_depth["n_segs"]):
            #z_at_center = stats_depth["z_at_center"][i]
            #t = 0.5 * (stats_depth["ts"][i][0] + stats_depth["ts"][i][1])
            #radius_3d = stats_depth["radius_3d"][i]

        #i_mid = len(stats_depth["z_at_center"]) // 2
        crv_3d = BezierCyl3D(pts[0], pts[1], pts[2], stats_depth["radius_3d"][0], stats_depth["radius_3d"][-1])
        return crv_3d


if __name__ == '__main__':
    # path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = FileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = True
    for ind in all_files.loop_masks():
        rgb_fname = all_files.get_image_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ind, b_add_tag=True)
        depth_fname = all_files.get_depth_image_name(path=all_files.path, index=ind, b_add_tag=True)
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        depth_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=False)
        if not b_do_debug:
            depth_fname_debug =  None

        edge_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)
        depth_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        if not exists(rgb_fname):
            raise ValueError(f"Error, file {rgb_fname} does not exist")

        edge_crv = FitBezierCyl2DEdge(rgb_fname, edge_fname, mask_fname, edge_fname_calculate, None, b_recalc=False)

        crv_3d = FitBezierCyl3dDepth(depth_fname, edge_crv.bezier_crv_fit_to_edge,
                                     params=None,
                                     fname_calculated=depth_fname_calculate,
                                     fname_debug=depth_fname_debug, b_recalc=b_do_recalc)
