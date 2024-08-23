#!/usr/bin/env python3

# Read in one masked image, the edge image, and the fitted bezier curve. Assumes one mask'd area
#   Fit the Bezier cylinder (FitBezierCrv2DEdge)
#     - This will calculate any additional information needed
#   Extract a continuous set of points along the edge boundary
#     - Do each boundary independently
#     - Store as t, percentage in/out from radius (so we can re-use at different scales)

import numpy as np
import cv2
import json
from os.path import exists
from line_seg_2d_draw import LineSeg2D
from fit_bezier_cyl_2d_edge import FitBezierCyl2DEdge
from FileNames import FileNames


class ExtractCurves:
    def __init__(self, fname_rgb_image, fname_edge_image, fname_mask_image, params=None, fname_calculated=None, fname_debug=None, b_recalc=False):
        """ Read in the mask image, use the stats to start the quad fit, then fit the quad
        @param fname_rgb_image: Edge image name
        @param fname_edge_image: Edge image name
        @param fname_mask_image: Mask image name
        @param params: Parameters for filtering the edge image - how finely to sample along the edge and how much to believe edge
           perc_width - percent of edge to search, should be 0.1 to 0.5
           edge_max - brightness of edge pixel to be considered an edge (0..255)
           n_per_seg - number of samples to keep along the edge, should be less than step_size (5-10)
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _stats.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc. Set to None if no debug
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the stats - this also reads the image in
        self.bezier_edge = FitBezierCyl2DEdge(fname_rgb_image=fname_rgb_image,
                                              fname_edge_image=fname_edge_image,
                                              fname_mask_image=fname_mask_image,
                                              fname_calculated=fname_calculated,
                                              params=params,
                                              fname_debug=fname_debug,
                                              b_recalc=b_recalc)
        
        # List o pairs (t, plus/minus)
        self.left_curve = []
        self.right_curve = []

        # Create the file names for the calculated data that we'll store (initial curve, curve fit to mask, parameters)
        if fname_calculated:
            self.fname_full_edge_stats = fname_calculated + "_edge_extract_stats.json"
            self.fname_params = fname_calculated + "_edge_extract_params.json"
            self.fname_edge_curves = fname_calculated + "_edge_extract_edge_curves.json"

        # Copy params used in fit mask and add the new ones
        self.params = {}
        for k in self.bezier_edge.params.keys():
            self.params[k] = self.bezier_edge.params[k]
        if "edge_max" not in self.params:
            self.params["edge_max"] = 128
        if "n_per_seg" not in self.params:
            self.params["n_per_seg"] = 10
        if "perc_width" not in self.params:
            self.params["width_profile"] = 0.3

        # Get the raw edge data
        if b_recalc or not fname_calculated or not exists(self.fname_full_edge_stats):
            # Recalculate and write
            self.edge_stats = ExtractCurves.full_edge_stats(self.bezier_edge.image_edge,
                                                            self.bezier_edge.bezier_crv_fit_to_edge,
                                                            self.params)
            # Write out the bezier curve
            if fname_calculated:
                with open(self.fname_full_edge_stats, 'w') as f:
                    json.dump(self.edge_stats, f, indent=" ")
                with open(self.fname_params, 'w') as f:
                    json.dump(self.params, f, indent=" ")
        else:
            # Read in the stored data
            with open(self.fname_full_edge_stats, 'r') as f:
                self.edge_stats = json.load(f)
            with open(self.fname_params, 'r') as f:
                self.params = json.load(f)

        # Now use the params to filter the raw edge location data - produces the left, right edge curves
        if b_recalc or not fname_calculated or not exists(self.fname_edge_curves):
            # Recalculate and write
            self.left_curve, self.right_curve = ExtractCurves.curves_from_stats(self.edge_stats, self.params)
            if fname_calculated:
                with open(self.fname_edge_curves, 'w') as f:
                    json.dump((self.left_curve, self.right_curve), f, indent=" ")
        else:
            # Read in the pre-calculated edge curves
            with open(self.fname_edge_curves, 'r') as f:
                curves = json.load(f)
                self.left_curve, self.right_curve = curves

        self.edge_stats["pixs_filtered"] = []
        for crv, side in zip([self.left_curve, self.right_curve], ["Left", "Right"]):
            for t_perc in crv:
                pt = self.bezier_edge.bezier_crv_fit_to_edge.edge_offset_pt(t_perc[0], t_perc[1], side)
                self.edge_stats["pixs_filtered"].append([pt[0], pt[1]])

        if fname_debug:
            # Draw the mask with the initial and fitted curve
            im_covert_back = cv2.cvtColor(self.bezier_edge.image_edge, cv2.COLOR_GRAY2RGB)
            im_rgb = np.copy(self.bezier_edge.image_rgb)
            for pix in self.edge_stats["pixs_edge"]:
                ix = int(pix[0])
                iy = int(pix[1])
                try:
                    LineSeg2D.draw_box(im_covert_back, (ix, iy), color=(255, 0, 0))
                    LineSeg2D.draw_box(im_rgb, (ix, iy), color=(255, 255, 255))
                    im_covert_back[ix, iy, :] = (255, 0, 0)
                    im_rgb[ix, iy, :] = (255, 255, 255)
                except IndexError:
                    print(f"Bad pixel x,y {ix}, {iy}")

            im_both = np.hstack([im_covert_back, im_rgb])
            cv2.imwrite(fname_debug + "_extract_pixs.png", im_both)
            for do_both_crv, do_both_name in [(self.left_curve, "Left"), (self.right_curve, "Right")]:
                for pt_e1, pt_e2 in zip(do_both_crv[0:-1], do_both_crv[1:]):
                    pt1 = self.bezier_edge.bezier_crv_fit_to_edge.edge_offset_pt(pt_e1[0], pt_e1[1], do_both_name)
                    pt2 = self.bezier_edge.bezier_crv_fit_to_edge.edge_offset_pt(pt_e2[0], pt_e2[1], do_both_name)
                    LineSeg2D.draw_line(im_covert_back, pt1, pt2, color=(255, 255, 0))
                    LineSeg2D.draw_line(im_rgb, pt1, pt2, color=(255, 255, 0))
            im_both = np.hstack([im_covert_back, im_rgb])
            cv2.imwrite(fname_debug + "_extract_profiles.png", im_both)

    @staticmethod
    def full_edge_stats(image_edge, bezier_edge, params):
        """ Get the best pixel offset (if any) for each point/pixel along the edge
        @param image_edge - the edge image
        @param bezier_edge - the Bezier curve
        @param params - parameters for extraction
        @return ts, perc off of edge, actual pixels (for debugging) and total number of segs"""

        # Fuzzy rectangles along the boundary
        bdry_rects1, ts1 = bezier_edge.boundary_rects(step_size=params["step_size"], perc_width=params["width_profile"])
        # Repeat, but offset by 1/2 of the width of the boundary
        bdry_rects2, ts2 = bezier_edge.boundary_rects(step_size=params["step_size"], perc_width=params["width_profile"], offset=True)
        n_bdry1 = len(bdry_rects1)
        try:
            t_step = ts1[2] - ts1[0]
        except IndexError:
            t_step = 1.0


        # Glue the rectangle lists together
        bdry_rects1.extend(bdry_rects2)
        ts1.extend(ts2)

        # Size of the rectangle(s) to cutout is based on the step size and the radius
        height = int(bezier_edge.radius(0.5))
        width = params["step_size"]

        ret_stats = {"n_segs": len(ts2),
                     "ts_left":[], "left_perc":[], "ts_right":[], "right_perc":[],
                     "pixs_edge":[],
                     "pixs_reconstruct":[]}
        for i_rect, r in enumerate(bdry_rects1):
            # Cutout the image for the boundary rectangle
            #   Note this will be a height x width numpy array
            im_warp, tform3_back = bezier_edge.image_cutout(image_edge, r, step_size=width, height=height)
            # Actual hough transform on the cut-out image
            lines = cv2.HoughLines(im_warp, 1, np.pi / 180.0, 10)

            # Switch to the offset rectangles, so t_step is a bit different
            if i_rect == n_bdry1:
                try:
                    t_step = ts2[2] - ts2[0]
                except IndexError:
                    t_step = 1.0

            # Rectangles alternate left, right side of curve
            i_side = i_rect % 2
            if i_side == 0:
                tag = "left"
            else:
                tag = "right"

            # One t value for each column in the image cutout
            ts_seg = np.linspace(ts1[i_rect] - t_step * 0.5, ts1[i_rect] + t_step * 0.5, width)

            # Check for any lines in the cutout image
            b_has_line = False

            if lines is not None:
                # .. and check if any of those are horizontal
                ret_pts = FitBezierCyl2DEdge.get_horizontal_lines_from_hough(lines, tform3_back, width, height)
                if len(ret_pts) > 0:
                    b_has_line = True

            if b_has_line is False:
                # Put a point in the middle at the estimated radius
                ret_stats["ts_" + tag].append(ts_seg[width // 2])
                ret_stats[tag + "_perc"].append(1.0)
                pt = bezier_edge.edge_pts(ts_seg[width // 2])
                ret_stats["pixs_edge"].append([pt[i_side][0], pt[i_side][1]])
            else:
                # Find the max y value in every column
                #max_y = im_warp.max(axis=0)

                # These should be the same... a 1x20 array
                #assert(len(max_y) == len(ts_seg))

                line_abc = FitBezierCyl2DEdge._get_line_abc(ret_pts[0][0], ret_pts[0][1])
                # Loop over each column
                for i_along in range(0, width):
                    t_seg = ts_seg[i_along]
                    i_best = -1
                    d_best = 100000
                    for j_in_column in range(0, height):
                        if im_warp[j_in_column, i_along] > params["edge_max"]:
                            d_dist = line_abc[0] * i_along + line_abc[1] * j_in_column + line_abc[2]
                            if d_dist < d_best:
                                i_best = j_in_column
                                d_best = d_dist
                    if i_best != -1:
                        # t value from linearly sampling the t value along the edge segment
                        ret_stats["ts_" + tag].append(t_seg)
                        # Where in the image the max value occured
                        #   Search for the one closest to the fit line
                        #   Should be between 0 and height
                        # This is how far up/down the pixel was in the height of the image

                        p1_in = np.transpose(np.array([i_along, i_best, 1.0]))
                        p1_back = tform3_back @ p1_in
                        pt_spine = bezier_edge.pt_axis(t_seg)
                        vec_norm = bezier_edge.norm_axis(t_seg, tag)
                        perc_along = np.dot(p1_back[:2] - pt_spine, vec_norm)
                        h_perc = perc_along / bezier_edge.radius(t_seg)

                        pt_reconstruct = bezier_edge.edge_offset_pt(t_seg, h_perc, tag)

                        ret_stats[tag + "_perc"].append(h_perc)

                        ret_stats["pixs_edge"].append([p1_back[0], p1_back[1]])
                        ret_stats["pixs_reconstruct"].append([pt_reconstruct[0], pt_reconstruct[1]])
                        #ret_stats["pixs_edge"].append([float(r[0][0]), float(r[0][1])])

        # There's probably a more graceful way to do this...
        data_in_matrix_form = np.zeros([2, len(ret_stats["ts_left"])])
        data_in_matrix_form[0, :] = np.array(ret_stats["ts_left"])
        data_in_matrix_form[1, :] = np.array(ret_stats["left_perc"])
        order_ts = data_in_matrix_form[0, :].argsort()
        for i, ind in enumerate(order_ts):
            ret_stats["ts_left"][i] = data_in_matrix_form[0, ind]
            ret_stats["left_perc"][i] = data_in_matrix_form[1, ind]

        data_in_matrix_form = np.zeros([2, len(ret_stats["ts_right"])])
        data_in_matrix_form[0, :] = np.array(ret_stats["ts_right"])
        data_in_matrix_form[1, :] = np.array(ret_stats["right_perc"])
        order_ts = data_in_matrix_form[0, :].argsort()
        for i, ind in enumerate(order_ts):
            ret_stats["ts_right"][i] = data_in_matrix_form[0, ind]
            ret_stats["right_perc"][i] = data_in_matrix_form[1, ind]

        # sort(zip(ret_stats["ts_left"], ret_stats["left_perc"]), key=0)
        # sort(zip(ret_stats["ts_right"], ret_stats["right"]), key=0)
        return ret_stats

    @staticmethod
    def curves_from_stats(stats_edge, params):
        """
        From the raw stats, create a set of evenly-spaced t values
        @param stats_edge: The stats from full_edge_stats
        @param params: max edge pixel value, step_size, perc to search, and n pts to reconstruct
        @return: a tuple of left, right edges as t, perc in/out
        """
        stats_edge["pixs_resampled"] = []
        crvs = []
        n_total = params["n_per_seg"] * stats_edge["n_segs"]
        ts_crvs = np.linspace(0, 1, n_total)
        for ts, ps in [(stats_edge["ts_left"], stats_edge["left_perc"]), (stats_edge["ts_right"], stats_edge["right_perc"])]:
            ps_array = np.array(ps)
            for _ in range(0, 5):
                ps_array[1:-1] = 0.5 * ps_array[1:-1] + 0.25 * (ps_array[0:-2] + ps_array[2:])
            ps_crvs = np.interp(ts_crvs, ts, ps_array)
            #crvs.append([(t, p) for t, p in zip(ts, ps)])
            crvs.append([(t, p) for t, p in zip(ts_crvs, ps_crvs)])
        # Left curve as t, perc pairs and same for right
        return crvs[0], crvs[1]

    @staticmethod
    def create_from_filenames(filenames, index=(0,0,0,0), b_do_recalc=False, b_do_debug=True, b_use_optical_flow_edge=False):
        """ Create a base image from a file name in FileNames
        @param filenames - FileNames instance
        @param index tuple (eg (0,0,0,0))
        @param b_do_recalc - recalculate from scratch
        @param b_do_debug - spit out a debug image y/n
        @param b_use_optical_flow_edge - use optical flow edge image instead of rgb edge image
        @return extract curves"""

        rgb_fname = filenames.get_image_name(index=index, b_add_tag=True)

        if not exists(rgb_fname):
            raise ValueError(f"No file {rgb_fname}")

        # File name
        mask_fname = filenames.get_mask_name(index=index, b_add_tag=True)
        edge_fname = filenames.get_edge_name(index=index, b_optical_flow=b_use_optical_flow_edge, b_add_tag=True)
        # Debug image file name
        if b_do_debug:
            mask_fname_debug = filenames.get_mask_name(index=index, b_debug_path=True, b_add_tag=False)
        else:
            mask_fname_debug = None

        # The stub of the filename to save all of the data to
        mask_fname_calculate = filenames.get_mask_name(index=index, b_calculate_path=True, b_add_tag=False)

        if not exists(mask_fname):
            print(f"Warning, file {mask_fname} does not exist")
        profile_crvs = ExtractCurves(rgb_fname, edge_fname, mask_fname,
                                     fname_calculated=mask_fname_calculate,
                                     fname_debug=mask_fname_debug, b_recalc=b_do_recalc)
        return profile_crvs


if __name__ == '__main__':
    path_bpd_envy = "/Users/cindygrimm/VSCode/treefitting/Image_based/data/EnvyTree/"
    all_fnames_envy = FileNames.read_filenames(path=path_bpd_envy, 
                                               fname="envy_fnames.json")
    ExtractCurves.create_from_filenames(all_fnames_envy, (0, 0, 0, 0), b_do_recalc=False, b_do_debug=False)

    # path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = FileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = True
    b_use_optical_flow_edge = True
    for ind in all_files.loop_masks():
        rgb_fname = all_files.get_image_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ind, b_optical_flow=b_use_optical_flow_edge, b_add_tag=True)
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        ec_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=False)
        if not b_do_debug:
            ec_fname_debug =  None

        ec_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        if not exists(rgb_fname):
            raise ValueError(f"Error, file {rgb_fname} does not exist")

        profile_crvs = ExtractCurves(rgb_fname, edge_fname, mask_fname,
                                     fname_calculated=ec_fname_calculate,
                                     fname_debug=ec_fname_debug, b_recalc=b_do_recalc)
