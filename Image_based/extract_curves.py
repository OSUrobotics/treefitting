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
from line_seg_2d import LineSeg2D
from HandleFileNames import HandleFileNames
from fit_bezier_cyl_2d_edge import FitBezierCyl2DEdge


class ExtractCurves:
    def __init__(
        self,
        fname_rgb_image,
        fname_edge_image,
        fname_mask_image,
        params=None,
        fname_calculated=None,
        fname_debug=None,
        b_recalc=False,
    ):
        """Read in the mask image, use the stats to start the quad fit, then fit the quad
        @param fname_rgb_image: Edge image name
        @param fname_edge_image: Edge image name
        @param fname_mask_image: Mask image name
        @param params: Parameters for filtering the edge image - how finely to sample along the edge and how much to believe edge
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _stats.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the stats - this also reads the image in
        self.bezier_edge = FitBezierCyl2DEdge(
            fname_rgb_image, fname_edge_image, fname_mask_image, fname_calculated, fname_debug, b_recalc
        )

        # List o pairs (t, plus/minus)
        self.left_curve = []
        self.right_curve = []

        # Create the file names for the calculated data that we'll store (initial curve, curve fit to mask, parameters)
        if fname_calculated:
            self.fname_full_edge_stats = fname_calculated + "_edge_extract_stats.json"
            self.fname_params = fname_calculated + "_edge_extract_params.json"
            self.fname_edge_curves = fname_calculated + "_edge_extract_edge_curves.json"

        # Current parameters for trunk extraction
        if params is None:
            self.params = {"step_size": 20, "perc_width": 0.2, "edge_max": 128, "n_avg": 10}
        else:
            self.params = params

        # Get the raw edge data
        if b_recalc or not fname_calculated or not exists(self.fname_full_edge_stats):
            # Recalculate and write
            self.edge_stats = ExtractCurves.full_edge_stats(
                self.bezier_edge.image_edge, self.bezier_edge.bezier_crv_fit_to_edge, self.params
            )
            # Write out the bezier curve
            if fname_calculated:
                with open(self.fname_full_edge_stats, "w") as f:
                    json.dump(self.edge_stats, f, indent=" ")
                with open(self.fname_params, "w") as f:
                    json.dump(self.params, f, indent=" ")
        else:
            # Read in the stored data
            with open(self.fname_full_edge_stats, "r") as f:
                self.edge_stats = json.load(f)
            with open(self.fname_params, "r") as f:
                self.params = json.load(f)

        # Now use the params to filter the raw edge location data - produces the left, right edge curves
        if b_recalc or not fname_calculated or not exists(self.fname_edge_curves):
            # Recalculate and write
            self.left_curve, self.right_curve = ExtractCurves.curves_from_stats(self.edge_stats, self.params)
            if fname_calculated:
                with open(self.fname_edge_curves, "w") as f:
                    json.dump((self.left_curve, self.right_curve), f, indent=" ")
        else:
            # Read in the pre-calculated edge curves
            with open(self.fname_edge_curves, "r") as f:
                curves = json.load(f)
                self.left_curve, self.right_curve = curves

        if fname_debug:
            # Draw the mask with the initial and fitted curve
            im_covert_back = cv2.cvtColor(self.bezier_edge.image_edge, cv2.COLOR_GRAY2RGB)
            im_rgb = np.copy(self.bezier_edge.image_rgb)
            for pix in self.edge_stats["pixs_edge"]:
                im_covert_back[pix[0], pix[1], :] = (255, 0, 0)
                im_rgb[pix[0], pix[1], :] = (255, 255, 255)

            for do_both_crv, do_both_name in [(self.left_curve, "Left"), (self.right_curve, "Right")]:
                for pt_e1, pt_e2 in zip(do_both_crv[0:-1], do_both_crv[1:]):
                    pt1 = self.bezier_edge.bezier_crv_fit_to_edge.edge_offset_pt(pt_e1[0], pt_e1[1], do_both_name)
                    pt2 = self.bezier_edge.bezier_crv_fit_to_edge.edge_offset_pt(pt_e2[0], pt_e2[1], do_both_name)
                    LineSeg2D.draw_line(im_covert_back, pt1, pt2, color=(255, 255, 0))
                    LineSeg2D.draw_line(im_rgb, pt1, pt2, color=(255, 255, 0))
            im_both = np.hstack([im_covert_back, im_rgb])
            cv2.imwrite(fname_debug, im_both)

    @staticmethod
    def full_edge_stats(image_edge, bezier_edge, params):
        """Get the best pixel offset (if any) for each point along the edge
        @param image_edge - the edge image
        @param bezier_edge - the Bezier curve
        @param params - parameters for extraction"""
        bdry_rects1, ts1 = bezier_edge.boundary_rects(step_size=params["step_size"], perc_width=params["perc_width"])
        bdry_rects2, ts2 = bezier_edge.boundary_rects(
            step_size=params["step_size"], perc_width=params["perc_width"], offset=True
        )
        n_bdry1 = len(bdry_rects1)
        try:
            t_step = ts1[2] - ts1[0]
        except IndexError:
            t_step = 1.0

        bdry_rects1.extend(bdry_rects2)
        ts1.extend(ts2)

        # Size of the rectangle(s) to cutout is based on the step size and the radius
        height = int(bezier_edge.radius(0.5))
        width = params["step_size"]
        # rect_destination = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

        ret_stats = {"ts_left": [], "left_perc": [], "ts_right": [], "right_perc": [], "pixs_edge": []}
        for i_rect, r in enumerate(bdry_rects1):
            # b_rect_inside = BezierCyl2D._rect_in_image(image_edge, r, pad=2)

            im_warp, tform3_back = bezier_edge.image_cutout(image_edge, r, step_size=width, height=height)
            # Actual hough transform on the cut-out image
            lines = cv2.HoughLines(im_warp, 1, np.pi / 180.0, 10)

            # Check for any lines in the cutout image
            if lines is None:
                continue
            # .. and check if any of those are horizontal
            ret_pts = FitBezierCyl2DEdge.get_horizontal_lines_from_hough(lines, tform3_back, width, height)
            if ret_pts is []:
                continue
            i_side = i_rect % 2

            if i_rect == n_bdry1:
                try:
                    t_step = ts2[2] - ts2[0]
                except IndexError:
                    t_step = 1.0

            max_y = im_warp.max(axis=0)

            ts_seg = np.linspace(ts1[i_rect] - t_step * 0.5, ts1[i_rect] + t_step * 0.5, len(max_y))
            for i_col, y in enumerate(max_y):
                if y > params["edge_max"]:
                    ids = np.where(im_warp[:, i_col] == y)
                    if i_side == 0:
                        tag = "left"
                    else:
                        tag = "right"
                    ret_stats["ts_" + tag].append(ts_seg[i_col])
                    h_min = params["perc_width"] * bezier_edge.radius(ts1[i_col])
                    h_max = (1 + params["perc_width"]) * bezier_edge.radius(ts1[i_col])
                    ret_stats[tag + "_perc"].append(h_min + h_max + ids[0] / width)

                    p1_in = np.transpose(np.array([ids[0][0], i_col, 1.0]))
                    p1_back = tform3_back @ p1_in

                    ret_stats["pixs_edge"].append([p1_back[0], p1_back[1]])

        np.array([ret_stats["ts_left"], ret_stats["left_perc"]])
        np.sort(axis=0)
        ret_stats["ts_left"] = list(np.array[0, :])
        ret_stats["left_perc"] = list(np.array[1, :])
        # sort(zip(ret_stats["ts_left"], ret_stats["left_perc"]), key=0)
        # sort(zip(ret_stats["ts_right"], ret_stats["right"]), key=0)
        return ret_stats

    @staticmethod
    def curves_from_stats(stats_edge, params):
        """
        From the raw stats, create a set of evenly-spaced t values
        @param stats_edge: The stats from full_edge_stats
        @param params: The params
        @return: a tuple of left, right edges as t, perc in/out
        """

        crvs = []
        for ts, ps in [
            (stats_edge["ts_left"], stats_edge["Left_perc"]),
            (stats_edge["ts_right"], stats_edge["Right_perc"]),
        ]:
            ps_filter = np.array(ps)
            for i_filter in range(0, params["n_filter"]):
                # np.filter(ps_filter)
                ts_crvs = np.linspace(0, 1, params["n_samples"])
                ps_crvs = np.interp(ts_crvs, ts, ps_filter)
                crvs.append([(t, p) for t, p in zip(ts_crvs, ps_crvs)])
        return crvs[0], crvs[1]


if __name__ == "__main__":
    # path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = True
    for ind in all_files.loop_masks():
        rgb_fname = all_files.get_image_name(path=all_files.path, index=ind, b_add_tag=True)
        edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ind, b_add_tag=True)
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        ec_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=False)
        if not b_do_debug:
            ec_fname_debug = ""
        else:
            ec_fname_debug = ec_fname_debug + "_extract_profile.png"

        ec_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        if not exists(rgb_fname):
            raise ValueError(f"Error, file {rgb_fname} does not exist")

        profile_crvs = ExtractCurves(
            rgb_fname,
            edge_fname,
            mask_fname,
            fname_calculated=ec_fname_calculate,
            fname_debug=ec_fname_debug,
            b_recalc=b_do_recalc,
        )
