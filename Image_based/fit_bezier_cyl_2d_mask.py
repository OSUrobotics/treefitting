#!/usr/bin/env python3

# Fit a Bezier cylinder to a mask
#  Adds fitting the bezier_cyl_2d to the mask by trying to place the Bezier curve's middle in the center of
#    the mask. Also adjusts the width
#  Essentially, chunk up the mask into pieces, find the average center, then set up a LS fit that (gradually)
#    moves the center by using each chunk's recommendation for where the center should be

import numpy as np
import cv2
import json
from os.path import exists
from bezier_cyl_2d import BezierCyl2D
from fit_bezier_cyl_2d import FitBezierCyl2D
from BaseStatsImage import BaseStatsImage
from HandleFileNames import HandleFileNames


class FitBezierCyl2DMask:
    def __init__(self, fname_mask_image, fname_calculated=None, fname_debug=None, b_recalc=False):
        """ Read in the mask image, use the stats to start the quad fit, then fit the quad
        @param fname_mask_image: Mask image name
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _stats.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the stats
        self.stats_dict = BaseStatsImage(fname_mask_image, fname_calculated, fname_debug, b_recalc)
        # Now initialize bezier curve with info from stats - no need to cache this because it's so light-weight
        p0 = self.stats_dict.stats_dict["lower_left"]
        p2 = self.stats_dict.stats_dict["upper_right"]
        width = 0.5 * self.stats_dict.stats_dict["width"]
        # These are the two curves we'll create
        #   This is the initial curve fit to the stats dictionary
        self.bezier_crv_initial = BezierCyl2D(start_pt=p0, end_pt=p2, radius=width)
        # This is the curve Fit to the mask
        self.bezier_crv_fit_to_mask = FitBezierCyl2D(self.bezier_crv_initial)

        # Fit a quad to the mask, using the end points of the base image as a starting point
        print(f"Fitting bezier curve to mask image {fname_mask_image}")
        self.fname_bezier_cyl_initial = None    # The actual quadratic bezier
        self.fname_bezier_cyl_fit_to_mask = None    # The actual quadratic bezier
        self.fname_params = None  # Parameters used to do the fit
        if fname_calculated:
            self.fname_bezier_cyl_initial = fname_calculated + "bezier_cyl_initial.json"
            self.fname_bezier_cyl_fit_to_mask = fname_calculated + "bezier_cyl_fit_mask.json"
            self.fname_params = fname_calculated + "bezier_cyl_params.json"

        # Current parameters for the vertical leader fit
        # TODO make this a parameter of this class
        self.params = {"step_size": int(width * 1.5), "width_mask": 1.4, "width": 0.25}

        # Get the initial curve
        if b_recalc or not fname_calculated or not exists(self.fname_bezier_cyl_initial):
            # Recalculate and write
            FitBezierCyl2DMask.create_bezier_crv_from_eigen_vectors(self.bezier_crv_initial,
                                                                    self.stats_dict.mask_image,
                                                                    stats=self.stats_dict,
                                                                    params=self.params)
            self.bezier_crv_initial.write_json(self.fname_bezier_cyl_initial)
            if fname_calculated:
                self.bezier_crv_initial.write_json(self.fname_bezier_cyl_initial)
                with open(self.fname_params, 'r') as f:
                    self.params = json.load(f)
        else:
            BezierCyl2D.read_json(self.fname_bezier_cyl_initial, self.bezier_crv_initial)
            with open(self.fname_params, 'r') as f:
                self.params = json.load(f)

        # Now do the fitted curve
        if b_recalc or not fname_calculated or not exists(self.fname_bezier_cyl_fit_to_mask):
            # Recalculate and write
            self.bezier_crv_fit_to_mask = FitBezierCyl2D(self.bezier_crv_initial)
            FitBezierCyl2DMask.fit_bezier_crv_to_mask(self.bezier_crv_fit_to_mask,
                                                      self.stats_dict.mask_image,
                                                      params=self.params)
            if fname_calculated:
                self.bezier_crv_fit_to_mask.write_json(self.fname_bezier_cyl_fit_to_mask)
        else:
            BezierCyl2D.read_json(self.fname_bezier_cyl_fit_to_mask, self.bezier_crv_fit_to_mask)

        if fname_debug:
            # Draw the mask with the initial and fitted curve
            im_covert_back = cv2.cvtColor(self.stats_dict.mask_image, cv2.COLOR_GRAY2RGB)
            self.stats_dict.debug_image(im_covert_back)  # The eigen vec
            self.bezier_crv_initial.draw_bezier(im_covert_back)
            self.bezier_crv_initial.draw_boundary(im_covert_back)

            im_covert_back_fit = cv2.cvtColor(self.stats_dict.mask_image, cv2.COLOR_GRAY2RGB)
            self.bezier_crv_fit_to_mask.draw_bezier(im_covert_back_fit)
            self.bezier_crv_fit_to_mask.draw_boundary(im_covert_back_fit)
            im_both = np.hstack([im_covert_back, im_covert_back_fit])
            cv2.imwrite(fname_debug, im_both)

        self.score = self.score_mask_fit(self.stats_dict.mask_image)

    @staticmethod
    def create_bezier_crv_from_eigen_vectors(bezier_crv, im_mask, stats, params):
        """ Fit a quad to the mask, edge image
        @param bezier_crv - a blank bezier curve (class BezierCyl2D)
        @param im_mask - the image mask
        @param stats - the stats from BaseStatsImage (class BaseStatsImage)
        @param params - the parameters to use in the fit
        @return fitted Bezier and parameters used in the fit"""

        # Fit a Bezier curve to the mask - this does a bit of tweaking to try to extend the end points as
        #  far as possible
        pt_lower_left = stats['center']
        vec_len = stats["Length"] * 0.4
        while pt_lower_left[0] > 2 + stats['x_min'] and pt_lower_left[1] > 2 + stats['y_min']:
            pt_lower_left = stats["center"] - stats["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        pt_upper_right = stats['center']
        vec_len = stats["Length"] * 0.4
        while pt_upper_right[0] < -2 + stats['x_max'] and pt_upper_right[1] < -2 + stats['y_max']:
            pt_upper_right = stats["center"] + stats["EigenVector"] * vec_len
            vec_len = vec_len * 1.1

        bezier_crv = BezierCyl2D(pt_lower_left, pt_upper_right, 0.5 * stats['width'])
        return bezier_crv, params

    @staticmethod
    def _adjust_bezier_crv_by_mask(fit_bezier_crv, im_mask, step_size=40, perc_width=1.2):
        """Replace the linear approximation with one based on following the mask
        @param fit_bezier_crv - the bezier curve to move
        @param im_mask - mask image
        @param step_size - how many pixels to step along
        @param perc_width - how much wider than the radius to look in the mask
        @param axs - optional axes to draw the cutout in
        @returns how much the points moved"""
        height = int(fit_bezier_crv.radius_2d)
        rects, ts = fit_bezier_crv.interior_rects(step_size=step_size, perc_width=perc_width)

        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = fit_bezier_crv._setup_least_squares(ts)

        x_grid, y_grid = np.meshgrid(range(0, step_size), range(0, height))
        for i, r in enumerate(rects):
            b_rect_inside = BezierCyl2D._rect_in_image(im_mask, r, pad=2)

            im_warp, tform_inv = fit_bezier_crv._image_cutout(im_mask, r, step_size=step_size, height=height)
            if b_rect_inside and np.sum(im_warp > 0) > 0:
                x_mean = np.mean(x_grid[im_warp > 0])
                y_mean = np.mean(y_grid[im_warp > 0])
                pt_warp_back = tform_inv @ np.transpose(np.array([x_mean, y_mean, 1]))
                print(f"{fit_bezier_crv.pt_axis(ts[i])} ({x_mean}, {y_mean}), {pt_warp_back}")
                b_rhs[i, :] = pt_warp_back[0:2]
            else:
                print(f"Empty slice {r}")

        return fit_bezier_crv._extract_least_squares(a_constraints, b_rhs)

    @staticmethod
    def fit_bezier_crv_to_mask(bezier_crv, im_mask, params):
        """ Fit a quad to the mask, edge image
        @param bezier_crv - the initial bezier curve to fit to the eigen vectors
        @param im_mask - the image mask
        @param params - the parameters to use in the fit
        @return fitted bezier and parameters used in the fit"""

        # Make a copy of the input curve and edit it
        fit_bezier_crv = FitBezierCyl2D(bezier_crv)
        print(f"Res: ", end="")
        for i in range(0, 5):
            res = FitBezierCyl2DMask._adjust_bezier_crv_by_mask(fit_bezier_crv,
                                                                im_mask,
                                                                step_size=params["step_size"],
                                                                perc_width=params["width_mask"])
            print(f"{res} ", end="")
        print("")
        return fit_bezier_crv, params

    def score_mask_fit(self, im_mask):
        """ A modified intersection over union that discounts pixels along the bezier cylinder mask
        @param in_mask - the mask image
        """

        # First, make a mask that is black where no cylinder, white in middle 75% of cylinder, and grey around the
        # edges
        im_bezier_mask = np.zeros((im_mask.shape[0], im_mask.shape[1]), np.uint8)
        self.bezier_crv_fit_to_mask.draw_interior_rects_filled(im_bezier_mask, b_solid=True,
                                                               col_solid=(255, 255, 255),
                                                               step_size=10,
                                                               perc_width=0.75)
        self.bezier_crv_fit_to_mask.draw_edge_rects_filled(im_bezier_mask, b_solid=True,
                                                           col_solid=(255, 255, 255),
                                                           step_size=10,
                                                           perc_width=0.75)
        self.bezier_crv_fit_to_mask.draw_edge_rects_filled

        # Two checks: one, are the depth/optical fow values largely consistent under the quad center
        #  Are there boundaries in the optical flow image where the edge of the quad is?
        im_flow_mask = cv2.cvtColor(im_flow, cv2.COLOR_BGR2GRAY)
        perc_consistant, stats_slice = quad.check_interior_depth(im_flow_mask)

        diff = 0
        for i in range(1, len(stats_slice)):
            diff_slices = np.abs(stats_slice[i]["Median"] - stats_slice[i-1]["Median"])
            if diff_slices > 20:
                print(f"Warning: Depth values not consistant from slice {self.fname_quad} {i} {stats_slice}")
            diff += diff_slices
        if perc_consistant < 0.9:
            print(f"Warning: not consistant {self.fname_quad} {stats_slice}")
        return perc_consistant, diff / (len(stats_slice) - 1)

    def debug_image_bezier_fit(self, image_debug):
        """ Draw the fitted quad on the image
        @param image_debug - rgb image"""
        # Draw the bezier curve with the two boundary curves
        self.quad.draw_bezier(image_debug)
        if self.quad.is_wire():
            LineSeg2D.(image_debug, self.quad.p0, (255, 0, 0), thickness=2, length=10)
            LineSeg2D.draw_cross(image_debug, self.quad.p2, (255, 0, 0), thickness=2, length=10)
        else:
            self.quad.draw_boundary(image_debug, 10)


if __name__ == '__main__':
    #path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = False
    for ind in all_files.loop_masks():
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        mask_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=True)
        if not b_do_debug:
            mask_fname_debug = ""

        mask_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        b_stats = FitBezierCyl2D(mask_fname, mask_fname_calculate, mask_fname_debug, b_recalc=b_do_recalc)
    from branchpointdetection import BranchPointDetection

    # Compute all the branch points/approximate lines for branches
    bp = BranchPointDetection("data/forcindy/", "0")

    # Read in/compute the additional images we need for debugging
    #   Original image, convert to canny edge
    #   Mask image
    #   Depth image
    im_orig = cv2.imread('data/forcindy/0.png')
    im_depth = cv2.imread('data/forcindy/0_depth.png')
    im_mask_color = cv2.imread('data/forcindy/0_trunk_0.png')
    im_mask = cv2.cvtColor(im_mask_color, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.cvtColor(im_orig, cv2.COLOR_BGR2GRAY)
    im_edge = cv2.Canny(im_gray, 50, 150, apertureSize=3)
    im_depth_color = cv2.cvtColor(im_depth, cv2.COLOR_BGR2RGB)
    im_covert_back = cv2.cvtColor(im_edge, cv2.COLOR_GRAY2RGB)

    # Write out edge image
    cv2.imwrite('data/forcindy/0_edges.png', im_edge)

    # For the vertical leader...
    trunk_pts = bp.trunks[0]["stats"]
    # Fit a quad to the trunk
    quad = BezierCyl2D(trunk_pts['lower_left'], trunk_pts['upper_right'], 0.5 * trunk_pts['width'])

    # Current parameters for the vertical leader
    step_size_to_use = int(quad.radius_2d * 1.5)  # Go along the edge at 1.5 * the estimated radius
    perc_width_to_use = 0.3  # How "fat" to make the edge rectangles
    perc_width_to_use_mask = 1.4  # How "fat" a rectangle to cover the mask

    # Debugging image - draw the interior rects
    quad.draw_interior_rects(im_mask_color, step_size=step_size_to_use, perc_width=perc_width_to_use_mask)
    cv2.imwrite('data/forcindy/0_mask.png', im_mask_color)

    # For debugging images
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(im_orig)
    axs[0, 1].imshow(im_mask_color)
    plt.tight_layout()

    # Iteratively move the quad to the center of the mask
    for i in range(0, 5):
        res = quad.adjust_quad_by_mask(im_mask,
                                       step_size=step_size_to_use, perc_width=perc_width_to_use_mask,
                                       axs=axs[1, 0])
        print(f"Res {res}")

    # Draw the original, the edges, and the depth mask with the fitted quad
    quad.draw_bezier(im_orig)
    quad.draw_boundary(im_orig, 10)
    quad.draw_bezier(im_covert_back)

    quad.draw_edge_rects(im_orig, step_size=step_size_to_use, perc_width=perc_width_to_use)
    quad.draw_edge_rects(im_covert_back, step_size=step_size_to_use, perc_width=perc_width_to_use)
    #quad.draw_edge_rects_markers(im_edge, step_size=step_size_to_use, perc_width=perc_width_to_use)
    quad.draw_interior_rects(im_depth_color, step_size=step_size_to_use, perc_width=perc_width_to_use)

    im_both = np.hstack([im_orig, im_covert_back, im_depth_color])
    cv2.imshow("Original and edge and depth", im_both)
    cv2.imwrite('data/forcindy/0_rects.png', im_both)

    # Now do the hough transform - first draw the hough transform edges
    for i in range(0, 5):
        ret = quad.adjust_quad_by_hough_edges(im_edge, step_size=step_size_to_use, perc_width=perc_width_to_use, axs=axs[1, 1])
        print(f"Res Hough {ret}")

    im_orig = cv2.imread('data/forcindy/0.png')
    quad.draw_bezier(im_orig)
    quad.draw_boundary(im_orig, 10)
    cv2.imwrite('data/forcindy/0_quad.png', im_orig)

    print("foo")
