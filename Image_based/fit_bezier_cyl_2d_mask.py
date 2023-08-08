#!/usr/bin/env python3

# Fit a Bezier cylinder to a mask
#  Adds fitting the bezier_cyl_2d to the mask by trying to place the Bezier curve's middle in the center of
#    the mask. Also adjusts the width
#  Essentially, chunk up the mask into pieces, find the average center, then set up a LS fit that (gradually)
#    moves the center by using each chunk's recommendation for where the center should be
#  Calculate IoU for mask and fitted curve
#    b.1) % pixels in center 80% of Bezier curve mask that are in original mask
#    b.2) % pixels outside of 1.1 * Bezier curve mask that are in original mask
#      Essentially, exclude the edge pixels from the intersection/union calculation

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
        """ Read in the mask image, use the stats to start the Bezier fit, then fit the Bezier to the mask
        @param fname_mask_image: Mask image name
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _crv.json
        @param fname_debug: the file name for a debug image showing the bounding box, etc
        @param b_recalc: Force recalculate the result, y/n"""

        # First do the stats - this also reads the image in
        self.stats_dict = BaseStatsImage(fname_mask_image, fname_calculated, fname_debug, b_recalc)

        # Now initialize bezier curve with info from stats - no need to cache this because it's so light-weight
        p0 = self.stats_dict.stats_dict["lower_left"]
        p2 = self.stats_dict.stats_dict["upper_right"]
        width = 0.5 * self.stats_dict.stats_dict["width"]
        # These are the two curves we'll create
        #   This is the initial curve fit to the stats dictionary
        self.bezier_crv_initial = BezierCyl2D(start_pt=p0, end_pt=p2, radius=width)
        #   This is the curve that will be fit to the mask
        self.bezier_crv_fit_to_mask = FitBezierCyl2D(self.bezier_crv_initial)

        # Create the calculated file names
        print(f"Fitting bezier curve to mask image {fname_mask_image}")
        self.fname_bezier_cyl_initial = None    # The actual quadratic bezier
        self.fname_bezier_cyl_fit_to_mask = None    # The actual quadratic bezier
        self.fname_params = None  # Parameters used to do the fit
        # Create the file names for the calculated data that we'll store (initial curve, curve fit to mask, parameters)
        if fname_calculated:
            self.fname_bezier_cyl_initial = fname_calculated + "_bezier_cyl_initial.json"
            self.fname_bezier_cyl_fit_to_mask = fname_calculated + "_bezier_cyl_fit_mask.json"
            self.fname_params = fname_calculated + "_bezier_cyl_params.json"

        # Current parameters for the vertical leader fit
        # TODO make this a parameter in the init function
        self.params = {"step_size": int(width * 1.5), "width_mask": 1.4, "width": 0.25}

        # Get the initial curve - either cached or create
        if b_recalc or not fname_calculated or not exists(self.fname_bezier_cyl_initial):
            # Recalculate and write
            self.bezier_crv_initial = FitBezierCyl2DMask.create_bezier_crv_from_eigen_vectors(self.stats_dict.stats_dict)
            # Write out the bezier curve
            if fname_calculated:
                self.bezier_crv_initial.write_json(self.fname_bezier_cyl_initial)
                with open(self.fname_params, 'w') as f:
                    json.dump(self.params, f, indent=" ")
        else:
            # Read in the stored data
            BezierCyl2D.read_json(self.fname_bezier_cyl_initial, self.bezier_crv_initial)
            with open(self.fname_params, 'r') as f:
                self.params = json.load(f)

        # Now do the fitted curve
        if b_recalc or not fname_calculated or not exists(self.fname_bezier_cyl_fit_to_mask):
            # Recalculate and write
            self.bezier_crv_fit_to_mask =\
                FitBezierCyl2DMask.fit_bezier_crv_to_mask(self.bezier_crv_initial,
                                                          self.stats_dict.mask_image,
                                                          params=self.params)
            if fname_calculated:
                self.bezier_crv_fit_to_mask.write_json(self.fname_bezier_cyl_fit_to_mask)
        else:
            # Read in the pre-calculated curve
            self.bezier_crv_fit_to_mask = BezierCyl2D.read_json(self.fname_bezier_cyl_fit_to_mask)

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
        print(f"Mask {fname_mask_image}, score {self.score}")

    @staticmethod
    def create_bezier_crv_from_eigen_vectors(stats):
        """ Fit a quad to the mask, edge image
        @param stats - the stats from BaseStatsImage (class BaseStatsImage)
        @return fitted Bezier"""

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
        return bezier_crv

    @staticmethod
    def _adjust_bezier_crv_by_mask(fit_bezier_crv, im_mask, step_size=40, perc_width=1.2):
        """Replace the linear approximation with one based on following the mask
        @param fit_bezier_crv - the bezier curve to move
        @param im_mask - mask image
        @param step_size - how many pixels to step along
        @param perc_width - how much wider than the radius to look in the mask
        @returns how much the points moved"""
        height = int(fit_bezier_crv.radius(0.5))
        rects, ts = fit_bezier_crv.interior_rects(step_size=step_size, perc_width=perc_width)

        # Set up the matrix - include the 3 current points plus the centers of the mask
        a_constraints, b_rhs = fit_bezier_crv.setup_least_squares(ts)

        x_grid, y_grid = np.meshgrid(range(0, step_size), range(0, height))
        for i, r in enumerate(rects):
            b_rect_inside = BezierCyl2D.rect_in_image(im_mask, r, pad=2)

            im_warp, tform_inv = fit_bezier_crv.image_cutout(im_mask, r, step_size=step_size, height=height)
            if b_rect_inside and np.sum(im_warp > 0) > 0:
                x_mean = np.mean(x_grid[im_warp > 0])
                y_mean = np.mean(y_grid[im_warp > 0])
                pt_warp_back = tform_inv @ np.transpose(np.array([x_mean, y_mean, 1]))
                print(f"{fit_bezier_crv.pt_axis(ts[i])} ({x_mean}, {y_mean}), {pt_warp_back}")
                b_rhs[i, :] = pt_warp_back[0:2]
            else:
                print(f"Empty slice {r}")

        return fit_bezier_crv.extract_least_squares(a_constraints, b_rhs)

    @staticmethod
    def fit_bezier_crv_to_mask(bezier_crv, im_mask, params):
        """ Fit a quad to the mask, edge image
        @param bezier_crv - the initial bezier curve to fit to the eigen vectors
        @param im_mask - the image mask
        @param params - the parameters to use in the fit
        @return fitted bezier"""

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
        return fit_bezier_crv.get_copy_of_2d_bezier_curve()

    def score_mask_fit(self, im_mask):
        """ A modified intersection over union that discounts pixels along the bezier cylinder mask
        @param im_mask - the mask image
        """

        # First, make a mask that is black where no cylinder, white in middle 75% of cylinder, and grey around the
        # edges
        im_bezier_mask = np.zeros((self.stats_dict.mask_image.shape[0], self.stats_dict.mask_image.shape[1]), np.uint8)

        self.bezier_crv_fit_to_mask.make_mask_image(im_bezier_mask, step_size=10, perc_fuzzy=0.25)

        pixs_in_mask_not_bezier = np.logical_and(im_mask > 0, im_bezier_mask == 0)
        pixs_in_bezier_not_mask = np.logical_and(im_mask == 0, im_bezier_mask == 255)
        pixs_in_both_masks = np.logical_and(im_mask > 0, im_bezier_mask == 255)
        pixs_in_union = np.logical_or(np.logical_or(pixs_in_bezier_not_mask, pixs_in_mask_not_bezier), pixs_in_both_masks)
        n_in_union = np.count_nonzero(pixs_in_union)
        if n_in_union == 0:
            return 0

        return np.count_nonzero(pixs_in_both_masks) / np.count_nonzero(pixs_in_union)


if __name__ == '__main__':
    # path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = True
    for ind in all_files.loop_masks():
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        mask_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=False)
        if not b_do_debug:
            mask_fname_debug = ""
        else:
            mask_fname_debug = mask_fname_debug + "_crv.png"

        mask_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        b_stats = FitBezierCyl2DMask(mask_fname, mask_fname_calculate, mask_fname_debug, b_recalc=b_do_recalc)
