#!/usr/bin/env python3

# Fit a Bezier cylinder to a sketch in an image
#  Uses sketch to make initial Bezier curve
#     backbone plus cross bars
#  Creates a mask image from that sketch
#  Option 1: Just make a fat cylinder
#  Option 2: Uses the fit edge to fit profile curves, then just fills to the profile curves
#  
import numpy as np
import cv2
import json
from os.path import exists
from bezier_cyl_2d import BezierCyl2D
from fit_bezier_cyl_2d import FitBezierCyl2D
from fit_bezier_cyl_2d_edge import FitBezierCyl2DEdge
from line_seg_2d import LineSeg2D
from HandleFileNames import HandleFileNames
from fit_bezier_cyl_2d_mask import FitBezierCyl2DMask

import os
import sys
sys.path.insert(0, os.path.abspath('./sketch_curves_gui'))

from Sketches_for_curves import SketchesForCurves


class FitBezierCyl2DSketch:
    def __init__(self, fname_rgb_image, sketch_curves, fname_mask_image, fname_edge_image=None, fname_calculated=None, params=None, fname_debug=None, b_recalc=False):
        """ Create a mask image of the same size as the rgb image and a curve for the sketch_curves
        @param fname_rgb_image: Original rgb image name (for making edge name if we have to)
        @param sketch_curves: SketchesForCurves - has backbone and cross bars
        @param fname_mask_image: Mask image name - create a mask with that name
        @param fname_edge_image: Create and store the edge image, or read it in if it exists. If none, don't use in the fit curve process
        @param fname_calculated: the file name for the saved .json file; should be image name w/o _crv.json
        @param params: Dictionary with
            "resample_mask_step_size": how many pixels to use in each reconstructed rectangle; 10-20 is reasonable
            "perc_fuzzy_mask": Percentage of the outer boundary to make fuzzy (128); 0 - 0.5
        @param fname_debug: the file name for a debug image showing the bounding box, etc. Set to None if no debug image
        @param b_recalc: Force recalculate the result, y/n"""

        # Read in the RGB image
        self.image_rgb = cv2.imread(fname_rgb_image)

        # Now calculate the edge image, if it doesn't exist
        if exists(fname_edge_image):
            im_edge_color = cv2.imread(fname_edge_image)
            self.image_edge = cv2.cvtColor(im_edge_color, cv2.COLOR_BGR2GRAY)
        else:
            im_gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_BGR2GRAY)
            self.image_edge = cv2.Canny(im_gray, 50, 150, apertureSize=3)
            cv2.imwrite(fname_edge_image, self.image_edge)

        # Create the calculated file names
        b_debug = False
        if fname_debug:
            print(f"Fitting bezier curve to sketch {sketch_curves.backbone_pts}")
            b_debug = True

        self.fname_params = None  # Parameters used to do the fit
        # Create the file names for the calculated data that we'll store (sketch curve, curve fit to sketch curve, mask, parameters)
        if fname_calculated:
            self.fname_sketch = fname_calculated + "_sketch_curve.json"
            self.fname_sketch_bezier_crv = fname_calculated + "_sketch_curve_bezier.json"
            self.fname_params = fname_calculated + "_sketch_curve_params.json"

        # Copy params and add new ones
        self.params = {}
        if "resample_mask_step_size" not in self.params:
            self.params["resample_mask_step_size"] = 10
        if "perc_fuzzy_mask" not in self.params:
            self.params["perc_fuzzy_mask"] = 0.2

        if params:
            for k in params:
                self.params[k] = params[k]

        # Don't save this - just do it
        self.sketch_crv = FitBezierCyl2DSketch._sketch_curve_to_bezier(sketch_curves)
        if fname_calculated:
            # Write out sketch curve
            sketch_curves.write_json(self.fname_sketch)
            self.sketch_crv.write_json(self.fname_sketch_bezier_crv)

            with open(self.fname_params, "w") as f:
                json.dump(self.params, f, indent=2)

        if fname_mask_image:
            # Gray scale/bw that is the same size as rbg
            self.mask = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
            # Make black
            self.mask[:, :] = 0
            self.sketch_crv.make_mask_image(self.mask, self.params["resample_mask_step_size"], self.params["perc_fuzzy_mask"])
            cv2.imwrite(fname_mask_image, self.mask)

        if fname_debug:
            # Draw the mask with the initial and fitted curve
            im_rgb = np.copy(self.image_rgb)
            self.sketch_crv.draw_bezier(im_rgb)
            self.sketch_crv.draw_boundary(im_rgb)

            cv2.imwrite(fname_debug + "_sketch.png", im_rgb)

        # self.score = self.score_mask_fit(self.stats_dict.mask_image)
        # print(f"Mask {mask_fname}, score {self.score}")

    @staticmethod
    def _sketch_curve_to_bezier(sketch_curves):
        """ Convert the sketch curve to the bezier
        @param sketch_curves - has backbone and cross bars
        @return bezier_cyl_2d"""
            # First get the points in a reasonable form
        start_pt = sketch_curves.backbone_pts[0]
        end_pt = sketch_curves.backbone_pts[-1]
        mid_pt = [0.5 * (start_pt[0] + end_pt[0]), 0.5 * (start_pt[1] + end_pt[1])]
        if len(sketch_curves.backbone_pts) > 2:            
            mid_pt = sketch_curves.backbone_pts[len(sketch_curves.backbone_pts) // 2]

        radii = np.zeros(len(sketch_curves.cross_bars))
        for i, pts in enumerate(sketch_curves.cross_bars):
            p1 = pts[0]
            p2 = pts[1]
            radii[i] = 0.5 * np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        radius = np.mean(np.array(radii))
        sketch_crv = BezierCyl2D(start_pt=start_pt, mid_pt=mid_pt, end_pt=end_pt, radius=radius)
        if len(sketch_crv.cross_bars) > 1:
            half_way = len(sketch_crv.cross_baars) // 2
            sketch_crv.start_radius = np.radius = np.mean(np.array(radii[0:half_way]))
            sketch_crv.end_radius = np.radius = np.mean(np.array(radii[half_way:]))

        fit_crv = FitBezierCyl2D(sketch_crv)
        ts = np.linspace(0, 1, len(sketch_curves.backbone_pts))        
        a_constraints, b_rhs = fit_crv.setup_least_squares(ts)
        for i, pt in enumerate(sketch_curves.backbone_pts):
            b_rhs[i, 0] = pt[0]
            b_rhs[i, 1] = pt[1]

        fit_crv.extract_least_squares(a_constraints=a_constraints, b_rhs=b_rhs)

        return fit_crv.get_copy_of_2d_bezier_curve()


if __name__ == '__main__':

    # path_bpd = "./data/trunk_segmentation_names.json"
    path_bpd = "./Image_based/data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    index = (-1, -1, 0, 0)
    ret_index_mask_name = all_files.add_mask_name(index, "sketch")
    index_add = (0, 0, ret_index_mask_name[2], 0)
    ret_index_mask_id = all_files.add_mask_id(index_add)

    crv_in_image_coords = SketchesForCurves.read_json("save_crv_in_image.json")

    b_do_debug = True

    rgb_fname = all_files.get_image_name(path=all_files.path, index=ret_index_mask_id, b_add_tag=True)
    edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ret_index_mask_id, b_add_tag=True)
    mask_fname = all_files.get_mask_name(path=all_files.path, index=ret_index_mask_id, b_add_tag=True)
    edge_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ret_index_mask_id, b_add_tag=False)
    if not b_do_debug:
        edge_fname_debug = None
    crv_from_sketch = FitBezierCyl2DSketch(fname_rgb_image=rgb_fname, 
                                           sketch_curves=crv_in_image_coords, 
                                           fname_mask_image=mask_fname,
                                           fname_edge_image=edge_fname,
                                           fname_debug=edge_fname_debug)

    print("foo")
