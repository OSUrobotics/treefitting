#!/usr/bin/env python3

# Fit a BSpline cylinder to a sketch in one, or more, images
#  Uses sketch to make initial BSplineCyl
#     backbone plus cross bars
#  Keeps track of image/sketch pairs as they're added (plus offsets)
#  Keeps one curve w/radii
#  Can
#     1) Generate a mask in an image
#     2) Given an edge image, fit to the edges
#     3) Draw curve in image (debugging purposes)
#

import numpy as np

from tree_geometry.b_spline_curve import BSplineCurve
from tree_geometry.b_spline_cyl import BSplineCyl
from fit_routines.b_spline_curve_fit import BSplineCurveFit
from utils.sketched_curve import SketchedCurve
from tree_geometry.point_lists import PointList
from fit_routines.bspline_fit_params import BSplineFitParams
from utils.camera_projections import CameraProjections


class FitBSplineCyl2DSketch:
    def __init__(self, params : dict=None):
        """ Contains a 2D BSpline curve
        @param params: Dictionary with
            resample_mask_step_size: how many pixels to use in each reconstructed rectangle; 10-20 is reasonable
            perc_fuzzy_mask: Percentage of the outer boundary to make fuzzy (128); 0 - 0.5
            degree: One of linear, quadratic, or cubic
            """
        self.curve = None
        self.pts_fit = None
        self.image_frame_data = []

        # Set required params and/or add new ones
        self.params = {}
        self.change_fit_params(params)

    def change_fit_params(self, params : dict=None):
        """ Contains a 2D BSpline curve
        @param params: Dictionary with
            "resample_mask_step_size": how many pixels to use in each reconstructed rectangle; 10-20 is reasonable
            "perc_fuzzy_mask": Percentage of the outer boundary to make fuzzy (128); 0 - 0.5"""

        if "resample_mask_step_size" not in self.params:
            self.params["resample_mask_step_size"] = 10
        if "perc_fuzzy_mask" not in self.params:
            self.params["perc_fuzzy_mask"] = 0.2
        if "degree" not in self.params:
            self.params["degree"] = "cubic"

        if params:
            for k in params:
                self.params[k] = params[k]

    def _sketch_curve_to_bspline_cyl(self, sketch : SketchedCurve, fit_params: BSplineFitParams =None ):
        """ Convert the sketch curve to the bspline cylinder
        @param sketch_curves - has backbone and cross bars
        @return bspline_cyl"""

        crv_linear = BSplineCurve(ctrl_pts=sketch.backbone_pts, degree="linear")

        n_pts_start = BSplineCurve._degree_dict[self.params["degree"]] + 1
        pts_start = []
        for t in np.linspace(0, crv_linear.max_t(), n_pts_start):
            pts_start.append(crv_linear.eval_crv(t))

        pts_fit_to = PointList(sketch.backbone_pts)
        # Fix if only 2 points
        if pts_fit_to.n_points() < BSplineCurve._degree_dict[self.params["degree"]] + 1:
            pts_fit_to = PointList(pts_start)

        start_curve = BSplineCurve(pts_start, degree=self.params["degree"])
        curve_fit, _, _ = BSplineCurveFit.fit_project_fit(start_curve, pts_fit_to)

        curve_final, _, _ = BSplineCurveFit.fit_adjust_control_pts(curve_fit, pts_fit_to, fit_params)

        radii_ctrs, radii_radii = sketch.radii()
        ts = []
        for r_ctr in radii_ctrs:
            ts.append(crv_linear.project_to_curve(r_ctr)[0])
        radii_sorted = [r for _, r in sorted(zip(ts, radii_radii), key=lambda  pair: pair[0])]

        return BSplineCyl(ctrl_pts=curve_final.points(), degree=self.params["degree"], radii=radii_sorted)

    def add_sketch(self, dict_sketch : dict, sketch : SketchedCurve):
        """ Add in a sketch. All inputs are in the dictionary.
        Assumes up to the last 2 points (cubic) or 1 point (quadratic) are "fixed" and should not be changes
        :param dict_sketch: All the parameters needed to add the sketch
           rgb_image_name: Original rgb image name
           edge_image_name: Original edge image name, if any
           depth_image_name: Original depth image name, if any
           match_pts: List of pairs of points, one for the 
        :return: None """

        if self.image_frame_data == []:
            curve = self._sketch_curve_to_bspline_cyl(sketch)
        else:
            pass


if __name__ == '__main__':
    from os.path import exists
    import json

    sketch_name = "test_sketches_for_crvs.txt"
    crv_name = "test_fit_sketches.txt"
    if exists(sketch_name):
        with open(sketch_name, 'r') as f:
            my_data = json.load(f)
            sk_read = SketchedCurve.read_json(my_data)

        fit_spline = FitBSplineCyl2DSketch()
        crv_from_sketch = fit_spline._sketch_curve_to_bspline_cyl(sk_read)
        with open(crv_name, 'w') as f:
            json.dump(crv_from_sketch.write_json(), f, indent=2)

    print("foo")
