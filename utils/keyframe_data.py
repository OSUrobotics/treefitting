#!/usr/bin/env python3

# Manually created data (from sketching)
# Note: Mirrors Mask data in filenames
#   Each sketch curve belongs to a specific mask name
#       After that, numbered 0, 1, 2 etc
#
# 2D transformation approximation (vector for pan, scale)
#
# Generated data (2D)
#   BSpline curve for each sketch curve
#   Matrix approximation for 2D motion
#
# Generated data (3D)
#   Input: How many points to sample along the backbone of the curve and in-out along the curve
#   6dof camera transform to the next keyframe
#

import numpy as np

from fit_routines.bspline_fit_params import BSplineFitParams
from utils.sketched_curve import SketchedCurve
from tree_geometry.b_spline_cyl import BSplineCyl
from fit_routines.fit_bspline_cyl_sketch import FitBSplineCyl2DSketch


class KeyFrameData:
    _sketch_fit = FitBSplineCyl2DSketch()

    def __init__(self, image_name):
        self.image_name = image_name
        self.mask_names = []
        self.sketch_curves = []
        self.bspline_cyls = []
        self.pan_vec = [0, 0]
        self.scale_amount = 1.0
        self.rot_amount = 1.0
        self.pts_2d_of_start = []
        self.pts_2d_of_end = []
        self.pts_2d_depth = []
        self.pts_2d_rgb_depth = []

        # Should satisfy:
        #  sx * (rgb width / 2) - tx = depth width / 2
        #  sy * (rgb height / 2) - ty = depth height / 2
        #    Note: Y is swapped, so need to add, not subtract
        #          x is a subtract
        # Making sx/sy smaller means more of the rgb image is covered
        self.rgb_to_depth_matrix = np.identity(3)
        self.camera_matrix = np.identity(4)

    def add_mask_name(self, name):
        """Add the mask name - should be called whenever add_mask_name is called on FileNames/VideoAnnotationData
        @name - the name of the mask - not realy used, except to ensure data consistancy"""
        self.mask_names.append(name)
        self.sketch_curves.append([])
        self.bspline_cyls.append([])

    def add_sketch(self, mask_index, sketch_curve):
        """ Add in a sketch and also build the BSpline cylinder
        @param mask_index is which mask to add to
        @param sketch_curve - sketched curve"""
        self.sketch_curves[mask_index].append(sketch_curve.copy())
        bspl_cyl = KeyFrameData._sketch_fit._sketch_curve_to_bspline_cyl(sketch_curve)
        self.bspline_cyls[mask_index].append(bspl_cyl)

    def replace_sketch(self, mask_index, mask_id_index, sketch_curve):
        """ Replace the sketch
        @param mask_index is which mask to add to
        @param mask_id_index is which mask curve to replace
        @param sketch_curve - sketched curve"""
        self.sketch_curves[mask_index][mask_id_index] = sketch_curve.copy()
        self.bspline_cyls[mask_index][mask_id_index] =KeyFrameData._sketch_fit._sketch_curve_to_bspline_cyl(sketch_curve)

    def get_sketch(self, mask_index, mask_id_index):
        """ Get the sketch corresponding to the mask, id"""
        return self.sketch_curves[mask_index][mask_id_index]

    def get_bsplinecyl(self, mask_index, mask_id_index):
        """ Get the bspline cyl corresponding to the mask, id"""
        return self.bspline_cyls[mask_index][mask_id_index]

    def depth_pts_in_rgb(self):
        pt = np.ones((3, 1))
        pts_ret = []

        mat_depth_to_rgb = np.linalg.inv(self.rgb_to_depth_matrix)
        for p in self.pts_2d_depth:
            pt[0, 0] = p[0]
            pt[1, 0] = p[1]
            pt_map = mat_depth_to_rgb @ pt
            pts_ret.append([pt_map[0, 0], pt_map[1, 0]])
        return pts_ret

    def rgb_pts_in_depth(self):
        pt = np.ones((3, 1))
        pts_ret = []

        for p in self.pts_2d_rgb_depth:
            pt[0, 0] = p[0]
            pt[1, 0] = p[1]
            pt_map = self.rgb_to_depth_matrix @ pt
            pts_ret.append([pt_map[0, 0], pt_map[1, 0]])
        return pts_ret

    def get_bsplinecyl_in_depth_image(self, mask_index, mask_id_index):
        """ Get the bspline cyl corresponding to the mask, id, and convert it to the depth image coordinates"""

        crv = self.get_bsplinecyl(mask_index, mask_id_index)
        return crv.transform(self.rgb_to_depth_matrix)

    def refit(self, fit_params : BSplineFitParams=None):
        fit_sketch = FitBSplineCyl2DSketch()

        for mask_indx, _ in enumerate(self.mask_names):
            for indx, (sk, bs) in enumerate(zip(self.sketch_curves[mask_indx], self.bspline_cyls[mask_indx])):
                self.bspline_cyls[mask_indx][indx] = fit_sketch._sketch_curve_to_bspline_cyl(sk, fit_params=fit_params)

    def write_json(self):
        """Create a dictionary and return it"""
        my_dict = {"Name": "KeyFrameData",
                   "ImageName": self.image_name,
                   "sketch_curves": [],
                   "bspline_cyls":  [],
                   "mask_names": self.mask_names,
                   "PanVec": self.pan_vec.copy(),
                   "pts_2d_of_start": self.pts_2d_of_start,
                   "pts_2d_of_end": self.pts_2d_of_end,
                   "pts_2d_depth": self.pts_2d_depth,
                   "pts_2d_rgb_depth": self.pts_2d_rgb_depth,
                   "ScaleAmt": self.scale_amount,
                   "RotAmt": self.rot_amount,
                   "rgb_to_depth_matrix": self.rgb_to_depth_matrix.tolist(),
                   "camera_matrix": self.camera_matrix.tolist()}

        for lst1, lst2 in zip(self.sketch_curves, self.bspline_cyls):
            my_dict["sketch_curves"].append([])
            my_dict["bspline_cyls"].append([])
            for crv in lst1:
                my_dict["sketch_curves"][-1].append(crv.write_json())
            for crv in lst2:
                my_dict["bspline_cyls"][-1].append(crv.write_json())

        return my_dict

    @staticmethod
    def read_json(json_dict, key_frame_instance=None):
        """ Read back in from json file
        @param json_dict - dictionary read in from file
        @param key_frame_instance - an existing key frame to put the data in"""
        if json_dict["Name"] != "KeyFrameData":
            raise ValueError(f"This is not a key frame dictionary {json_dict}")

        if not key_frame_instance:
            key_frame_instance = KeyFrameData(json_dict["ImageName"])

        key_frame_instance.sketch_curves = []
        for lst in json_dict["sketch_curves"]:
            key_frame_instance.sketch_curves.append([])
            for crv in lst:
                key_frame_instance.sketch_curves[-1].append(SketchedCurve.read_json(crv))

        key_frame_instance.bspline_cyls = []
        for lst in json_dict["bspline_cyls"]:
            key_frame_instance.bspline_cyls.append([])
            for crv in lst:
                key_frame_instance.bspline_cyls[-1].append(BSplineCyl.read_json(crv))

        key_frame_instance.image_name = json_dict["ImageName"]
        key_frame_instance.mask_names = json_dict["mask_names"].copy()
        key_frame_instance.pan_vec = json_dict["PanVec"]
        key_frame_instance.scale_amount = json_dict["ScaleAmt"]
        key_frame_instance.rgb_to_depth_matrix = np.array(json_dict["rgb_to_depth_matrix"])
        key_frame_instance.camera_matrix = np.array(json_dict["camera_matrix"])

        # Other oddball ones
        try:
            key_frame_instance.pts_2d_of_start = json_dict["pts_2d_of_start"]
            key_frame_instance.pts_2d_of_end = json_dict["pts_2d_of_end"]
            key_frame_instance.pts_2d_rgb_depth = json_dict["pts_2d_rgb_depth"]
            key_frame_instance.pts_2d_depth = json_dict["pts_2d_depth"]
        except KeyError:
            pass
        try:
            key_frame_instance.rot_amount = json_dict["RotAmt"]
        except KeyError:
            pass

        return key_frame_instance


if __name__ == '__main__':
    import json

    kf = KeyFrameData("image")
    kf.add_mask_name("check")

    fname = "test_sketches_for_crvs.json"
    sk_check = None
    try:
        with open(fname, 'r') as f:
            my_data = json.load(f)
            sk_check = SketchedCurve.read_json(my_data)

        kf.add_sketch(mask_index=0, sketch_curve=sk_check)
    except:
        pass

    fname = "test_key_frame.txt"
    with open(fname, "w") as f:
        json.dump(kf.write_json(), f, indent=2)

    with open(fname, 'r') as f:
        my_data = json.load(f)
        sk_check = KeyFrameData.read_json(my_data)
        KeyFrameData.read_json(my_data, sk_check)
        assert sk_check.image_name == kf.image_name



