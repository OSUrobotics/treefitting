#!/usr/bin/env python3

import numpy as np


def set_default_params(params):
    """ Just set the default parameter values in case there aren't any
    @params params - dictionary with values
    """
    if not "z_near" in params:
        params["z_near"] = 1.0
    if not "z_far" in params:
        params["z_far"] = 100.0
    if not "camera_width_angle" in params:
        params["camera_width_angle"] = 90.0


def from_image_to_box(params, pt_uv):
    """ Convert a point in width/height to -1, 1 x -1, 1
    @params - has image size
    @params - pt_uv - list/point with u, v
    @return - pt_xy_in_box """
    im_size = params["image_size"]
    pt_xy = [2.0 * (pt_uv[i] - im_size[i] / 2.0) / im_size[i] for i in range(0, 2)]

    # Images are indexed from upper left corner, so y needs to be inverted
    pt_xy[1] *= -1
    return pt_xy

def frame_at_z_near(params):
    """ return left, right, bottom, top for the frame at the near plane
    params has information about the camera
    @param params - z_near, z_far, image_size as 2x1 array, camera_width_angle in degrees
    @return 4 numbers in a list"""

    set_default_params(params)
    ang_width_half = 0.5 * np.pi * params["camera_width_angle"] / 180.0

    width_window = params["image_size"][0]
    height_window = params["image_size"][1]
    aspect_ratio = width_window / height_window

    frame_width = params["z_near"] * np.tan(ang_width_half)
    frame_height = frame_width / aspect_ratio

    return [-frame_width, frame_width, -frame_height, frame_height]


def frustrum_matrix(params):
    """ params has information about the camera
    @param params - z_near, z_far, image_size as 2x1 array, camera_width_angle in degrees
    @return 4x4 projection matrix"""
    mat = np.identity(4)
    frame = frame_at_z_near(params)

    left = frame[0]
    right = frame[1]
    bottom = frame[2]
    top = frame[3]

    print(f"Frame {frame}")
    print(f"params {params}")
    mat[0, 0] = 2.0 * params["z_near"] / (right - left) 
    mat[1, 1] = 2.0 * params["z_near"] / (top - bottom)
    # Shifts due to center of projection not being 0, 0
    mat[0, 2] = (right + left) / (right - left)
    mat[1, 2] = (top + bottom) / (top - bottom)
    # Also known as k - the scaling factor
    mat[2, 2] = (params["z_far"] + params["z_near"]) / (params["z_far"] - params["z_near"])
    mat[2, 3] = -(2.0 * params["z_far"] * params["z_near"]) / (params["z_far"] - params["z_near"])
    mat[3, 3] = 0.0
    mat[3, 2] = -1.0

    return mat


if __name__ == '__main__':
    params = {"image_size":[640, 480]}
    # Check image to box
    pt_ul_im = [640, 0]
    pt_ul_xy = from_image_to_box(params, pt_ul_im)
    assert(np.isclose(pt_ul_xy[0], 1.0))
    assert(np.isclose(pt_ul_xy[1], 1.0))

    pt_lr_im = [0, 480]
    pt_lr_xy = from_image_to_box(params, pt_lr_im)
    assert(np.isclose(pt_lr_xy[0], -1.0))
    assert(np.isclose(pt_lr_xy[1], -1.0))
