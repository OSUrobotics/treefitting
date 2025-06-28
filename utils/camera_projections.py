#!/usr/bin/env python3

import numpy as np
import cv2

class CameraProjections():
    def __init__(self, camera_fname="", params={}):
        """ Just set the default parameter values in case there aren't any
        @params params - dictionary with values
        """
        from os.path import exists
        import json

        self.params = {}

        cam_params = {}
        self.world_to_rgb_image = np.identity(3)
        self.world_to_depth_image = np.identity(3)
        self.rgb_image_distortion_coefs = np.zeros((5,))
        self.depth_image_distortion_coefs = np.zeros((5,))
        if exists(camera_fname):
            with open(camera_fname, "r") as f:
                cam_params = json.load(f)
                try:
                    for indx, val in enumerate(cam_params["color_intrinsic"]):
                        self.world_to_rgb_image[indx // 3, indx % 3] = val
                except ValueError:
                    pass

                try:
                    for indx, val in enumerate(cam_params["depth_intrinsic"]):
                        self.world_to_depth_image[indx // 3, indx % 3] = val
                except ValueError:
                    pass

                try:
                    self.rgb_image_distortion_coefs = np.array(cam_params["color_distortion_coef"])
                    self.depth_image_distortion_coefs = np.array(cam_params["depth_distortion_coef"])

        self.z_near = 1.0
        self.z_far = 100.0
        self.camera_width_angle = 90.0
        if "z_near" in params:
            self.z_near = params["z_near"]
        if "z_far" in params:
            self.z_far = params["z_near"]
        if "camera_width_angle" in params:
            self.camera_width_angle = params["camera_width_angle"]


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


def from_box_to_image(params, pt_xy):
    """Convert a point in -1, 1 x -1, 1 to width, height
    @params - has image size
    @params pt_xy_in_bx - the point in -1, 1
    @return pt_uv in widthxheight"""
    im_size = params["image_size"]
    pt_xy_flip_y = [pt_xy[0], pt_xy[1]]
    pt_uv = [im_size[i] * (pt_xy_flip_y[i] + 1.0) / 2.0 for i in range(0, 2)]

    return pt_uv


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


def frustrum_matrix(params: dict):
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


def from_xyz_to_box(params: dict, pt_xyz: np.array):
    """ Project the point and do the divide
    @param params - camera parameters
    @param pt_xyz - numpy array 4x1
    @return point in box -1,1 x -1,1 and 3D depth"""
    pt_xyz_proj = frustrum_matrix(params) @ pt_xyz
    depth = pt_xyz_proj[2]
    pt_xy_box = [pt_xyz_proj[indx] / pt_xyz_proj[3] for indx in range(0, 2)]

    return pt_xy_box, depth


def from_xyz_to_image(params: dict, pt_xyz: np.array):
    """ Project the point, do the divide and convert to image point
    @param params - camera parameters
    @param pt_xyz - numpy array 4x1
    @return point in W, H and 3D depth"""
    pt_xy_box, depth = from_xyz_to_box(params, pt_xyz)
    return from_box_to_image(params, pt_xy_box), depth


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
