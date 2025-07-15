#!/usr/bin/env python3

import numpy as np
import cv2

class CameraProjections():
    def __init__(self, camera_fname=("azure_camera.json", "rgb_half_size"), camera_calibration_fname=("azure_camera_calibration.json", "color"), params={}):
        """ camera_fname has one of the default camera setups in utils/camera_fname[0] - holds image size, field of view
        camera_calibration is the camera calibration matrix for the camera (if any) also in utils/ directory (or give
        full path name)
        params is an alternative for overriding image size and field of view (camera_width_angle)
        @params params - dictionary with values to override width/height/size
        """
        from os.path import exists
        import json

        self.image_size = (1040, 480)
        self.camera_width_angle = 90
        self.camera_height_angle = 90

        self.world_to_image = np.identity(3, dtype=np.float32)
        self.image_distortion_coefs = np.zeros((5,), dtype=np.float32)

        locs = ("./", "treefitting/utils/", "../utils/")
        for loc in locs:
            if exists(loc + camera_fname[0]):
                with open(loc + camera_fname[0]) as f:
                    cam_params = json.load(f)
                    for cam in cam_params["options_list"]:
                        if cam["name"] == camera_fname[1]:
                            print(f"Found camera {cam['name']}")
                            self.image_size = cam["image_size"]
                            self.camera_width_angle = cam["fov"][0]
                            self.camera_height_angle = cam["fov"][1]
                            if "depth_range_m" in cam:
                                self.depth_range = cam["depth_range_m"]

            if exists(loc + camera_calibration_fname[0]):
                print(f"Found camera intrinsics {camera_calibration_fname[0]}")
                with open(loc + camera_calibration_fname[0], "r") as f:
                    cam_params = json.load(f)
                    intrinsic_name = camera_calibration_fname[1] + "_intrinsic"
                    distortion_name = camera_calibration_fname[1] + "_distortion_coef"
                    try:
                        for indx, val in enumerate(cam_params[intrinsic_name]):
                            self.world_to_image[indx // 3, indx % 3] = val
                    except ValueError:
                        pass

                    try:
                        self.image_distortion_coefs = np.array(cam_params[distortion_name], dtype=np.float32)
                    except ValueError:
                        pass

        self.z_near = 1.0
        self.z_far = 100.0
        if "z_near" in params:
            self.z_near = params["z_near"]
        if "z_far" in params:
            self.z_far = params["z_near"]
        if "rgb_camera_width_angle" in params:
            self.rgb_camera_width_angle = params["rgb_camera_width_angle"]
        if "rgb_camera_height_angle" in params:
            self.rgb_camera_height_angle = params["rgb_camera_height_angle"]
        if "depth_camera_width_angle" in params:
            self.depth_camera_width_angle = params["depth_camera_width_angle"]
        if "depth_camera_height_angle" in params:
            self.depth_camera_height_angle = params["depth_camera_height_angle"]
        if "rgb_image_size" in params:
            self.rgb_image_size = params["rgb_image_size"]
        if "depth_image_size" in params:
            self.depth_image_size = params["depth_image_size"]

    def from_image_to_ndc(self, pt_uv):
        """ Convert a point in width/height to -1, 1 x -1, 1 (normalized device coords)
        Image: upper left is 0,0, bottom right is width, height
        NDC: upper left is -1, 1, bottom right is 1, -1
        @params - pt_uv - list/point with u, v
        @return - pt_xy_in_box """

        # Image is 0,0 upper left,
        pt_xy = [2.0 * (pt_uv[i] - self.image_size[i] / 2.0) / self.image_size[i] for i in range(0, 2)]

        # Images are indexed from upper left corner, so y needs to be inverted
        pt_xy[1] *= -1
        return pt_xy

    def from_ndc_to_image(self, pt_xy):
        """Convert a point in -1, 1 x -1, 1 to width, height
        @params pt_xy_in_bx - the point in -1, 1
        @return pt_uv in widthxheight"""
        pt_xy_flip_y = [pt_xy[0], -pt_xy[1]]
        pt_uv = [self.image_size[i] * (pt_xy_flip_y[i] + 1.0) / 2.0 for i in range(0, 2)]

        return pt_uv

    def frame_at_z_near(self, do_far=False):
        """ return left, right, bottom, top for the frame at the near plane
        @param do_far - do far plane
        @return 4 numbers in a list (x left, x right, y_bottom, y_top)"""

        ang_width_half = 0.5 * np.pi * self.camera_width_angle / 180.0
        ang_height_half = 0.5 * np.pi * self.camera_height_angle / 180.0

        if do_far:
            z_depth = self.z_far
        else:
            z_depth = self.z_near

        frame_width = z_depth * np.tan(ang_width_half)
        frame_height = z_depth * np.tan(ang_height_half)

        return [-frame_width, frame_width, -frame_height, frame_height]

    def frustrum_matrix(self):
        """ Opengl-style projection matrix (no center of projection).
        See: https://www.songho.ca/opengl/gl_projectionmatrix.html
        Assumes "up" is y, "left-right" is x, and looking out the camera is -z
        After transformation, the view volume (defined by the near, far clipping planes and the
         angles of the field of view) will be the cube -1, -1, -1 X 1, 1, 1, with -z near going to
         -1 and -z far going to 1, and the upper left corner going to the bottom right corner (Normalized device coordinates,
          or NDC). To get from NDC (box) to image, basically drop z and flip x
        @param params - z_near, z_far, image_size as 2x1 array, camera_width_angle in degrees
        @param do_far - do far plane
        @return 4x4 projection matrix"""
        mat = np.identity(4)
        frame = self.frame_at_z_near(do_far=False)

        left = frame[0]
        right = frame[1]
        bottom = frame[2]
        top = frame[3]

        print(f"Frustrum Frame {frame}")
        mat[0, 0] = 2.0 * self.z_near / (right - left)
        mat[1, 1] = 2.0 * self.z_near / (top - bottom)
        # Shifts due to center of projection not being 0, 0
        mat[0, 2] = (right + left) / (right - left)
        mat[1, 2] = (top + bottom) / (top - bottom)
        # Also known as k - the scaling factor
        mat[2, 2] = -(self.z_far + self.z_near) / (self.z_far - self.z_near)
        mat[2, 3] = -(2.0 * self.z_far * self.z_near) / (self.z_far - self.z_near)
        mat[3, 3] = 0.0
        mat[3, 2] = -1.0

        return mat

    def from_xyz_to_ndc(self, pt_xyz: np.array):
        """ Project the point and do the divide
        @param pt_xyz - numpy array 4x1
        @return point in box -1,1 x -1,1 and 3D depth"""
        pt_xyz_proj = self.frustrum_matrix() @ pt_xyz
        depth = pt_xyz_proj[2] / pt_xyz_proj[3]
        pt_xy_box = [pt_xyz_proj[indx] / pt_xyz_proj[3] for indx in range(0, 2)]

        return pt_xy_box, depth

    def from_xyz_to_image_via_ndc(self, pt_xyz: np.array):
        """ Project the point, do the divide and convert to image point
        @param pt_xyz - numpy array 4x1
        @param do_depth - depth image y/n
        @return point in W, H and 3D depth"""
        pt_xy_ndc, depth = self.from_xyz_to_ndc(pt_xyz)
        return self.from_ndc_to_image(pt_xy_ndc), depth

    def from_xyz_to_image_calibration(self, pt_xyz: np.array):
        """ This uses the calibration matrix to go straight to an image coordinate
        @param pt_xyz - numpy array 4x1
        @param do_depth - depth image y/n
        @return point in W, H and 3D depth"""
        pt_xyz_only = np.zeros((3,))

        # intrinsic matrix from opencv has y flipped and z negated
        pt_xyz_only[0] = pt_xyz[0]
        pt_xyz_only[1] = -pt_xyz[1]
        pt_xyz_only[2] = -pt_xyz[2]

        # Apply the matrix from the camera calibration
        pt_uv = self.world_to_image @ pt_xyz_only[0:3]
        pt_uv /= pt_uv[2]

        _, depth = self.from_xyz_to_ndc(pt_xyz)

        return pt_uv[0:2], depth


if __name__ == '__main__':
    cam_rgb = CameraProjections(camera_fname=("azure_camera.json", "rgb_half_size"),
                                camera_calibration_fname=("azure_camera_calibration.json", "color"),
                                params={})
    cam_depth = CameraProjections(camera_fname=("azure_camera.json", "depth_narrow_unbinned"),
                                  camera_calibration_fname=("azure_camera_calibration.json", "depth"),
                                  params={})

    for cam in (cam_rgb, cam_depth):
        # Check image to ndc, upper left point
        pt_ul_im = np.array([0, 0])
        pt_ul_ndc = cam.from_image_to_ndc(pt_ul_im)
        assert(np.isclose(pt_ul_ndc[0], -1.0))
        assert(np.isclose(pt_ul_ndc[1], 1.0))

        # Go back
        pt_ul_im_back = cam.from_ndc_to_image(pt_ul_ndc)
        assert(np.isclose(pt_ul_im[0], pt_ul_im_back[0]))
        assert(np.isclose(pt_ul_im[1], pt_ul_im_back[1]))

        # Check image to ndc, lower right point
        pt_lr_im = np.array([cam.image_size[0], cam.image_size[1]])
        pt_lr_ndc = cam.from_image_to_ndc(pt_lr_im)
        assert(np.isclose(pt_lr_ndc[0],  1.0))
        assert(np.isclose(pt_lr_ndc[1], -1.0))

        # Go back
        pt_lr_im_back = cam.from_ndc_to_image(pt_lr_ndc)
        assert(np.isclose(pt_lr_im[0], pt_lr_im_back[0]))
        assert(np.isclose(pt_lr_im[1], pt_lr_im_back[1]))

        # Check projection from camera frame (x,y,z looking down -z) to normalized device coords
        pt_xyz_center = np.array([0.0, 0.0, -cam.z_far, 1.0])   # center point
        pt_uv_center, depth_far = cam.from_xyz_to_image_via_ndc(pt_xyz_center)
        assert np.isclose(pt_uv_center[0], cam.image_size[0] / 2.0)
        assert np.isclose(pt_uv_center[1], cam.image_size[1] / 2.0)
        assert np.isclose(depth_far, 1.0)

        # The box around the world at the z near distance
        frame_near = cam.frame_at_z_near()
        pt_ul_world = np.array([frame_near[0], frame_near[3], -cam.z_near, 1.0])
        pt_lr_world = np.array([frame_near[1], frame_near[2], -cam.z_near, 1.0])

        pt_ul_uv, pt_ll_depth = cam.from_xyz_to_image_via_ndc(pt_ul_world)
        pt_lr_uv, pt_ur_depth = cam.from_xyz_to_image_via_ndc(pt_lr_world)

        assert np.isclose(pt_ul_uv[0], 0.0)
        assert np.isclose(pt_ul_uv[1], 0.0)
        assert np.isclose(pt_lr_uv[0], cam.image_size[0])
        assert np.isclose(pt_lr_uv[1], cam.image_size[1])
        assert np.isclose(pt_ll_depth, -1.0)
        assert np.isclose(pt_ur_depth, -1.0)

        # Check that the aspect ratio matches the given angles
        rgb_cam_ang_h = (np.arctan2(frame_near[3], cam.z_near) * 2.0) * 180.0 / np.pi
        assert np.isclose(rgb_cam_ang_h, cam.camera_height_angle)
        aspect_ratio_rgb_image = cam.image_size[0] / cam.image_size[1]
        aspect_ratio_rgb_frame = frame_near[1] / frame_near[3]
        print(f"Difference in aspect ratio: Image {aspect_ratio_rgb_image}, Frame {aspect_ratio_rgb_frame}")

        #  Check camera calibration matrix
        pt_uv_cal, depth_cal = cam.from_xyz_to_image_calibration(pt_xyz_center)
        pt_uv_ul_cal, depth_ll_cal = cam.from_xyz_to_image_calibration(pt_ul_world)
        pt_uv_lr_cal, depth_ur_cal = cam.from_xyz_to_image_calibration(pt_lr_world)

        print(f"Diff between calibration image and pure frustrum:")
        print(f"  center {pt_uv_center[0:2]} {pt_uv_cal[0:2]}")
        print(f"  upper left {pt_ul_uv[0:2]} {pt_uv_ul_cal[0:2]}")
        print(f"  lower right {pt_lr_uv[0:2]} {pt_uv_lr_cal[0:2]}")

    print("Done")


