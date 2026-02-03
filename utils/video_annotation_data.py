#!/usr/bin/env python3
# Video annotation data format, built off of FileNames
# Assumptions (video)
#   Video has been extracted into a sequence of RGB images (see FileNames naming)
#     Image names is a list of lists with the image names in it sans any file extension (eg .png)
#   Camera motion is (mostly) either a pan or a rotate around a point or a zoom in
#   There is a single tree the camera is focused on, with the trunk visible at some point
#
# Manual labeling (video sequence)
#   Sequence has a start and stop point, with one camera motion between those two points
#      May have more than one sequence in a video frame
#   Sequence has a labeling "skip" value - i.e., every 10th image is labeled
#       These are the keyframes
#   Each keyframe has a 2D image vector (and an optional scale value) that best represents the 2D transform between
#     this frame and the next
#
# Manual labeling (key frame)
#   2D sketch curve (axis + radii) for each of the following:
#     The trunk
#     Main support branches (left and right)
#   2D sketch curve (assumed 1-10 pixel radii) for all tertiary branches
#   Conversion of the sketch to a 2D b_spline_cyl
#
# 3D generated data (key frame, from pix2pix or equiv)
#   Input: How many points to sample along the backbone of the curve and in-out along the curve
#   Each keyframe has a 6dof camera transform to the next keyframe
#   For all sketch or radii points on branches/trunks that are visible in the next keyframe
#     The corresponding 2D location in the next frame
#     Estimated 3d location
#
# Image generated data (key frame)
#   Generate a mask image for each trunk/branch
#

import numpy as np
from utils.file_names import FileNames
from utils.file_names_sub_dirs import FileNamesSubDirs
from utils.keyframe_data import KeyFrameData
from draw_routines.image_draw_geom_utils import draw_cross
from utils.camera_projections import CameraProjections


class VideoAnnotationData(FileNames):
    def __init__(self, path, img_type="png"):
        """Make directories/filenames
        @param path: the top level path
        @param img_type: the .png or .jpg or whatever"""
        super().__init__(path, img_type)

        self.start_index = 0
        self.skip_index = 1
        self.end_index = -1

        self.sketch_points_3d = []
        self.crvs_3d = []
        self.matrix_rgb_to_depth = np.identity(3)
        self.keyframes = []

    def n_keyframes(self):
        return len(self.keyframes)

    def add_directory(self, name_filter="", start_index=0, end_index=-1, skip_index=1):
        """Assumes all of the images are in a top-level directory (path) - no subdirectories
        Make sure self.image_tag is set as well as self.name_separator (assumes "_")
        If mask_names is set will also add in mask names
        @param name_filter: Optional; requires name_filter to be in the file name
        @param start_index - where to start adding images from
        @param end_index - where to end, -1 says end of names
        @param skip_index - how many images to skip
        @return None"""
        # No subdirectory, set to be blank
        self.sub_dirs = [""]
        self.image_names = [[]]
        self.mask_ids = [[]]
        # This function does the hard work
        image_names = self._find_files(self.path, name_filter=name_filter)
        print(f"Path {self.path} found {len(image_names)} files filter {name_filter}")

        self.start_index = start_index
        self.end_index = end_index
        self.skip_index = skip_index

        if end_index == -1:
            self.end_index = len(image_names)
        for indx in range(self.start_index, self.end_index, skip_index):
            self.image_names[0].append(image_names[indx])
            self.mask_ids[0].append([])
            for mn in self.mask_names:
                self.mask_ids[0][-1].append([])

        self.keyframes = [KeyFrameData(im_name) for im_name in self.image_names[0]]

        # If mask_names exist, add them in to both key frames and the mask names
        for kf in self.keyframes:
            for mn in self.mask_names:
                kf.add_mask_name(mn)

    def add_mask_name(self, mask_type_name):
        """ Add another mask type/name to the list
                Will make empty mask_id lists for that name
        @param mask_type_name - actual name to use
        @return index of mask id"""
        ret_indx = super().add_mask_name(mask_type_name)
        for kf in self.keyframes:
            kf.add_mask_name(mask_type_name)

        return ret_indx

    def add_sketch(self, image_index, sketch):
        """ Add another mask id to this image/mask pair
        @param image_index - which image
        @param mask_index - which index
        @param sketch - the sketch
        @return index for new sketch"""

        if sketch.n_points() > 2:
            mask_id = len(self.mask_ids[0][image_index[0]][image_index[1]])
            ret_indx = super().add_mask_id(image_index, mask_id)
            self.keyframes[image_index[0]].add_sketch(image_index[1], sketch)

    def replace_sketch(self, image_index, mask_index, id_index, sketch):
        """ Add another mask id to this image/mask pair
        @param image_index - which image
        @param mask_index - which index
        @param sketch - the sketch
        @return index for new sketch"""

        self.keyframes[image_index].replace_sketch(mask_index, id_index, sketch)

    def get_sketch(self, index):
        """ Get the keyframe from the index
        @param index - tuple with image_id, mask_number, mask_id
        @return keyframe"""
        return self.keyframes[index[0]].get_sketch(mask_index=index[1], mask_id_index=index[2])

    def crvs_in_depth_image(self, mat_rgb_to_depth : np.array):
        """ A debugging tool - for each keyframe, output a debug depth image with the curves drawn on top
        @param mat_rgb_to_depth - transform rgb to depth"""
        from PIL import Image
        from draw_routines.b_spline_image import BSplineCylImage
        import utils.matrix_routines_2d as mt

        pt_3d = np.ones((3, 1))

        for kf_indx, kf in enumerate(self.keyframes):
            rgb_image = np.array(Image.open(self.get_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)
            depth_image = np.array(Image.open(self.get_depth_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)
            mat_trans1 = mt.make_translation_matrix(-rgb_image.shape[1]/2, -rgb_image.shape[0]/2)
            mat_trans2 = mt.make_translation_matrix(depth_image.shape[1]/2, depth_image.shape[0]/2)
            mat_scl = mt.make_scale_matrix(depth_image.shape[1] / rgb_image.shape[1], depth_image.shape[0] / rgb_image.shape[0])
            mat = mat_trans2 @ mat_scl @ mat_trans1
            # kf.rgb_to_depth_matrix = mat
            kf.draw_crv_in_image(rgb_image, b_do_spine=False)
            kf.draw_sketch_in_image(rgb_image)
            for mi in range(0, len(kf.bspline_cyls)):
                for m_id in range(0, len(kf.bspline_cyls[mi])):
                    crv = kf.get_bsplinecyl(mi, m_id)
                    depth_crv = kf.get_bsplinecyl_in_depth_image(mat_transform=mat_rgb_to_depth, mask_index=mi, mask_id_index=m_id)

                    #crv_image = BSplineCylImage(crv.points(), crv.degree_name(), crv.radii_crv.points())
                    #crv_image.draw_curve(rgb_image)
                    #crv_image.draw_boundary(rgb_image)

                    crv_image_depth = BSplineCylImage(depth_crv.points(), depth_crv.degree_name(), depth_crv.radii_crv.points())
                    # crv_image_depth.draw_curve(depth_image)
                    crv_image_depth.draw_boundary(depth_image)

            for pt in kf.pts_2d_background:
                draw_cross(rgb_image, pt, (255, 255, 120), thickness=2, length=6)

            pt_depth_corner = np.ones((3, 1))
            for x in (10, depth_image.shape[1] // 2, depth_image.shape[1] - 10):
                pt_depth_corner[0, 0] = x
                for y in (10, depth_image.shape[0] // 2, depth_image.shape[0] - 10):
                    pt_depth_corner[1, 0] = y
                    pt_rgb_corner =  mat_rgb_to_depth @ pt_depth_corner

                    print(f"x, y {x}, {y}  {pt_rgb_corner[0]}, {pt_rgb_corner[1]}")
                    draw_cross(rgb_image, pt_rgb_corner[0:2, 0], (155, 255, 10), thickness=2, length=8)
                    draw_cross(depth_image, (x, y), (155, 255, 10), thickness=2, length=8)

            for pt in kf.pts_2d_background:
                pt_3d[0, 0] = pt[0]
                pt_3d[1, 0] = pt[1]
                pt_depth = mat_rgb_to_depth @ pt_3d
                draw_cross(depth_image, pt_depth[0:2].transpose(), (55, 155, 120), thickness=2, length=6)

            rgb_write = Image.fromarray(rgb_image)
            rgb_name = self.get_image_name((0, kf_indx, 0, 0), b_debug_path=self.path_debug)
            rgb_write.save(rgb_name)

            depth_write = Image.fromarray(depth_image)
            depth_name = self.get_depth_image_name((0, kf_indx, 0, 0), b_debug_path=self.path_debug)
            depth_write.save(depth_name)

            depth_edge = self.get_edge_name((0, kf_indx, 0, 0), b_depth=True, b_add_tag=True)
            if exists(depth_edge):
                #mat_inv = kf.rgb_to_depth_matrix.inverse()
                im_depth_edge = np.array(Image.open(self.get_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)


    def check_names(self):
        """ Run through all the image/mask names and make sure they exist
            Also check that keyframes data is consistent"""
        for ind in self.loop_images():
            im_name = self.get_image_name(index=ind, b_add_tag=True)
            if not exists(im_name):
                raise ValueError(f"Filename {im_name} does not exist")

            if not self.image_name[ind[0]] == self.keyframes[ind[0]].image_name:
                raise ValueError(f"Mismatch {self.image_name[ind[0]]} and keyframe name {self.keyframes[ind[0]].image_name}")

            if not len(self.mask_ids) == len(self.mask_names):
                raise ValueError("Mask ids not right size {len(self.mask_ids)} {len(self.mask_names)}")

            for mask_i, n in enumerate(self.mask_names):
                if not self.keyframes[ind[0]].mask_names[mask_i] == n:
                    raise ValueError(
                        f"Mismatch {n} and keyframe mask name {self.keyframes[ind[0]].mask_names[mask_i]}")
                if not len(self.keyframes[ind[0]].sketch_curves[mask_i]) == len(self.mask_ids[ind[0]][mask_i]):
                    raise ValueError(
                        f"Mismatch {len(self.keyframes[ind[0]].sketch_curves[mask_i])} and number mask ids {len(self.mask_ids[ind[0]][mask_i])}")

    def write_json(self):
        """Create a dictionary and return it"""
        my_dict = {"Name": "Video_annotation_data",
                   "file_name_data" : super().write_json(),
                   "keyframe_data" : [kf.write_json() for kf in self.keyframes],
                   "start_index" : self.start_index,
                   "end_index" : self.end_index,
                   "skip_index" : self.skip_index,
                   "sketch_points_3d" : self.sketch_points_3d,
                   "matrix_rgb_to_depth" : self.matrix_rgb_to_depth.tolist(),
                   "crvs_3d" : [] }
        for crv in self.crvs_3d:
            my_dict["crvs_3d"].append(crv.write_json())
        return my_dict

    @staticmethod
    def read_json(json_dict, video_annotation_instance=None):
        """ Read back in from json file
        @param json_dict - dictionary read in from file
        @param video_annotation_instance - an existing of this class to put the data in"""
        from b_spline_cyl_3d import BSplineCyl3d

        if json_dict["Name"] != "Video_annotation_data":
            raise ValueError(f"This is not a video_annotation_instance dictionary {json_dict}")

        if not video_annotation_instance:
            video_annotation_instance = VideoAnnotationData("")

        FileNames.read_json(json_dict["file_name_data"], video_annotation_instance)
        # Just make sure this comes after above
        video_annotation_instance.keyframes = []
        for kf in json_dict["keyframe_data"]:
            video_annotation_instance.keyframes.append(KeyFrameData.read_json(kf))

        video_annotation_instance.start_index = json_dict["start_index"]
        video_annotation_instance.end_index = json_dict["end_index"]
        video_annotation_instance.skip_index = json_dict["skip_index"]

        if "matrix_rgb_to_depth" in json_dict:
            video_annotation_instance.matrix_rgb_to_depth = np.array(json_dict["matrix_rgb_to_depth"])
        if "sketch_points_3d" in json_dict:
            video_annotation_instance.sketch_points_3d = json_dict["sketch_points_3d"]
        if "crvs_3d" in json_dict:
            video_annotation_instance.crvs_3d = []
            for crv_dict in json_dict["crvs_3d"]:
                video_annotation_instance.crvs_3d.append(BSplineCyl3d.read_json(crv_dict))

        return video_annotation_instance


if __name__ == '__main__':
    import json
    from os.path import exists

    # Grab the current path
    path_start = FileNamesSubDirs.get_path() + "bush_3_east/video_annotation_data.json"

    if exists(path_start):
        va = VideoAnnotationData("path_start")
