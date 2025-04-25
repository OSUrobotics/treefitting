#!/usr/bin/env python3
from sphinx.writers.text import my_wrap

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

from utils.file_names import FileNames
from utils.keyframe_data import KeyFrameData


class VideoAnnotationData(FileNames):
    def __init__(self, path, img_type="png"):
        """Make directories/filenames
        @param path: the top level path
        @param img_type: the .png or .jpg or whatever"""
        super().__init__(path, img_type)

        self.start_index = 0
        self.skip_index = 1
        self.end_index = -1

        self.keyframes = []

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

        self.keyframes = [KeyFrameData(im_name) for im_name in self.image_names]

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

    def add_sketch(self, image_index, mask_index, sketch):
        """ Add another mask id to this image/mask pair
        @param image_index - which image
        @param mask_index - which index
        @param sketch - the sketch
        @return index for new sketch"""

        ret_indx = super().add_mask_id((image_index, mask_index, -1))
        self.keyframes[image_index].add_sketch(mask_index, sketch)

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

    def check_names(self):
        """ Run through all the image/mask names and make sure they exist
            Also check that keyframes data is consistent"""
        from os.path import exists
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
                   "skip_index" : self.skip_index}
        return my_dict

    @staticmethod
    def read_json(json_dict, video_annotation_instance=None):
        """ Read back in from json file
        @param json_dict - dictionary read in from file
        @param video_annotation_instance - an existing of this class to put the data in"""
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

        return video_annotation_instance

    @staticmethod
    def read_envy(src_path, dest_path, tree_name, b_get_box_files = True):
        """ Read in one of the Envy trees from Prosser 2023 that has already been extracted from the mkv file
        @param src_path : Complete box drive director
        @param dest_path : Where to put all the files on your harddrive
        @param tree_name : Shortcut name to use for the tree,
        @param b_get_box_files : set to True to do the box copy"""
        from os.path import exists
        from os import mkdir
        from glob import glob
        from shutil import copyfile
        import json
        if b_get_box_files:
            if not exists(dest_path):
                mkdir(dest_path)
            dest_path = dest_path + tree_name
            if not exists(dest_path):
                mkdir(dest_path)

            search_path = f"{src_path}/output_rgb/*.png"
            fnames = glob(search_path)
            for n in fnames:
                # Get rid of the path
                im_name = str.split(n, "/")[-1]
                dest_im_name = dest_path + "/" + im_name
                if exists(dest_im_name):
                    continue
                print(f"copying {n} {dest_im_name}")
                copyfile(n, dest_im_name)
        else:
            dest_path = dest_path + tree_name + "/"

        all_fnames = FileNames(path=dest_path, img_type="png")
        all_fnames.mask_names = ["trunk", "left_support", "right_support", "tertiary"]
        all_fnames.add_directory(name_filter="rgb")
        fname_write = dest_path + "/all_fnames.json"
        all_fnames.image_name = ""
        with open(fname_write, "w") as f:
            json.dump(all_fnames.write_json(), f, indent=2)
        return all_fnames


if __name__ == '__main__':
    import json

    # path_start = "/Users/grimmc/"
    path_start = "/Users/cindygrimm/"
    # box_path = "Library/CloudStorage/Box-Box/"
    box_path = "MyBox/"
    src_box_path = path_start + box_path
    dest_path = path_start + "PycharmProjects/data/EnvyTree/"
    src_tree_pruning_path = src_box_path + "Robotic pruning and thinning/Datasets/2023/Jan 2023 Azure and ZED Videos/OSU Envy Orchard/"
    src_path = src_tree_pruning_path + "BeforePruning/row1East/EAST/tree2"
    tree_name = "BP_R1_East_tree2"
    # fn = VideoAnnotationData.read_envy(src_path=src_path, dest_path=dest_path, tree_name=tree_name, b_get_box_files=False)

    va = VideoAnnotationData(dest_path + tree_name + "/", img_type="png")

    va.add_directory(name_filter="rgb", start_index=0, end_index=115, skip_index=10)

    va_fname = dest_path + tree_name + "/video_annot.json"
    with open(va_fname, "w") as f:
        my_dict = va.write_json()
        json.dump(my_dict, f, indent=2)

    with open(va_fname, "r") as f:
        my_dict = json.load(f)
        va_back = VideoAnnotationData.read_json(my_dict)



