#!/usr/bin/env python3

# Handle file names when processing images in directories
# Assumptions (directories)
#   Path is the path to the top of the directory
#   Sub_dirs is the list of sub directories in Path (if any)
#     Image names is a list of lists with the image names in it sans any .png or _RGB.png
#   Will create CalculatedData and DebugImages in Path, and re-create the Sub_dirs in those folders
#
# Assumptions (images)
#   Image name is everything before the . or _ (given by name_separator)
#   Masks: Mask images are image name_mask with an optional number after the mask (if there's more than one)
#   Edge: image name_edge, stored in CalculatedData
#   Flow; image name_flow
#   Depth: image name_depth
# The only image files in the directory are either images, masks, edge, flow or depth
#
# Default names for calculated data
#   _stats.json - the main eigenvalue/vector
#   _bezier.json - the fitted bezier curve (in 2D)
#
# Iterators
#   General use is get the i,j,k triple for each image. From that you can generate
#     image name, mask name, etc

from glob import glob
import json
from os.path import exists, isdir
from os import mkdir


class HandleFileNames:
    def __init__(self, path, img_type="png"):
        """Make directories/filenames
        @param path: the top level path
        @param img_type: the .png or .jpg or whatever"""
        self.path = path
        self.path_debug = path + "DebugImages/"
        self.path_calculated = path + "CalculatedData/"

        if not exists(self.path_debug):
            mkdir(self.path_debug)
        if not exists(self.path_calculated):
            mkdir(self.path_calculated)

        # List of sub directoriew
        self.sub_dirs = []
        self.image_names = []  # List of image names in each sub directory (list of lists)
        self.mask_names = []   # List of mask name(s) for each image (list of lists of lists)
        self.mask_ids = []     # Mask ids for each mask_name (0, 1, etc) (list of lists of lists of lists)
        self.flow_name = ""    # What is the tag (eg, _flow) for the flow images
        self.depth_name = ""   # What is the tag (eg, _depth) for the depth images

        # For if all the images are, eg, RGB.jpg
        self.image_tag = "." + img_type
        self.mask_tag = ""
        self.mask_id_separator = ""  # Could be _, set in add masks

    def _find_files(self, path, name_filter="", name_separator=""):
        """ Find all of the image files in the given directory
        Example 1: If the name is name.png, and all other files are name_blah.png, then set name_separator to .
        #Example 2: If the name has an RGB in it, then set name_filter to be RGB
        @param path: The directory to look in
        @param name_filter: If not none, all image names need to have this in their name
        @param name_separator: Usually _ or.; the last character before the image name
        @returns a list of image names"""
        search_path = f"{path}*{name_filter}*" + self.image_tag
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No files in directory {search_path}")

        ret_names = []
        for n in fnames:
            im_name = str.split(n, "/")[-1]
            im_name_split = im_name[0:-len(self.image_tag)]
            if name_separator:
                im_name_split = str.split(im_name_split, name_separator)[0]

            if im_name_split not in ret_names:
                ret_names.append(im_name_split)

        ret_names.sort()
        return ret_names

    def add_directory(self, name_filter="", name_separator=""):
        """Assumes all of the images are in the top-level directory
        @param name_filter: Optional tag for imgaes, eg, _rgb
        @param name_separator: Optional separator for the start of the name (eg . or _)"""
        self.sub_dirs = [""]
        self.image_names = []
        self.mask_names = []
        self.mask_ids = []
        self.image_names.append(self._find_files(self.path, name_filter=name_filter, name_separator=name_separator))
        self.mask_names.append([[] for _ in self.image_names[0]])
        self.mask_ids.append([[] for _ in self.image_names[0]])

    def add_sub_directories(self, dir_name_filter="", im_name_filter="", im_name_separator=""):
        """Process all the sub directories in path and add their names
        Also makes sub directory folders in CalculatedData and Debug images
        @param dir_name_filter: Optional tag for directory sub names, eg, "row"
        @param im_name_filter: Optional tag for imgaes, eg, _rgb
        @param im_name_separator: Optional separator for the start of the name (eg . or _)"""
        search_path = f"{self.path}{dir_name_filter}*"
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No sub directories in directory {search_path}")

        self.sub_dirs = []
        self.image_names = []
        self.mask_names = []
        self.mask_ids = []
        fnames.sort()
        for n in fnames:
            if not isdir(n):
                continue

            im_names = self._find_files(n + "/", name_filter=im_name_filter, name_separator=im_name_separator)
            if im_names is []:
                print(f"Warning, subdirectory {n} is empty")
            else:
                self.sub_dirs.append(str.split(n, "/")[-1])
                self.image_names.append(im_names)
                self.mask_names.append([[] for _ in self.image_names[-1]])
                self.mask_ids.append([[] for _ in self.image_names[-1]])

                path_debug = self.path_debug + self.sub_dirs[-1]
                if not exists(path_debug):
                    mkdir(path_debug)

                path_calculated = self.path_calculated + self.sub_dirs[-1]
                if not exists(path_calculated):
                    mkdir(path_calculated)

    def add_mask_images(self, mask_names):
        """Loop over all of the images and find all the mask images that go with it
        Assumes that masks can be number 0, 1, etc
        @param mask_names - possible names for mask images (eg mask, sidebranch, trunk...)"""

        # Loop over all sub directories, all images
        for i, d in enumerate(self.sub_dirs):
            for j, im_name in enumerate(self.image_names[i]):
                for k, mask_name in enumerate(mask_names):
                    search_path = f"{self.path}{d}/{im_name}_{mask_name}*"
                    mask_name_start = len(f"{self.path}{d}/{im_name}")
                    fnames = glob(search_path)
                    b_found_mask_name = False
                    self.mask_names[i][j].append(mask_name)
                    self.mask_ids[i][j].append([])
                    for n in fnames:
                        mask_name_im = n[mask_name_start:-4]
                        if mask_name_im[0] == "_":
                            mask_name_im = mask_name_im[1:]
                        if self.mask_tag == "":
                            self.mask_tag = n[-4:]
                        else:
                            check_mask_tag = n[-4:]
                            if not self.mask_tag == check_mask_tag:
                                raise ValueError(f"Masks not all same time {self.mask_tag}, {check_mask_tag}")
                        if len(mask_name_im) == len(mask_name):
                            if not b_found_mask_name:
                                # No, eg, mask 0, 1, 2 etc - so just add the one name and -1 for the mask id
                                self.mask_id_separator = ""
                                self.mask_ids[i][j][k].append(-1)
                                b_found_mask_name = True
                            else:
                                raise ValueError("Multiple files with same mask name but not different ids {fnames}")
                        else:
                            mask_bit_left = mask_name_im[len(mask_name):]
                            if not b_found_mask_name:
                                # Add the mask name and the index
                                if not str.isnumeric(mask_bit_left):
                                    self.mask_id_separator = mask_bit_left[0]
                                self.mask_ids[i][j].append([])
                                b_found_mask_name = True
                            if not self.mask_id_separator == "":
                                mask_bit_left = mask_bit_left[1:]
                            self.mask_ids[i][j][k].append(int(mask_bit_left))
                    self.mask_ids[i][j][k].sort()

    def get_image_name(self, path, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param path should be one of self.path, self.path_calculated, or path_debug
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        im_name = path
        if len(self.sub_dirs[index[0]]) > 0:
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]]
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_edge_image_name(self, path, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param path should be one of self.path, self.path_calculated, or path_debug
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        im_name = path
        if len(self.sub_dirs[index[0]]) > 0:
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + "_edge"
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_mask_name(self, path, index, b_add_tag=True):
        """ Get the mask name corresponding to the index given by (subdirectory index, image index, mask name, mask id)
        @param path should be one of self.path, self.path_calculated, or path_debug
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full mask name with path"""
        im_name = self.get_image_name(path, index, b_add_tag=False)
        im_name = im_name + "_" + self.mask_names[index[0]][index[1]][index[2]]
        mask_id = self.mask_ids[index[0]][index[1]][index[2]][index[3]]
        if mask_id != -1:
            im_name = im_name + self.mask_id_separator + str(mask_id)
        if b_add_tag:
            im_name = im_name + self.mask_tag
        return im_name

    def loop_images(self):
        """ A generator that loops over all of the images and generates an index for each
        The index can be passed to get_image_name to get the actual image name
        @return a tuple that can be used to get the image name"""
        for i, _ in enumerate(self.sub_dirs):
            for j, _ in enumerate(self.image_names[i]):
                yield i, j

    def loop_masks(self, mask_type=""):
        """ A generator that loops over all of the masks and generates an index for each
        The index can be passed to get_mask_name to get the actual mask name
        @param mask_type: Optional parameter; if set, return only masks of the given name (eg trunk)
        @return a tuple that can be used to get the mask name"""
        for i, _ in enumerate(self.sub_dirs):
            for j, _ in enumerate(self.image_names[i]):
                for k, m in enumerate(self.mask_names[i][j]):
                    if mask_type == "" or mask_type == m:
                        for mask_id, _ in enumerate(self.mask_ids[i][j][k]):
                            yield i, j, k, mask_id

    def check_names(self):
        """ Run through all the image/mask names and make sure they exist"""
        for ind in self.loop_images():
            im_name = self.get_image_name(self.path, ind, b_add_tag=True)
            if not exists(im_name):
                raise ValueError(f"Filename {im_name} does not exist")

        for ind in self.loop_masks():
            im_name = self.get_mask_name(self.path, ind, b_add_tag=True)
            if not exists(im_name):
                raise ValueError(f"Filename {im_name} does not exist")

    def write_filenames(self, fname):
        """json dump this file list to fname
        @param fname file to dump to"""

        with open(fname, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @staticmethod
    def read_filenames(fname):
        """ Read in all the variables and put them back in the class
        @param fname file to read from
        @return a Handle File Names instance"""
        with open(fname, "r") as f:
            my_data = json.load(f)

            handle_files = HandleFileNames(my_data["path"])
            for k, v in my_data.items():
                setattr(handle_files, k, v)

        return handle_files


if __name__ == '__main__':
    # Example 2
    path_bpd = "./data/forcindy/"
    all_files = HandleFileNames(path_bpd)
    # Filename is, eg, 0.png
    all_files.add_directory(name_separator="_")
    all_files.add_mask_images(["trunk", "sidebranch"])
    all_files.write_filenames("./data/forcindy_fnames.json")
    all_files.check_names()

    for ind_img in all_files.loop_images():
        print(f"{all_files.get_image_name(all_files.path, index=ind_img, b_add_tag=True)}")

    for ind_msk in all_files.loop_masks("trunk"):
        print(f"{all_files.get_mask_name(all_files.path_calculated, index=ind_msk, )}")

    # Example 1
    path_trunk_seg = "./data/trunk_segmentations/"
    all_files_trunk = HandleFileNames(path_trunk_seg)
    all_files_trunk.image_tag = "_img.png"
    all_files_trunk.add_sub_directories(dir_name_filter="row", im_name_separator="_")
    all_files_trunk.add_mask_images(["mask"])
    all_files_trunk.write_filenames("./data/trunk_segmentation_names.json")
    all_files_trunk.check_names()

    check_read = HandleFileNames.read_filenames("./data/trunk_segmentation_names.json")
