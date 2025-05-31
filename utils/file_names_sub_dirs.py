#!/usr/bin/env python3
import os
# File name handling
# Assumptions (directories)
#   Path is the path to the top of the directory
#   Sub_dirs is the list of sub directories in Path (if any)
#     Image names is a list of lists with the image names in it sans any file extension (eg .png)
#   Will create CalculatedData and DebugImages in Path, and re-create the Sub_dirs in those folders
#
# Assumptions (images)
#   Image name is everything before the . or _ (specified by name_separator)
#   Masks: Mask images are imagename_mask with an optional number/name after the mask (if there's more than one)
#   Edge: imagename_edge, stored in CalculatedData (edge image calculated from rgb image)
#   EdgeOF: imagename_edgeOF, stored in CalculatedData (edge image calculated from optical flow image)
#   Flow; imagename_flow, stored in CalculatedData (optical flow image)
#   Depth: imagename_depth (if stored as image) OR imagename_depth.depth_suffix if stored numerically
# The only image files in the directory are either images, masks, depth
#  Edge and optical flow and optical flow edge should be stored in CacluatedData
#
# Default names for calculated data
#   _stats.json - the main eigenvalue/vector
#   _bezier.json - the fitted bezier curve (in 2D)
#
# Iterators
#   General use is get the i,j,k triple for each image. From that you can generate
#     image name, mask name, etc
#   An index is:
#     Which subdirectory (may be only one)
#     Which image
#     Which mask name/type
#     Which id for that mask
#
# Two use cases:
#   Have images and masks: will build file names from the directories/sub directories
#    Functions to call (in order)
#      add_mask_name   (for each mask name you have) OR set self.mask_names directly
#      Set other file name parameters
#          self.image_tag and self.mask_tag (defaults to png)
#          self.name_seperator and self.mask_id_seperator (defaults to underscore _)
#      add_sub_directories  (has an optional filter parameter to filter file names by a string)
#      add_directory (same as above - but assumes files are in path)
#
#   Option two: Incrementally add masks/mask id images
#       do add_mask_name as above
#       Set other file name parameters as above
#       Call add_mask_id with the new mask id; uses indexing scheme above
#
# There are several methods for getting a full path name from an index for each type of image (get_*)
# There are several methods for generating lists of names (loop_*)

from glob import glob
import json
from os.path import exists, isdir
from os import mkdir, getcwd
from shutil import copyfile


class FileNamesSubDirs:
    @staticmethod
    def get_path():
        """ return the path up to the home directory"""
        path_to_here = os.getcwd()
        name_split = path_to_here.split("/")
        assert "pycharm" in name_split[3].lower()
        return "/" + name_split[1] + "/" + name_split[2] + "/"

    @staticmethod
    def fix_path(fname):
        """ Replace the first 3 dirs with current working directory"""
        name_split = fname.split("/")
        return FileNamesSubDirs.get_path() + "/".join(name_split[3:])

    @staticmethod
    def alphanumeric_key(key):
        numbers = ''.join(filter(str.isdigit, key))
        if len(numbers) == 0:
            return key.lower()
        else:
            return int(numbers)

    def __init__(self, path, img_type="png"):
        """Make directories/filenames
        @param path: the top level path
        @param img_type: the .png or .jpg or whatever"""
        if path:
            if path[-1] != '/':
                path = path + '/'

        self.path = path
        self.path_debug = path + "DebugImages/"
        self.path_calculated = path + "CalculatedData/"

        # Parameters that you can change
        self.name_seperator = "_"

        if not exists(self.path_debug):
            mkdir(self.path_debug)
        if not exists(self.path_calculated):
            mkdir(self.path_calculated)

        # Keep the file names here
        self.mask_names = []   # List of possible mask name(s) (list)
        self.sub_dirs = []     # List of subdirectories (may be empty)
        self.image_names = []  # For each subdirectory (list) a list of image names; so list of lists
        self.mask_ids = []     # Mask ids for each image and mask name combo, (0, 1, etc) (list of lists of lists of lists)

        self.image_name = "rgb"
        self.edge_name = "edge"          # Tag for edge image made from rgb image
        self.edge_flow_name = "edgeOF"   # Tag for edge image made from optical flow
        self.flow_name = "flow"          # Tag for optical flow images
        self.depth_name = "depth"        # Tag for depth images
        self.depth_suffix = "csv"        # What file ending the depth images have (stored as an array of numbers_

        # For if all the images are, eg, RGB.jpg
        self.image_tag = "." + img_type
        # Optional if mask image types have a different image format
        self.mask_tag = self.image_tag
        self.mask_id_separator = self.name_seperator  # Could be _ set in add masks

    def n_masks(self):
        return len(self.mask_names)

    def n_mask_ids(self, subdir=0, image=0, mask=0):
        return len(self.mask_ids[subdir][image][mask])

    def _find_files(self, path, name_filter=""):
        """ Find all of the image files in the given directory
        Make sure self.image_tag is set as well as self.name_separator (assumes "_")
        Example 1: If the name is name.png, and all other files are name_blah.png, then set name_separator to .
        #Example 2: If the name has an RGB in it, then set name_filter to be RGB
        @param path: The directory to look in
        @param name_filter: If not none, all image names need to have this in their name
        @returns a list of image names, sorted by number in name, if any"""
        search_path = f"{path}*{name_filter}*" + self.image_tag
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No files in directory {search_path}")

        self.image_name = name_filter

        ret_names = []
        for n in fnames:
            # Get rid of the path
            im_name = str.split(n, "/")[-1]
            # image name no .xxx extension
            im_name_no_extension = im_name[0:-len(self.image_tag)]

            ret_names.append(im_name_no_extension)

        ret_names.sort(key=FileNamesSubDirs.alphanumeric_key)
        return ret_names

    def _add_mask_image_ids(self):
        """Loop over all of the subdir, image, and mask names and find all images that match
        subdir/imagename_maskname_id.mask_tag   - assuming seperator is _
        Assumes that mask_names, mask_tag, and mask_id_separator are set"""

        if len(self.mask_names) == 0:
            self.mask_names = ["trunk"]
            print("Warning, no mask names")

        # Loop over all sub directories, all images
        self.mask_ids = []
        for i, d in enumerate(self.sub_dirs):
            self.mask_ids.append([])
            for im_name in self.image_names[i]:
                self.mask_ids[-1].append([])  # a list for every image
                for mask_name in self.mask_names:
                    name_to_search = f"{im_name}{self.name_seperator}{mask_name}{self.mask_id_separator}"
                    len_mask_name = len(name_to_search)
                    search_path = f"{self.path}/{d}/{name_to_search}*{self.mask_tag}"

                    fnames = glob(search_path)
                    self.mask_ids[-1][-1].append([])  # a list for every image-mask pair (may be empty)
                    for full_path_name in fnames:
                        fname = str.split(full_path_name, "/")[-1]

                        # Just the part of the name
                        #    May be empty...
                        mask_id_name = fname[len_mask_name:-len(self.mask_tag)]
                        self.mask_ids[-1][-1][-1].append(mask_id_name)

                    # Sort the list
                    self.mask_ids[-1][-1][-1].sort(key=FileNamesSubDirs.alphanumeric_key)

    def add_directory(self, name_filter=""):
        """Assumes all of the images are in a top-level directory (path) - no subdirectories
        Make sure self.image_tag is set as well as self.name_separator (assumes "_")
        Also make sure mask_names set
        @param name_filter: Optional; requires name_filter to be in the file name
        @return None"""
        # No subdirectory, set to be blank
        self.sub_dirs = [""]
        self.image_names = []
        self.mask_ids = []
        # This function does the hard work
        self.image_names.append(self._find_files(self.path, name_filter=name_filter))
        self._add_mask_image_ids()

    def add_sub_directories(self, dir_name_filter="", im_name_filter=""):
        """Process all the sub directories in path and add their image names
        Also makes sub directory folders in CalculatedData and Debug images
        @param dir_name_filter - Optional tag for directory sub names, eg, "row"
        @param im_name_filter - Optional tag for imgaes, eg, _rgb"""
        search_path = f"{self.path}{dir_name_filter}*"
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No sub directories in directory {search_path}")

        self.sub_dirs = []
        self.image_names = []
        self.mask_ids = []
        fnames.sort(key=FileNamesSubDirs.alphanumeric_key)
        for n in fnames:
            if not isdir(n):
                continue
            if "CalculatedData" in n or "DebugImages" in n:
                continue

            im_names = self._find_files(n + "/", name_filter=im_name_filter)
            if im_names is []:
                print(f"Warning, subdirectory {n} is empty")
            else:
                self.sub_dirs.append(str.split(n, "/")[-1])
                self.image_names.append(im_names)

                path_debug = self.path_debug + self.sub_dirs[-1]
                if not exists(path_debug):
                    mkdir(path_debug)

                path_calculated = self.path_calculated + self.sub_dirs[-1]
                if not exists(path_calculated):
                    mkdir(path_calculated)
        # Get any mask id names
        self._add_mask_image_ids()

    def add_mask_name(self, mask_type_name):
        """ Add another mask type/name to the list
                Will make empty mask_id lists for that name
        @param mask_type_name - actual name to use
        @return index of mask id"""

        for ind, n in enumerate(mask_type_name):
            if n == mask_type_name:
                print(f"Mask name {n} already exists")
                return 0, 0, ind, 0
        # Add the actual name
        self.mask_names.append(mask_type_name)

        # Add the mask id lists
        for i, _ in enumerate(self.mask_ids):
            for j, _ in enumerate(self.mask_ids[i]):
                # One new list for the mask for each image
                self.mask_ids[i][j].append([])
        return 0, 0, len(self.mask_names) - 1, 0

    def add_mask_id(self, index, mask_id):
        """ Add another mask id to this image/mask pair
        @param index - which subdir, image, mask
        @param mask_id - should be string
        @return new index"""

        self.mask_ids[index[0]][index[1]][index[2]].append(mask_id)
        ret_index = (index[0], index[1], index[2], len(self.mask_ids[index[0]][index[1]][index[2]]))
        return ret_index

    def get_image_name_no_path(self, index):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @return just the image name"""

        assert len(index) == 4

        im_name = ""
        if self.sub_dirs[index[0]] != "":
            im_name = im_name + self.sub_dirs[index[0]] + "/"

        im_name = im_name + self.image_names[index[0]][index[1]]
        if self.image_name != "":
            im_name = im_name + self.name_seperator + self.image_name

        return im_name

    def get_image_name(self, index, b_debug_path=False, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_debug_path - use debug path y/n
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        assert len(index) == 4

        if b_debug_path:
            im_name = self.path_debug
        else:
            im_name = self.path

        if self.sub_dirs[index[0]] != "":
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]]
        if self.image_name != "":
            im_name = im_name + self.name_seperator + self.image_name
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_edge_name(self, index, b_rgb=False, b_optical_flow=False, b_depth=False, b_add_tag=True):
        """ Get the edge image name corresponding to the index given by (subdirectory index, image index, -)
        Note: Only one of b_rgb, r_optical_flow, b_depth should be true....
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_optical_flow True if add OF to edge name
        @param b_rgb True if add RGB to edge name
        @param b_bepth True if add _dpth to edge name
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        assert len(index) == 4

        im_name = self.path_calculated
        im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + self.name_seperator + "edge"
        if b_optical_flow:
            im_name = im_name + self.name_seperator + "OF"
        elif b_depth:
            im_name = im_name + self.name_seperator + "DPTH"
        elif b_rgb:
            im_name = im_name + self.name_seperator + "RGB"

        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_flow_image_name(self, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        assert len(index) == 4

        im_name = self.path_calculated
        if len(self.sub_dirs[index[0]]) > 0:
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + self.name_seperator + "flow"
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_depth_image_name(self, index, b_debug_path=False, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_debut_path - which path to use
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        assert len(index) == 4

        if b_debug_path:
            im_name = self.path_debug
        else:
            im_name = self.path

        if len(self.sub_dirs[index[0]]) > 0:
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + self.name_seperator + "depth"
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_depth_data_name(self, index, b_add_tag=True):
        """ Get the depth csv file name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the csv tag, y/n
        @return full data file name with path"""

        assert len(index) == 4

        f_name = self.path
        if len(self.sub_dirs[index[0]]) > 0:
            f_name = f_name + self.sub_dirs[index[0]] + "/"
        f_name = f_name + self.image_names[index[0]][index[1]] + self.name_seperator + "depth"
        if b_add_tag:
            f_name = f_name + ".csv"

        return f_name

    def _get_mask_name(self, index, b_add_tag):
        """ Get JUST the mask name corresponding to the index (no directory)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return just the mask name """
        assert len(index) == 4

        image_name = self.image_names[index[0]][index[1]]
        mask_name = self.mask_names[index[2]]

        # No mask id for this mask
        if len(self.mask_ids[index[0]][index[1]][index[2]]) <= index[3]:
            mask_id = ""
        else:
            mask_id = f"{self.mask_ids[index[0]][index[1]][index[2]][index[3]]}"

        mask_name_full = image_name + self.name_seperator + mask_name + self.mask_id_separator + mask_id

        if b_add_tag:
            mask_name_full = mask_name_full + self.mask_tag
        return mask_name_full

    def get_mask_name(self, index, b_debug_path=False, b_calculate_path=False, b_add_tag=True):
        """ Get the mask name with path corresponding to the index given by (subdirectory index, image index, mask name, mask id)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_debug_path Use debug path y/n
        @param b_calcualte_path Use calculate path y/n [only pick one of these two]
        @param b_add_tag - add the image tag, y/n
        @return full mask name with path"""
        assert len(index) == 4

        if b_debug_path:
            im_name = self.path_debug
        elif b_calculate_path:
            im_name = self.path_calculated
        else:
            im_name = self.path

        im_name = im_name + self.sub_dirs[index[0]] + "/" + self._get_mask_name(index=index, b_add_tag=b_add_tag)

        return im_name

    def loop_images(self):
        """ a generator that loops over all of the images and generates an index for each
        The index can be passed to get_image_name to get the actual image name
        @return a tuple that can be used to get the image name"""
        for i, _ in enumerate(self.sub_dirs):
            for j, _ in enumerate(self.image_names[i]):
                yield i, j, 0, 0

    def loop_masks(self, mask_type=""):
        """ a generator that loops over all of the masks and generates an index for each
        The index can be passed to get_mask_name to get the actual mask name
        @param mask_type: Optional parameter; if set, return only masks of the given name (eg trunk)
        @return a tuple that can be used to get the mask name"""
        for i, _ in enumerate(self.sub_dirs):
            for j, _ in enumerate(self.image_names[i]):
                for k, mask_name in enumerate(self.mask_names):
                    if mask_type == "" or mask_type == mask_name:
                        for m, _ in enumerate(self.mask_ids[i][j][k]):
                            yield i, j, k, m

    def check_names(self):
        """ Run through all the image/mask names and make sure they exist"""
        for ind in self.loop_images():
            im_name = self.get_image_name(index=ind, b_add_tag=True)
            if not exists(im_name):
                raise ValueError(f"Filename {im_name} does not exist")

        for ind in self.loop_masks():
            im_name = self.get_mask_name(index=ind, b_add_tag=True)
            if not exists(im_name):
                raise ValueError(f"Filename {im_name} does not exist")

    def write_json(self):
        """Create a dictionary and return it"""
        foo = FileNamesSubDirs(self.path)
        
        my_dict = {"Name": "FileNamesSubDirs", "data" : {}}
        for k, v in foo.__dict__.items():
            my_dict["data"][k] = self.__dict__[k]
        return my_dict

    @staticmethod
    def read_json(json_dict, file_names_sub_instance=None):
        """ Read back in from json file
        @param json_dict - dictionary read in from file
        @param file_names_sub_instance - an existing of this class to put the data in"""
        if json_dict["Name"] != "FileNamesSubDirs":
            raise ValueError(f"This is not a FileNamesSub dictionary {json_dict}")

        if not file_names_sub_instance:
            file_names_sub_instance = FileNamesSubDirs("")

        for k, v in json_dict["data"].items():
            setattr(file_names_sub_instance, k, v)

        file_names_sub_instance.path = FileNamesSubDirs.fix_path(file_names_sub_instance.path)
        file_names_sub_instance.path_debug = FileNamesSubDirs.fix_path(file_names_sub_instance.path_debug)
        file_names_sub_instance.path_calculated = FileNamesSubDirs.fix_path(file_names_sub_instance.path_calculated)
        return file_names_sub_instance

    def create_edge_images(self, b_use_optical_flow=False):
        """ Create edge images if they don't already exist
        @param b_use_optical_flow - if optical flow exists, use optical flow image to create edge image"""

        import cv2
        from os.path import exists
        for im_indx in self.loop_images():
            # Read in the RGB or optical flow image
            im_index_full = (im_indx[0], im_indx[1], 0, 0)
            fname_edge_rgb = self.get_edge_name(im_index_full, b_rgb=True, b_add_tag=True)
            fname_edge_flow = self.get_edge_name(im_index_full, b_optical_flow=True, b_add_tag=True)
            fname_edge_depth = self.get_edge_name(im_index_full, b_depth=True, b_add_tag=True)
            fname_rgb = self.get_image_name(im_index_full, b_add_tag=True)
            fname_opt_flow = self.get_flow_image_name(im_index_full, b_add_tag=True)
            fname_depth = self.get_depth_data_name(im_index_full, b_add_tag=True)
            if not exists(fname_edge_rgb) and exists(fname_rgb):
                im = cv2.imread(fname_rgb)
                # Now calculate the edge image
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                image_edge = cv2.Canny(im_gray, 100, 250, apertureSize=3)
                cv2.imwrite(fname_edge_rgb, image_edge)

            if not exists(fname_edge_flow) and exists(fname_opt_flow):
                im = cv2.imread(fname_opt_flow)
                # Now calculate the edge image
                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                image_edge = cv2.Canny(im_gray, 1, 10, apertureSize=3)
                cv2.imwrite(fname_edge_flow, image_edge)

            if not exists(fname_edge_depth) and exists(fname_depth):
                im = cv2.imread(fname_depth)
                im_gray = im[:, :, 2].squeeze()
                # Now calculate the edge image
                #im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                image_edge = cv2.Canny(im_gray, 100, 300, apertureSize=3)
                cv2.imwrite(fname_edge_depth, image_edge)


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="PycharmProjects/data/bush_8_west/", type=str, help="where to grab images from")
    parser.add_argument('--filenames', default="all_fnames.json", type=str, help="which file fnames to check")

    args = parser.parse_args()

    path_start = FileNamesSubDirs.get_path()
    fname = path_start + args.path + args.filenames

    f_check = FileNamesSubDirs("")
    with open(fname, 'r') as f:
        my_data = json.load(f)
        FileNamesSubDirs.read_json(my_data, f_check)
