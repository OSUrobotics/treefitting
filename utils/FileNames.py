#!/usr/bin/env python3

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
from os import mkdir, walk


class FileNames:
    def __init__(self, path, img_type="png"):
        """Make directories/filenames
        @param path: the top level path
        @param img_type: the .png or .jpg or whatever"""
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

        self.edge_name = "edge"          # Tag for edge image made from rgb image
        self.edge_flow_name = "edgeOF"   # Tag for edge image made from optical flow
        self.flow_name = "flow"          # Tag for optical flow images
        self.depth_name = "depth"        # Tag for depth images
        self.depth_suffix = "obj"        # What file ending the depth images have (stored as an array of numbers_

        # For if all the images are, eg, RGB.jpg
        self.image_tag = "." + img_type
        # Optional if mask image types have a different image format
        self.mask_tag = self.image_tag
        self.mask_id_separator = self.name_seperator  # Could be _ set in add masks

    def _find_files(self, path, name_filter=""):
        """ Find all of the image files in the given directory
        Make sure self.image_tag is set as well as self.name_separator (assumes "_")
        Example 1: If the name is name.png, and all other files are name_blah.png, then set name_separator to .
        #Example 2: If the name has an RGB in it, then set name_filter to be RGB
        @param path: The directory to look in
        @param name_filter: If not none, all image names need to have this in their name
        @returns a list of image names"""
        search_path = f"{path}*{name_filter}*" + self.image_tag
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No files in directory {search_path}")

        ret_names = []
        for n in fnames:
            # Get rid of the path
            im_name = str.split(n, "/")[-1]
            # image name no .xxx extension
            im_name_no_extension = im_name[0:-len(self.image_tag)]
            if len(self.name_seperator) > 0:
                im_name_split = str.split(im_name_no_extension, self.name_seperator)[0]
            else:
                im_name_split = im_name_no_extension

            # Just in case duplicates - shouldn't really happen
            if im_name_split not in ret_names:
                ret_names.append(im_name_split)

        ret_names.sort()
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
                    self.mask_ids[-1][-1][-1].sort()

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
        fnames.sort()
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
                return (0, 0, ind, 0)
        # Add the actual name
        self.mask_names.append(mask_type_name)

        # Add the mask id lists
        for i, _ in enumerate(self.mask_ids):
            for j, _ in enumerate(self.mask_ids[i]):
                # One new list for the mask for each image
                self.mask_ids[i][j].append([])
        return (0, 0, len(self.mask_names) - 1, 0)

    def add_mask_id(self, index, mask_id):
        """ Add another mask id to this image/mask pair
        @param index - which subdir, image, mask
        @param mask_id - should be string
        @return new index"""

        if index[3] == -1 or len(self.mask_ids[index[0]][index[1]][index[2]]) == 0:
            ret_index = (index[0], index[1], index[2], 0)
        else:
            # Adding the new mask id at the end
            ret_index = (index[0], index[1], index[2], len(self.mask_ids[index[0]][index[1]][index[2]] + 1))

        self.mask_ids[index[0]][index[1]][index[2]].append(mask_id)
        return ret_index

    def get_image_name(self, index, b_debug_path=False, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_debug_path - use debug path y/n
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        if b_debug_path:
            im_name = self.path_debug
        else:
            im_name = self.path

        im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]]
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_edge_name(self, index, b_optical_flow=False, b_add_tag=True):
        """ Get the edge image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_optical_flow True if add OF to edge name
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        im_name = self.path_calculated
        im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + self.name_seperator + "edge"
        if b_optical_flow:
            im_name = im_name + self.name_seperator + "OF"
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_flow_image_name(self, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        im_name = self.path_calculated
        if len(self.sub_dirs[index[0]]) > 0:
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + self.name_seperator + "flow"
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def get_depth_image_name(self, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        im_name = self.path
        if len(self.sub_dirs[index[0]]) > 0:
            im_name = im_name + self.sub_dirs[index[0]] + "/"
        im_name = im_name + self.image_names[index[0]][index[1]] + self.name_seperator + "depth"
        if b_add_tag:
            im_name = im_name + self.image_tag

        return im_name

    def _get_mask_name(self, index, b_add_tag):
        """ Get JUST the mask name corresponding to the index (no directory)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return just the mask name """
        image_name = self.image_names[index[0]][index[1]]
        mask_name = self.mask_names[index[2]]

        # No mask id for this mask
        if len(self.mask_ids[index[0]][index[1]][index[2]]) <= index[3]:
            mask_id = ""
        else:
            mask_id = self.mask_ids[index[0]][index[1]][index[2]][index[3]]

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
                yield i, j

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

    def write_filenames(self, fname):
        """json dump this file list to fname
        @param fname file to dump to"""

        with open(fname, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @staticmethod
    def read_filenames(fname, path=None):
        """ Read in all the variables and put them back in the class
        @param fname file to read from
        @return a Handle File Names instance"""
        if not exists(fname):
            if exists(path + "/" + fname):
                fname = path + "/" + fname
                
        with open(fname, "r") as f:
            my_data = json.load(f)

            if not path:
                path = my_data["path"]

            handle_files = FileNames(path)
            for k, v in my_data.items():
                setattr(handle_files, k, v)

        return handle_files


if __name__ == '__main__':
    from shutil import copyfile
    b_get_box_files = False
    if b_get_box_files:
        dest_path = "/Users/cindygrimm/PyCharmProjects/treefitting/Image_based/data/EnvyTree/"
        if not exists(dest_path):
            mkdir(dest_path)

        tree_search_path = f"/Users/cindygrimm/MyBox/Robotic pruning and thinning/Datasets/2023/Jan 2023 Azure and ZED Videos/OSU Envy Orchard/"
        for (root, dirs, files) in walk(tree_search_path, topdown=True):
            follow_path_name = root[len(tree_search_path):]
            path_pieces = str.split(follow_path_name, "/")
            if "depth" in path_pieces[-1]:
                continue
            sub_dir_name = "_".join(path_pieces[0:-1])
            count = 0
            files.sort()
            n_skip = 10   #max(1, len(files) // 10)
            for nf, ff in enumerate(files):
                if ff[-4:] == ".png" and nf % n_skip == 0:
                    if not exists(dest_path + "/" + sub_dir_name):
                        mkdir(dest_path + "/" + sub_dir_name)
                    copyfile(root + "/" + ff, dest_path + "/" + sub_dir_name + "/" + ff)
                    print(f"{ff}")

    path_bpd_envy = "/Users/cindygrimm/PyCharmProjects/treefitting/Image_based/data/EnvyTree/"
    all_files_envy = FileNames(path_bpd_envy, img_type="png")
    all_files_envy.mask_names = ["trunk", "sidebranch", "tertiary"]
    all_files_envy.add_sub_directories()
    all_files_envy.write_filenames(path_bpd_envy + "envy_fnames.json")
    # Example bb
    """
    path_bpd = "./data/blueberries/"
    all_files = FileNames(path_bpd, img_type="jpg")
    all_files.add_directory(name_filter="rgb", name_separator="_")
    all_files.add_mask_images(["all"])
    all_files.write_filenames("./data/blueberries_fnames.json")
    """

    # Example 2
    b_do_mask = False
    fname_for_json_file = "../Image_based/data/forcindy_bspline.json"
    path_bpd = "../Image_based/data/forcindy/"
    all_files = FileNames(path=path_bpd, img_type="png")
    all_files.mask_names = ["vertical", "side"]
    # Filename is, eg, 0.png
    all_files.add_directory()
    all_files.write_filenames(fname_for_json_file)
    all_files.check_names()

    for ind_img in all_files.loop_images():
        print(f"{all_files.get_image_name(index=ind_img, b_add_tag=True)}")

    for ind_msk in all_files.loop_masks("trunk"):
        print(f"{all_files.get_mask_name(index=ind_msk, )}")

    # Example 1
    """
    path_trunk_seg = "./data/trunk_segmentations/"
    all_files_trunk = HandleFileNames(path_trunk_seg)
    all_files_trunk.image_tag = "_img.png"
    all_files_trunk.add_sub_directories(dir_name_filter="row", im_name_separator="_")
    all_files_trunk.add_mask_images(["mask"])
    all_files_trunk.write_filenames("./data/trunk_segmentation_names.json")
    all_files_trunk.check_names()
    """

    check_read = FileNames.read_filenames(fname_for_json_file)
