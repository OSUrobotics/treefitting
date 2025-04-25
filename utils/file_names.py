#!/usr/bin/env python3

# File name handling
#   Essentially FileNamesSubDirs with no sub directories
from utils.file_names_sub_dirs import FileNamesSubDirs


class FileNames(FileNamesSubDirs):
    @staticmethod
    def alphanumeric_key(key):
        return [lambda c: int(c) if key.isdigit() else key.lower() for c in re_split('([0-9]+)', key)]

    def __init__(self, path, img_type="png"):
        """Make directories/filenames
        @param path: the top level path
        @param img_type: the .png or .jpg or whatever"""
        super().__init__(path=path, img_type=img_type)

    def n_images(self):
        """ How many images total?"""
        return len(self.image_names[0])

    def add_mask_id(self, index, mask_id):
        """ Add another mask id to this image/mask pair
        @param index - always 0 subdir, image, mask
        @param mask_id - should be string
        @return new index"""

        index_subdir = (0, index[0], index[1], index[2])
        new_index_subdr = super().add_mask_id(index=index_subdir, mask_id=mask_id)
        return new_index_subdr[1:]

    def get_image_name(self, index, b_debug_path=False, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_debug_path - use debug path y/n
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        index_subdir = (0, index[0], index[1], index[2])
        return super().get_image_name(index=index_subdir, b_debug_path=b_debug_path, b_add_tag=b_add_tag)

    def get_edge_name(self, index, b_optical_flow=False, b_add_tag=True):
        """ Get the edge image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_optical_flow True if add OF to edge name
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        index_subdir = (0, index[0], index[1], index[2])
        return super().get_edge_name(index=index_subdir, b_optical_flow=b_optical_flow, b_add_tag=b_add_tag)

    def get_flow_image_name(self, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        index_subdir = (0, index[0], index[1], index[2])
        return super().get_flow_image_name(index=index_subdir, b_add_tag=b_add_tag)

    def get_depth_image_name(self, index, b_add_tag=True):
        """ Get the image name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return full image name with path"""

        index_subdir = (0, index[0], index[1], index[2])
        return super().get_depth_image_name(index=index_subdir, b_add_tag=b_add_tag)

    def get_depth_data_name(self, index, b_add_tag=True):
        """ Get the depth csv file name corresponding to the index given by (subdirectory index, image index, -)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the csv tag, y/n
        @return full data file name with path"""

        index_subdir = (0, index[0], index[1], index[2])
        return super().get_depth_data_name(index=index_subdir, b_add_tag=b_add_tag)

    def _get_mask_name(self, index, b_add_tag):
        """ Get JUST the mask name corresponding to the index (no directory)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_add_tag - add the image tag, y/n
        @return just the mask name """

        index_subdir = (0, index[0], index[1], index[2])
        return super()._get_mask_name(index=index_subdir, b_add_tag=b_add_tag)

    def get_mask_name(self, index, b_debug_path=False, b_calculate_path=False, b_add_tag=True):
        """ Get the mask name with path corresponding to the index given by (subdirectory index, image index, mask name, mask id)
        @param index (tuple, either 2 dim or 3 dim, index into sorted lists)
        @param b_debug_path Use debug path y/n
        @param b_calcualte_path Use calculate path y/n [only pick one of these two]
        @param b_add_tag - add the image tag, y/n
        @return full mask name with path"""

        index_subdir = (0, index[0], index[1], index[2])
        return super().get_mask_name(index=index_subdir, b_debug_path=b_debug_path, b_calculate_path=b_calculate_path, b_add_tag=b_add_tag)

    def loop_images(self):
        """ a generator that loops over all of the images and generates an index for each
        The index can be passed to get_image_name to get the actual image name
        @return a tuple that can be used to get the image name"""
        for j, _ in enumerate(self.image_names[i]):
            yield 0, j

    def loop_masks(self, mask_type=""):
        """ a generator that loops over all of the masks and generates an index for each
        The index can be passed to get_mask_name to get the actual mask name
        @param mask_type: Optional parameter; if set, return only masks of the given name (eg trunk)
        @return a tuple that can be used to get the mask name"""
        for j, _ in enumerate(self.image_names[i]):
            for k, mask_name in enumerate(self.mask_names):
                if mask_type == "" or mask_type == mask_name:
                    for m, _ in enumerate(self.mask_ids[0][j][k]):
                        yield 0, j, k, m

    def create_edge_images(self, b_use_optical_flow=False):
        """ Create edge images if they don't already exist
        @param b_use_optical_flow - if optical flow exists, use optical flow image to create edge image"""

        import cv2
        from os.path import exists
        for im_indx in self.loop_images():
            # Read in the RGB or optical flow image
            fname_opt_flow = self.get_flow_image_name(im_indx, b_add_tag=True)
            fname_edge = self.get_edge_name(im_indx, b_add_tag=True)
            fname_rgb = self.get_image_name(im_indx, b_add_tag=True)
            if exists(fname_edge):
                continue
            im = None
            if b_use_optical_flow and exists(fname_opt_flow):
                im = cv2.imread(fname_opt_flow)
            elif exists(fname_rgb):
                im = cv2.imread(fname_rgb)
            else:
                continue

            # Now calculate the edge image
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            image_edge = cv2.Canny(im_gray, 50, 150, apertureSize=3)
            cv2.imwrite(fname_edge, image_edge)


if __name__ == '__main__':

    """ Example envy
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
    """

    #check_read = FileNames.read_filenames(fname_for_json_file)
