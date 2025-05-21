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

    def loop_images(self):
        """ a generator that loops over all of the images and generates an index for each
        The index can be passed to get_image_name to get the actual image name
        @return a tuple that can be used to get the image name"""
        for j, _ in enumerate(self.image_names[0]):
            yield 0, j, 0, 0

    def loop_masks(self, mask_type=""):
        """ a generator that loops over all of the masks and generates an index for each
        The index can be passed to get_mask_name to get the actual mask name
        @param mask_type: Optional parameter; if set, return only masks of the given name (eg trunk)
        @return a tuple that can be used to get the mask name"""
        for j, _ in enumerate(self.image_names[0]):
            for k, mask_name in enumerate(self.mask_names):
                if mask_type == "" or mask_type == mask_name:
                    for m, _ in enumerate(self.mask_ids[0][j][k]):
                        yield 0, j, k, m


if __name__ == '__main__':

    import json
    path_start = "/Users/grimmc/"
    # path_start = "/Users/cindygrimm/"
    tree_name = "BP_R1_East_tree2"
    data_path = path_start + "PycharmProjects/data/EnvyTree/" + tree_name + "/"
    key_frame_names = "video_annot.json"
    va_name = data_path + key_frame_names
    with open(va_name, "r") as f:
        my_dict = json.load(f)

    fn = FileNames.read_json(my_dict["file_name_data"])
    fn.create_edge_images()

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
