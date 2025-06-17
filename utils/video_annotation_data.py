#!/usr/bin/env python3
from draw_routines.image_draw_geom_utils import draw_cross
from fit_routines.bspline_fit_params import BSplineFitParams
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
from os.path import exists


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

    def crvs_in_depth_image(self):
        """ A debugging tool - for each keyframe, output a debug depth image with the curves drawn on top"""
        from PIL import Image
        from draw_routines.b_spline_image import BSplineCylImage
        import utils.matrix_routines_2d as mt

        for kf_indx, kf in enumerate(self.keyframes):
            rgb_image = np.array(Image.open(self.get_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)
            depth_image = np.array(Image.open(self.get_depth_image_name((0, kf_indx, 0, 0)))).astype(np.uint8)
            mat_trans1 = mt.make_translation_matrix(-rgb_image.shape[1]/2, -rgb_image.shape[0]/2)
            mat_trans2 = mt.make_translation_matrix(depth_image.shape[1]/2, depth_image.shape[0]/2)
            mat_scl = mt.make_scale_matrix(depth_image.shape[1] / rgb_image.shape[1], depth_image.shape[0] / rgb_image.shape[0])
            mat = mat_trans2 @ mat_scl @ mat_trans1
            # kf.rgb_to_depth_matrix = mat
            for mi in range(0, len(kf.bspline_cyls)):
                for m_id in range(0, len(kf.bspline_cyls[mi])):
                    crv = kf.get_bsplinecyl(mi, m_id)
                    depth_crv = kf.get_bsplinecyl_in_depth_image(mi, m_id)

                    crv_image = BSplineCylImage(crv.points(), crv.degree_name(), crv.radii_crv.points())
                    #crv_image.draw_curve(rgb_image)
                    crv_image.draw_boundary(rgb_image)

                    crv_image_depth = BSplineCylImage(depth_crv.points(), depth_crv.degree_name(), depth_crv.radii_crv.points())
                    # crv_image_depth.draw_curve(depth_image)
                    crv_image_depth.draw_boundary(depth_image)

            for pt in kf.pts_2d_rgb_depth:
                draw_cross(rgb_image, pt, (255, 255, 120), thickness=2, length=6)
            for pt in kf.depth_pts_in_rgb():
                draw_cross(rgb_image, pt, (155, 255, 220), thickness=2, length=6)

            pt_depth_corner = np.ones((3, 1))
            mat_depth_to_rgb = np.linalg.inv(kf.rgb_to_depth_matrix)
            mat_check = kf.rgb_to_depth_matrix @ mat_depth_to_rgb
            for x in (10, depth_image.shape[1] // 2, depth_image.shape[1] - 10):
                pt_depth_corner[0, 0] = x
                for y in (10, depth_image.shape[0] // 2, depth_image.shape[0] - 10):
                    pt_depth_corner[1, 0] = y
                    pt_rgb_corner =  mat_depth_to_rgb @ pt_depth_corner

                    print(f"x, y {x}, {y}  {pt_rgb_corner[0]}, {pt_rgb_corner[1]}")
                    draw_cross(rgb_image, pt_rgb_corner[0:2, 0], (155, 255, 10), thickness=2, length=8)
                    draw_cross(depth_image, (x, y), (155, 255, 10), thickness=2, length=8)

            for pt in kf.pts_2d_depth:
                draw_cross(depth_image, pt, (55, 155, 120), thickness=2, length=6)
            for pt in kf.rgb_pts_in_depth():
                draw_cross(depth_image, pt, (255, 55, 220), thickness=2, length=6)

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

    @staticmethod
    def read_blueberry(src_path, dest_path, bush_name, b_get_box_files = True):
        """ Read in one of the bleuberry data sets that has already been extracted from the mkv file
        @param src_path : Complete box drive director
        @param dest_path : Where to put all the files on your harddrive
        @param bush_name : Shortcut name to use for the bush,
        @param b_get_box_files : set to True to do the box copy"""
        from os.path import exists
        from os import mkdir
        from glob import glob
        from shutil import copyfile
        import json

        if b_get_box_files:
            if not exists(dest_path):
                mkdir(dest_path)
            dest_path = dest_path + bush_name
            if not exists(dest_path):
                mkdir(dest_path)

            search_path = f"{src_path}{bush_name}/color/*.jpg"
            print(f"Search path {search_path}")
            fnames = glob(search_path)
            for n in fnames:
                # Get rid of the path
                im_name = str.split(n, "/")[-1]
                im_num = FileNamesSubDirs.alphanumeric_key(im_name)
                dest_im_name = dest_path + f"/rgb{im_num:05d}.jpg"
                if not exists(dest_im_name):
                    print(f"copying {n} {dest_im_name}")
                    copyfile(n, dest_im_name)

                depth_im = f"{src_path}{bush_name}/depth/depth_raw_{im_num}.jpg"
                depth_data = f"{src_path}{bush_name}/depth/depth_raw_{im_num}.csv"
                dest_depth_name = dest_path + f"/rgb{im_num:05d}_depth.jpg"
                dest_depth_data_name = dest_path + f"/rgb{im_num:05d}_depth.csv"
                if exists(depth_im):
                        if not exists(dest_depth_name):
                            print(f"copying {depth_im} {dest_depth_name}")
                            copyfile(depth_im, dest_depth_name)
                if exists(depth_data) and not exists(dest_depth_data_name):
                    print(f"copying {depth_data} {dest_depth_data_name}")
                    copyfile(depth_im, dest_depth_data_name)
        else:
            dest_path = dest_path + bush_name + "/"

        all_fnames = FileNames(path=dest_path, img_type="jpg")
        all_fnames.mask_names = ["cane", "branch"]
        all_fnames.add_directory(name_filter="rgb")
        fname_write = dest_path + "/all_fnames.json"
        all_fnames.image_name = ""
        with open(fname_write, "w") as f:
            json.dump(all_fnames.write_json(), f, indent=2)
        return all_fnames


def read_and_rerun(dir_name, annot_name):
    import json

    path_start = FileNamesSubDirs.get_path()
    dest_path = path_start + "PycharmProjects/data/" + dir_name + "/"
    va_fname = dest_path + annot_name + ".json"

    with open(va_fname, "r") as f:
        my_dict = json.load(f)
        va = VideoAnnotationData.read_json(my_dict)

    fit_params = BSplineFitParams()
    fit_params["inlier threshold"] = 8
    fit_params["average fit"] = 5
    fit_params["outlier ratio"] = 0.0
    fit_params["degree"] = "cubic"
    for kf in va.keyframes:
        kf.refit(fit_params=fit_params)

    va_fname_refit = dest_path + "/" + annot_name + "_refit.json"
    with open(va_fname_refit, "w") as f:
        json.dump(va.write_json(), f, indent=2)
    # Draw debug images
    va.crvs_in_depth_image()


def produce_pips2_data(annot_full_path_name, kf=0, backbone_spacing=8, radial_spacing=4, grid_spacing=30):
    """get the N images needed for processing the points along with 2D points
    @param annot_name - annotation name
    @param kf - which keyframe
    @return images and points"""

    import json
    import cv2
    from draw_routines.image_draw_geom_utils import draw_cross

    path_parts = annot_full_path_name.split("/")

    with open(annot_full_path_name, "r") as f:
        my_dict = json.load(f)
        va = VideoAnnotationData.read_json(my_dict)

    full_fname = "/".join(path_parts[:-1]) + "/all_fnames.json"
    with open(full_fname, "r") as f:
        my_dict = json.load(f)
        full_list = FileNames.read_json(my_dict)

    info = {"KeyFrame":kf,
            "BackboneSpacing":backbone_spacing,
            "RadialSpacing":radial_spacing,
            "GridSpacing":grid_spacing}

    im_name = va.keyframes[kf].image_name
    if kf == va.n_keyframes() - 1:
        im_name_next = im_name
        im_name = va.keyframes[kf-1].image_name
    else:
        im_name_next = va.keyframes[kf + 1].image_name

    info["ImageName"] = im_name
    info["ImageNameNext"] = im_name_next
    images = []
    b_in_section = False
    image_size = (960, 540)  # input resolution, H, W
    scl_image = [1, 1]
    shift_image = [0, 0]
    print(f"Reading images")
    b_found_end = False
    for img_indx in full_list.loop_images():
        img_name_full = full_list.get_image_name_no_path(img_indx)
        if img_name_full == im_name:
            b_in_section = True
        if b_in_section:
            print(f" {full_list.get_image_name(img_indx, b_add_tag=True)}")
            im = cv2.imread(full_list.get_image_name(img_indx, b_add_tag=True))
            #im_resize = cv2.resize(im, image_size)
            shift_image = [(im.shape[1] - image_size[0]) // 2, (im.shape[0] - image_size[1]) // 2 ]
            im_resize = im[shift_image[0]:shift_image[0] + image_size[1], shift_image[1]:shift_image[1] + image_size[0], :]
            #scl_image[0] = image_size[0] / im.shape[1]
            #scl_image[1] = image_size[1] / im.shape[0]
            images.append(im_resize)
        if img_name_full == im_name_next:
            b_found_end = True
        if len(images) == 8:
            info["ImageNext"] = img_indx
            break
    info["x_shift"] = shift_image[1]
    info["y_shift"] = shift_image[0]
    info["x_scale"] = scl_image[1]
    info["y_scale"] = scl_image[0]

    pts_2d = []
    for crvs in va.keyframes[kf].bspline_cyls:
        for crv in crvs:
            # Probably too small to track
            if crv.radius(0.5) < 2:
                continue
            crv_len = crv.curve_length() * scl_image[0]
            crv_width = crv.radius(0.5) * scl_image[0]

            n_spacing = max(int(crv_len / backbone_spacing), 2)
            n_across = max(int(crv_width / radial_spacing), 1)

            # Points along the branch
            ts = np.linspace(0, crv.max_t(), n_spacing)
            # Points along the centerline
            pts = crv.eval_crv(ts)
            across_min = 0.1

            # Points moving out from the center (for fat branches)
            across_min = 0.3
            across_max = 0.7
            for perc_across in range(0, n_across // 2):
                p = (perc_across + 0.5) / (n_across * 0.5)
                p_across = across_min + p * (across_max - across_min)
                pts = np.vstack((pts, crv.edge_pts(ts, p_across)))
                pts = np.vstack((pts, crv.edge_pts(ts, -p_across)))

            for pt in pts:
                #pts_2d.append([pt[0] * scl_image[0], pt[1] * scl_image[1]])
                pts_2d.append([pt[0] - shift_image[1], pt[1] - shift_image[0]])
                # Remember image scale is height, width

    pts_keep = []
    im_start = images[0] // 2
    im_end = images[-1] // 2
    vec_move = [va.keyframes[kf].pan_vec[0] * scl_image[0], va.keyframes[kf].pan_vec[1] * scl_image[1]]
    for pt in pts_2d:
        pt_moved = [pt[0] + vec_move[0], pt[1] + vec_move[1]]
        if 0 <= pt_moved[0] <= image_size[0]:
            if 0 <= pt_moved[1] <= image_size[1]:
                pts_keep.append(pt)
                draw_cross(im_start, pt, color=[0, 255, 0])
                draw_cross(im_end, pt_moved, color=[0, 255, 0])
            else:
                draw_cross(im_start, pt, color=[255, 0, 0])
        else:
            draw_cross(im_start, pt, color=[0, 0, 255])

    step = grid_spacing
    for ix in range(2 * step, image_size[0] - 2 * step, step):
        for iy in range(2 * step, image_size[1] - 2 * step, step):
            pts_keep.append([ix, iy])
            draw_cross(im_start, [ix, iy], color=[255, 255, 0])
            draw_cross(im_end, [ix + vec_move[0], iy + vec_move[1]], color=[0, 255, 0])

    full_output_name = "/".join(path_parts[:-1]) + "/CalculatedData/pips2/input/"

    cv2.imwrite(full_output_name + "im_start.png", im_start)
    cv2.imwrite(full_output_name + "im_end.png", im_end)

    with open("full_output_name" + "info.json", "w") as f:
        json.dumps(info, f)

    return images, pts_keep


def add_2d_tracks(annot_full_path_name, kf, pts_name):
    """ Add 2d points to the keyframes
    @param annot_name - name
    @param pts_name - pts as a list"""
    import json
    import cv2
    from draw_routines.image_draw_geom_utils import draw_cross, draw_line

    path_parts = annot_full_path_name.split("/")

    with open(annot_full_path_name, "r") as f:
        my_dict = json.load(f)
        va = VideoAnnotationData.read_json(my_dict)

    im_name = va.get_image_name((kf, 0, 0), b_add_tag=True)
    if kf == va.n_keyframes() - 1:
        im_name_next = im_name
        im_name = va.get_image_name((kf-1, 0, 0), b_add_tag=True)
    else:
        im_name_next = va.get_image_name((kf+1, 0, 0), b_add_tag=True)

    im_start = cv2.imread(im_name)
    im_end = cv2.imread(im_name_next)
    b_in_section = False
    image_size = (960, 540)  # input resolution, H, W
    shift_image = [(im_start.shape[1] - image_size[0]) // 2, (im_start.shape[0] - image_size[1]) // 2]
    #scl_image = [image_size[0] / im_start.shape[1], image_size[1] / im_start.shape[0]]
    scl_image = [1, 1]

    with open(pts_name, "r") as f:
        pts_2d = json.load(f)

    kf_data = va.keyframes[kf]
    kf_data.pts_2d = []
    vx = 0.0
    vy = 0.0
    for indx in range(0, len(pts_2d[0])):
        pt_start = pts_2d[0][indx]
        pt_end = pts_2d[-1][indx]

        pt_start[0] += shift_image[1]
        pt_start[1] += shift_image[0]
        pt_end[0] += shift_image[1]
        pt_end[1] += shift_image[0]
        pt_start = [pt_start[0] / scl_image[0], pt_start[1] / scl_image[1]]
        pt_end = [pt_end[0] / scl_image[0], pt_end[1] / scl_image[1]]
        vx += pt_end[0] - pt_start[0]
        vy += pt_end[1] - pt_start[1]
        kf_data.pts_2d.append(pt_start)

        draw_cross(im_start, pt_start, color=[0, 255, 0])
        draw_cross(im_end, pt_end, color=[0, 255, 0])
        draw_line(im_start, pt_start, pt_end, color=[0, 255, 255])
        draw_line(im_end, pt_start, pt_end, color=[0, 255, 255])
    kf_data.pan_vec = [vx / len(pts_2d[0]), vy / len(pts_2d[0])]

    full_output_name = "/".join(path_parts[:-1]) + "/CalculatedData/pips2/output/"

    cv2.imwrite(full_output_name + "im_start.png", im_start)
    cv2.imwrite(full_output_name + "im_end.png", im_end)

    full_output_va_name = "/".join(path_parts[:-1]) + "/CalculatedData/pips2/output/output_va.json"

    with open(full_output_va_name, "w") as f:
        json.dump(va.write_json(), f, indent=2)
    return va


if __name__ == '__main__':
    import cv2
    import json
    from os.path import exists
    from os import mkdir

    # EG, /users/cindygrimm - the top of your folder
    path_start = FileNamesSubDirs.get_path()
    # Where you want all of your stuff to go. Please use this or set up an "if not exists" like
    #   the box path below
    dest_path = path_start + "PyCharmProjects/data/"

    # This is where Box lives in your home directory
    box_path = "Library/CloudStorage/Box-Box/"
    # Since my laptop and desktop have two different Box locations, I use this to switch between them
    if not exists(path_start + box_path):
        box_path = "MyBox/"

    # Put the two together
    src_box_path = path_start + box_path

    tree_name = "CindyEnvyPhone"

    b_draw_debug = False
    b_rename_files = False
    b_make_edge_images = False
    b_copy_tree = False
    b_copy_blueberry = False
    b_make_pips = False
    b_add_tracks = False
    b_redo_fit = True
    b_propagate_matrix = False

    if b_draw_debug:
        tree_name = "bush_8_west"
        va_fname = dest_path + tree_name + "/video_annot.json"
        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va_back = VideoAnnotationData.read_json(my_dict)
            va_back.crvs_in_depth_image()

    if b_rename_files:
        import glob
        from shutil import copyfile

        tree_name = "bush_1_east"
        images = glob.glob(dest_path + tree_name + '/rgb_*depth*.*')
        for im in images:
            print(f"Im {im}")
            im_name = im.split("/")[-1]
            #im_change = im[0:-len(im_name)] + "rgb_" + im_name[6:11] + "_depth" + im_name[-4:]
            im_change = im[0:-len(im_name)] + im_name[0:3] + im_name[4:]
            print(f"  {im_name} {im_change}")
            copyfile(im, im_change)

    if b_copy_tree:
        dest_path = path_start + "data/EnvyTree/"
        src_tree_pruning_path = src_box_path + "Robotic pruning and thinning/Datasets/2023/Jan 2023 Azure and ZED Videos/OSU Envy Orchard/"
        src_path = src_tree_pruning_path + "BeforePruning/row1East/EAST/tree2/"
        tree_name = "BP_R1_East_tree2"
        fn = VideoAnnotationData.read_envy(src_path=src_path, dest_path=dest_path, tree_name=tree_name,
                                           b_get_box_files=False)

        va = VideoAnnotationData(dest_path + tree_name + "/", img_type="png")

        va.mask_names = ["trunk", "left_support", "right_support", "tertiary"]
        va.add_directory(name_filter="rgb", start_index=0, end_index=115, skip_index=10)
        va.image_name = ""

        va_fname = dest_path + tree_name + "/video_annot.json"
        with open(va_fname, "w") as f:
            my_dict = va.write_json()
            json.dump(my_dict, f, indent=2)

        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va_back = VideoAnnotationData.read_json(my_dict)

    if b_copy_blueberry:
        # Slides for which blueberries matter https://docs.google.com/presentation/d/1334kmM_dOyAWyDPob_ZQNf7sxLo88_kX1r4042m80sQ/edit?slide=id.g33458aa9be0_0_0#slide=id.g33458aa9be0_0_0

        src_path = src_box_path + "Robotic pruning and thinning/Datasets/2024/Blueberry March/"
        bush_name = "bush_1_east"
        if not exists(dest_path + "/" + bush_name):
            mkdir(dest_path + "/" + bush_name)
        full_fn = VideoAnnotationData.read_blueberry(src_path=src_path, dest_path=dest_path, bush_name=bush_name, b_get_box_files=True)

        va = VideoAnnotationData(dest_path + bush_name + "/", img_type="jpg")

        va.mask_names = full_fn.mask_names
        va.add_directory(name_filter="rgb", start_index=147, end_index=614, skip_index=32)
        va.image_name = ""

        va_fname = dest_path + bush_name + "/video_annot.json"
        with open(va_fname, "w") as f:
            my_dict = va.write_json()
            json.dump(my_dict, f, indent=2)

        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va_back = VideoAnnotationData.read_json(my_dict)

    if b_make_edge_images:
        tree_name = "bush_1_east"
        va_name = dest_path + tree_name + "/video_annot.json"
        with open(va_name, "r") as f:
            my_dict = json.load(f)
            va = VideoAnnotationData.read_json(my_dict)
            va.create_edge_images()

    if b_make_pips:
        annot_name = 'first_tree_annot'
        va_fname = dest_path + tree_name + "/" + annot_name + ".json"
        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va = VideoAnnotationData.read_json(my_dict)
        dir_pips = dest_path + tree_name + "/CalculatedData/pips2"
        if not exists(dir_pips):
            mkdir(dir_pips)
            if not exists(dir_pips + "/input"):
                mkdir(dir_pips + "/input")
            if not exists(dir_pips + "/output"):
                mkdir(dir_pips + "/output")

        images, pts = produce_pips2_data(annot_full_path_name=va, kf=0)
        for indx, im in enumerate(images):
            fname = dest_path + tree_name + "/CalculatedData/pips2/input/im" + f"{indx}" + ".png"
            cv2.imwrite(fname, im)

        fname = dest_path + tree_name + "/CalculatedData/pips2/input/pts" + "" + ".json"
        with open(fname, "w") as f:
            json.dump(pts, f, indent=2)

    if b_redo_fit:
        read_and_rerun("bush_1_east", "video_annot")

    if b_add_tracks:
        fname_pts = dest_path + tree_name + "/CalculatedData/pips2/output/pts_2d.json"
        add_2d_tracks(va_fname, 0, fname_pts)

    if b_propagate_matrix:
        tree_name = "bush_1_east"
        va_name = dest_path + tree_name + "/video_annot.json"
        kf_copy = 0
        with open(va_name, "r") as f:
            my_dict = json.load(f)
            va = VideoAnnotationData.read_json(my_dict)
            mat_rgb_to_depth = va.keyframes[0].rgb_to_depth_matrix
            for kf in va.keyframes:
                kf.rgb_to_depth_matrix = mat_rgb_to_depth

        va_fname = dest_path + tree_name + "/video_annot_mats.json"
        with open(va_fname, "w") as f:
            my_dict = va.write_json()
            json.dump(my_dict, f, indent=2)

    import json

