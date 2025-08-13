#!/usr/bin/env python3

# Video annotation processing
#
# Various routines for processing video annotations

import numpy as np
from utils.file_names import FileNames
from utils.file_names_sub_dirs import FileNamesSubDirs
from utils.keyframe_data import KeyFrameData
from utils.video_annotation_data import VideoAnnotationData
from fit_routines.bspline_fit_params import BSplineFitParams
from utils.camera_projections import CameraProjections


def read_envy(src_path, dest_path, tree_name, b_get_box_files=True):
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

def read_blueberry(src_path, dest_path, bush_name, b_get_box_files=True):
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


def read_and_rerun(dir_name, annot_name, cam_rgb: CameraProjections, cam_depth: CameraProjections):
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
    mat_rgb_to_depth = cam_depth.world_to_image @ np.linalg.inv(cam_rgb.world_to_image)
    va.crvs_in_depth_image(mat_rgb_to_depth=mat_rgb_to_depth)


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

    info = {"KeyFrame": kf,
            "BackboneSpacing": backbone_spacing,
            "RadialSpacing": radial_spacing,
            "GridSpacing": grid_spacing}

    im_name = va.keyframes[kf].image_name
    if kf == va.n_keyframes() - 1:
        im_name_next = im_name
        im_name = va.keyframes[kf - 1].image_name
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
            # im_resize = cv2.resize(im, image_size)
            shift_image = [(im.shape[1] - image_size[0]) // 2, (im.shape[0] - image_size[1]) // 2]
            im_resize = im[shift_image[0]:shift_image[0] + image_size[1], shift_image[1]:shift_image[1] + image_size[0],
                        :]
            # scl_image[0] = image_size[0] / im.shape[1]
            # scl_image[1] = image_size[1] / im.shape[0]
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
                # pts_2d.append([pt[0] * scl_image[0], pt[1] * scl_image[1]])
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
        im_name = va.get_image_name((kf - 1, 0, 0), b_add_tag=True)
    else:
        im_name_next = va.get_image_name((kf + 1, 0, 0), b_add_tag=True)

    im_start = cv2.imread(im_name)
    im_end = cv2.imread(im_name_next)
    b_in_section = False
    image_size = (960, 540)  # input resolution, H, W
    shift_image = [(im_start.shape[1] - image_size[0]) // 2, (im_start.shape[0] - image_size[1]) // 2]
    # scl_image = [image_size[0] / im_start.shape[1], image_size[1] / im_start.shape[0]]
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
    import argparse

    b_draw_debug = False
    b_rename_files = False
    b_make_edge_images = False
    b_copy_tree = False
    b_copy_blueberry = False
    b_make_pips = False
    b_add_tracks = False
    b_redo_fit = True
    b_propagate_matrix = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default="redo_fit", type=str,
                        help="One of: draw_debug, rename_files, make_edge_images, copy_tree, copy_blueberry, make_pips, add_tracks, redo_fit, copy_matrix")
    parser.add_argument('--dest_path', default="PycharmProjects/data/", type=str, help="where tree/bush data is stored")
    parser.add_argument('--box_path', default="Library/CloudStorage/Box-Box/", type=str,
                        help="where your box folder lives")
    parser.add_argument('--bush_tree_name', default="bush_3_east", type=str, help="Tree or bush name")
    parser.add_argument('--annot', default="video_annot_final.json", type=str, help="which video annotation to use")
    parser.add_argument('--key_frame', default=-1, type=int, help="which key frame, set to -1 for all")
    parser.add_argument('--mask', default=-1, type=int, help="which mask, set to -1 for all")
    parser.add_argument('--mask_id', default=-1, type=int, help="which curve, set to -1 for all")
    parser.add_argument('--camera', default="azure", type=str, help="Camera, one of azure, intel TODO")
    parser.add_argument('--start_index', default=149, type=int, help="Start index for copy tree/bush")
    parser.add_argument('--end_index', default=-1, type=int, help="End index for copy tree/bush, -1 is all")
    parser.add_argument('--skip_index', default=30, type=int,
                        help="skip frames for copy tree/bush, 10 for tree, 32 for blueberry")

    args = parser.parse_args()

    # Grab the current path
    path_start = FileNamesSubDirs.get_path()
    # Where the video_annot.json lives
    path_full = path_start + args.dest_path + args.bush_tree_name + "/"
    # The video_annot.json file
    va_fname = path_full + args.annot

    # This is where Box lives in your home directory
    box_path = "Library/CloudStorage/Box-Box/"
    # Since my laptop and desktop have two different Box locations, I use this to switch between them
    if not exists(path_start + box_path):
        box_path = "MyBox/"

    # Put the two together
    src_box_path = path_start + box_path

    # Read in a camera, if there is one
    try:
        if args.camera == "azure":
            cam_rgb = CameraProjections(camera_fname=("azure_camera.json", "rgb_half_size"),
                                        camera_calibration_fname=("azure_camera_calibration.json", "color"),
                                        camera_rgb_to_depth_name="azure_rgb_to_depth.json",
                                        params={})
            cam_depth = CameraProjections(camera_fname=("azure_camera.json", "depth_narrow_unbinned"),
                                          camera_calibration_fname=("azure_camera_calibration.json", "depth"),
                                          params={})
        elif args.camera == "intel":
            cam_rgb = CameraProjections(camera_fname=("intel_camera.json", "FIX"),
                                        camera_calibration_fname=("intel_camera_calibration.json", "color"),
                                        camera_rgb_to_depth_name="intel_rgb_to_depth.json",
                                        params={})
            cam_depth = CameraProjections(camera_fname=("intel_camera.json", "FIX"),
                                          camera_calibration_fname=("intel_camera_calibration.json", "depth"),
                                          params={})
    except:
        cam_rgb = None
        cam_depth = None

    if args.action == "draw_debug":
        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va_back = VideoAnnotationData.read_json(my_dict)
            va_back.crvs_in_depth_image()

    if args.action == "rename_files":
        import glob
        from shutil import copyfile

        # This one usually needs a code change
        # Which files to change the name for
        images = glob.glob(path_full + '/rgb_*depth*.*')
        for im in images:
            print(f"Im {im}")
            im_name = im.split("/")[-1]
            # im_change = im[0:-len(im_name)] + "rgb_" + im_name[6:11] + "_depth" + im_name[-4:]
            im_change = im[0:-len(im_name)] + im_name[0:3] + im_name[4:]
            print(f"  {im_name} {im_change}")
            copyfile(im, im_change)

    if args.action == "copy_tree":
        dest_path = path_start + "data/EnvyTree/"
        src_tree_pruning_path = src_box_path + "Robotic pruning and thinning/Datasets/2023/Jan 2023 Azure and ZED Videos/OSU Envy Orchard/"
        src_path = src_tree_pruning_path + "BeforePruning/row1East/EAST/tree2/"
        tree_name = args.bush_tree_name
        fn = VideoAnnotationData.read_envy(src_path=src_path, dest_path=dest_path, tree_name=tree_name,
                                           b_get_box_files=False)

        va = VideoAnnotationData(dest_path + tree_name + "/", img_type="png")

        va.mask_names = ["trunk", "left_support", "right_support", "tertiary"]
        va.add_directory(name_filter="rgb", start_index=args.skip_index, end_index=args.end_index,
                         skip_index=args.skip_index)
        va.image_name = ""

        va_fname = dest_path + tree_name + "/video_annot.json"
        with open(va_fname, "w") as f:
            my_dict = va.write_json()
            json.dump(my_dict, f, indent=2)

        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va_back = VideoAnnotationData.read_json(my_dict)

    if args.action == "copy_blueberry":
        # Slides for which blueberries matter https://docs.google.com/presentation/d/1334kmM_dOyAWyDPob_ZQNf7sxLo88_kX1r4042m80sQ/edit?slide=id.g33458aa9be0_0_0#slide=id.g33458aa9be0_0_0

        src_path = src_box_path + "Robotic pruning and thinning/Datasets/2024/Blueberry March/"
        bush_name = args.bush_tree_name
        if not exists(path_full):
            mkdir(path_full)
        full_fn = VideoAnnotationData.read_blueberry(src_path=src_path, dest_path=path_start + args.dest_path,
                                                     bush_name=bush_name, b_get_box_files=True)

        va = VideoAnnotationData(path_full, img_type="jpg")

        va.mask_names = full_fn.mask_names
        va.add_directory(name_filter="rgb", start_index=args.skip_index, end_index=args.end_index,
                         skip_index=args.skip_index)
        va.image_name = ""

        with open(va_fname, "w") as f:
            my_dict = va.write_json()
            json.dump(my_dict, f, indent=2)

    if args.action == "make_edge_images":
        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va = VideoAnnotationData.read_json(my_dict)
            va.create_edge_images()

    if args.action == "make_pips":
        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va = VideoAnnotationData.read_json(my_dict)
        dir_pips = path_full + "/CalculatedData/pips2"
        if not exists(dir_pips):
            mkdir(dir_pips)
            if not exists(dir_pips + "/input"):
                mkdir(dir_pips + "/input")
            if not exists(dir_pips + "/output"):
                mkdir(dir_pips + "/output")

        images, pts = produce_pips2_data(annot_full_path_name=va, kf=0)
        for indx, im in enumerate(images):
            fname = path_full + "/CalculatedData/pips2/input/im" + f"{indx}" + ".png"
            cv2.imwrite(fname, im)

        fname = path_full + "/CalculatedData/pips2/input/pts" + "" + ".json"
        with open(fname, "w") as f:
            json.dump(pts, f, indent=2)

    if args.action == "redo_fit":
        read_and_rerun(dir_name=args.bush_tree_name, annot_name=args.annot[:-5], cam_rgb=cam_rgb, cam_depth=cam_depth)

    if args.action == "add_tracks":
        fname_pts = path_full + "/CalculatedData/pips2/output/pts_2d.json"
        add_2d_tracks(va_fname, 0, fname_pts)

    if args.action == "propagate_matrix":
        kf_copy = args.key_frame
        with open(va_fname, "r") as f:
            my_dict = json.load(f)
            va = VideoAnnotationData.read_json(my_dict)
            mat_rgb_to_depth = va.keyframes[0].rgb_to_depth_matrix
            for kf in va.keyframes:
                kf.rgb_to_depth_matrix = mat_rgb_to_depth

        va_fname = path_full + "/video_annot_mats.json"
        with open(va_fname, "w") as f:
            my_dict = va.write_json()
            json.dump(my_dict, f, indent=2)

    import json

