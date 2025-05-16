#!/usr/bin/env python3
import pathlib

# Put images together into videos and pull out frames from a video
#   Keeps images full size

import numpy as np
import os
import cv2
import json
from utils.file_names import FileNames
from utils.file_names_sub_dirs import FileNamesSubDirs
from utils.video_annotation_data import VideoAnnotationData


def video_to_frames(fname_video, path_to_frames, name_data, start_frame=0, n_frames=-1, skip_frames=8):
    """
    Take a video and create an filenames data structure for all of the frames and a "skip" subset
    @param fname_video: name of the video file
    @param path_to_frames: path to the directory to output the frames in
    @param name_data: name of the data set
    @param start_frame: index of the first frame in the video
    @param n_frames: number of frames to keep
    @param skip_frames: number of frames to skip
    """
    if not os.path.exists(fname_video):
        print(f"Filename {fname_video} does not exist")
        return None

    name_output = path_to_frames + '/' + name_data
    if not os.path.exists(name_output):
        os.makedirs(name_output)

    vidcap = cv2.VideoCapture(fname_video)
    if n_frames == -1:
        n_cap = 10000
    else:
        n_cap = n_frames

    frame_index = 0
    path_to_frame_data = path_to_frames + '/' + name_data + '/'
    while vidcap.isOpened() and n_cap > 0:
        ret, frame = vidcap.read()
        if ret == False:
            break
        if start_frame > 0:
            start_frame -= 1
        else:
            #frame_small = frame[height_start:height_end, width_start:width_end, :]
            #frame.resize((image_size[0], image_size[1], 3))
            im_name = f"{path_to_frame_data}/rgb{frame_index:05d}.png"
            cv2.imwrite(im_name, frame)
            frame_index += 1
            n_cap -= 1
    vidcap.release()
    all_frames = FileNamesSubDirs(path_to_frame_data)
    all_frames.mask_names = ["trunk", "left_support", "right_support", "tertiary"]
    all_frames.add_directory()
    fname_write = path_to_frames + name_data + '_all_fnames.json'
    with open(fname_write, "w") as f:
        json.dump(all_frames.write_json(), f, indent=2)

    va = VideoAnnotationData(path_to_frame_data)
    va.mask_names = ["trunk", "left_support", "right_support", "tertiary"]
    va.add_directory(name_filter="rgb", start_index=0, skip_index=skip_frames)
    va.image_name = ""
    fname_write = path_to_frames + name_data + '_video_annot.json'
    with open(fname_write, "w") as f:
        json.dump(va.write_json(), f, indent=2)

    return all_frames, va


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

def make_blueberry_dataset(path_src, path_dest, img_type="jpg", n_each_folder=2, pair_spacing=0):
    Assumes there's a folder with multiple folders with multiple images in each folder
    Grab n_each_folder evenly spaced (time wise) for each director, either single or pairs
    @param path_src - directory that has the directories
    @param path_dest - directory to put the images in (will put images from each in a sub folder)
    @param img_type - one of jpg, png, etc
    @param n_each_folder - how many frames to grab from each sub folder
    @param pair_spacing - grab pairs of images, spaced n apart in time (if zero, only grabs one image)

    from os import system

    if not exists(path_src):
        print(f"No directory {path_src} exists, bailing")
        return

    if not exists(path_dest):
        mkdir(path_dest)

    search_path = f"{path_src}/*"
    fnames = glob(search_path)
    if fnames is None:
        raise ValueError(f"No sub directories in directory {search_path}")

    # Look in path_src for the sub directories
    fnames.sort()
    # Handles the case where the files are in the given directory
    fnames.append(".")
    for n in fnames:
        if not isdir(n):
            continue

        # List of images
        search_dir_path = f"{path_src}/{n}/*.{img_type}"
        fnames_images = glob(search_dir_path)
        path_dest_subdir = f"{path_dest}/"
        b_use_sub_folder_names = False
        if fnames_images is None:
            search_dir_path = f"{path_src}/{n}/color/*.{img_type}"
            fnames_images = glob(search_dir_path)
            path_dest_subdir = f"{path_dest}/{n}/"
            b_use_sub_folder_names = True

        if fnames_images is None:
            print("Subdir {n} has no images of type {img_type}")
            continue

        if not exists(path_dest_subdir):
            if b_use_sub_folder_names:
                mkdir(path_dest_subdir)

        # Copy the images over first
        fnames_images.sort()
        im_step = len(fnames_images) // n_each_folder
        im_keep = np.linspace(im_step // 2, len(fnames_images), n_each_folder)
        for im in im_keep:
            im_i = int(im)
            im_prev = max(0, im_i - pair_spacing // 2)
            im_next = min(im_prev + pair_spacing, len(fnames_images) - 1)

            name_pieces_prev = fnames_images[im_prev].split("_")
            name_prev = ""
            for s in name_pieces_prev:
                name_pieces_prev = fnames_images[im_prev]
            sys_cmd_str = f"cp {search_dir_path}/{fnames_images[im_prev]} {path_dest_subdir}{fnames_images[im_prev]}"
            system(sys_cmd_str)

            if im_next != im_prev:
                sys_cmd_str = f"cp {search_dir_path}/{fnames_images[im_next]} {path_dest_subdir}{fnames_images[im_next]}"
                system(sys_cmd_str)

    all_files = HandleFileNames(path_dest, img_type=img_type)
    """


if __name__ == '__main__':
    home_dir = "/Users/cindygrimm/"
    video_path = "PycharmProjects/cindygr_pips2/stock_videos/"
    video_file_name = "EnvyTree.mp4"
    video_name = home_dir + video_path + video_file_name
    data_dir = home_dir + "PycharmProjects/data/"
    data_name = "CindyEnvyPhone"
    video_to_frames(fname_video=video_name, path_to_frames=data_dir, name_data=data_name, start_frame=0, n_frames=-1, skip_frames=2)
