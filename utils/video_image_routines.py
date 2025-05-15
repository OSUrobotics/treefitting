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


if __name__ == '__main__':
    home_dir = "/Users/cindygrimm/"
    video_path = "PycharmProjects/cindygr_pips2/stock_videos/"
    video_file_name = "EnvyTree.mp4"
    video_name = home_dir + video_path + video_file_name
    data_dir = home_dir + "PycharmProjects/data/"
    data_name = "CindyEnvyPhone"
    video_to_frames(fname_video=video_name, path_to_frames=data_dir, name_data=data_name, start_frame=0, n_frames=-1, skip_frames=2)
