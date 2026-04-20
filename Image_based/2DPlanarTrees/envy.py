import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import cv2
import imageio
from utils.file_names_sub_dirs import FileNamesSubDirs
from flux_medial import find_points_ma, find_long_curves, label_chains, check_hough
import pickle
from trees import compute_aof_tree_medial_axis, show_labels, visualizegradient, sample_circle_points

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default="make_flow", type=str, help="One of: make_flow, find_ma")
    parser.add_argument('--dest_path', default="PycharmProjects/data/tree_14/ma/", type=str, help="where tree/bush data is stored")
    parser.add_argument('--dir_src_image', default="PycharmProjects/data/tree_14/horizontal_00", type=str, help="Tree or bush name")
    parser.add_argument('--dir_src', default="PycharmProjects/data/tree_14_of/horizontal_00_01/", type=str, help="Tree or bush name")

    args = parser.parse_args()

    home_dir = os.getcwd().split("/")
    home_dir = "/" + home_dir[0] + "/" + home_dir[1] + "/" + home_dir[2] + "/"
    source_dir = home_dir + args.dir_src
    dest_dir = home_dir + args.dest_path

    str_image = args.dir_src_image.split("/")
    fname = str_image[-2]
    source_image = home_dir + args.dir_src_image + ".png"
    source_depth = home_dir + args.dir_src_image + ".npy"
    source_flow_quad = source_dir + "npy/" + "flow_00000.npy"
    source_flow_scale = source_dir + "npy/" + "flow_scale_00000.npy"

    if args.action == "make_flow":
        compute_aof_tree_medial_axis(filename=source_flow_quad, fname_out=dest_dir + fname + "_quad_ma.png",
                                     boundarythreshold=2, medialthreshold=-0.2)
        compute_aof_tree_medial_axis(filename=source_flow_scale, fname_out=dest_dir + fname + "_scale_ma.png",
                                     boundarythreshold=2, medialthreshold=-0.2)
    elif args.action == "find_ma":
        for im_end in ["quad", "scale"]:
            im_ma = imageio.imread(dest_dir + fname + im_end + "_ma.png")
            if im_end == "quad":
                im_flow = imageio.imread(source_flow_quad)
            else:
                im_flow = imageio.imread(source_flow_scale)
            im_dist = imageio.imread(dest_dir + fname + im_end + "_dist.png")

            b_new_data = True
            save_pickle_pts = dest_dir + fname + im_end + "_pts.pickle"
            save_label_pts = dest_dir + fname + im_end + "_pts.png"
            save_pickle_chains = dest_dir + fname + im_end + "_chains.pickle"
            if b_new_data:
                print("Making new edge data")
                pts, edges, junctions, labels = find_points_ma(im_ma, im_flow, im_dist)
                imageio.imwrite("labels.png", labels.astype(np.uint8))
                imageio.imwrite(save_label_pts, labels.astype(np.uint8))
                with (open(save_pickle_pts, "wb") as f):
                    dict = {"pts": pts, "edges": edges, "junctions": junctions}
                    pickle.dump(dict, f)
                    show_labels(points=pts, edges=edges, junctions=junctions)
            else:
                print("Reading edge data")
                with (open(save_pickle_pts, "rb") as f):
                    dict = pickle.load(f)
                pts = dict["pts"]
                edges = dict["edges"]
                junctions = dict["junctions"]
                labels = np.zeros((im_ma.shape[0], im_ma.shape[1], 3))

                b_new_chain_data = True
                if b_new_chain_data:
                    chains = find_long_curves(pts, edges, junctions, labels)
                    with (open(save_pickle_chains, "wb") as f):
                        pickle.dump(chains, f)
                else:
                    with (open(save_pickle_chains, "rb") as f):
                        chains = pickle.load(f)

                chain_labels = label_chains(pts, chains, im_dist=im_dist, im_flow=im_flow)

    print("Completed Run")
