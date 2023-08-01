#!/usr/bin/env python3

# Split masks
#  1) if there's one mask image with multiple masks, split it into mask image 0, 1, etc

import numpy as np
from glob import glob
import cv2
import json
from os.path import exists
from bezier_cyl_2d import BezierCyl2D
from line_seg_2d import LineSeg2D
from scipy.cluster.vq import kmeans, whiten, vq
from BaseStatsImage import BaseStatsImage


def split_mask(in_im_mask, b_one_mask=True):
    """Split the mask image up into connected components, discarding anything really small
    @param in_im_mask - the mask image
    @param b_one_mask - output only one mask y/n
    @param b_debug - print out mask labeled image
    @return a list of boolean indices for each component, original labels"""
    output = cv2.connectedComponentsWithStats(in_im_mask)
    labels = output[1]
    stats = output[2]

    ret_masks = []
    i_widest = 0
    i_area = 0
    for i, stat in enumerate(stats):
        if np.sum(in_im_mask[labels == i]) == 0:
            continue

        if stat[cv2.CC_STAT_WIDTH] < 5:
            continue
        if stat[cv2.CC_STAT_HEIGHT] < 0.5 * in_im_mask.shape[1]:
            continue
        if i_area < stat[cv2.CC_STAT_AREA]:
            i_widest = len(ret_masks)
            i_area = stat[cv2.CC_STAT_AREA]
        ret_masks.append(labels == i)

    try:
        if b_one_mask:
            return [ret_masks[i_widest]]
    except:
        pass

    labels = 128 + labels * (120 // output[0])

    return ret_masks, labels


def split_masks(path, im_name, b_one_mask=True, b_output_debug=True):
    """ Create mask images
    @param path: where the image is located
    @param im_name: name of the image
    @return: None
    """
    path_debug = path + "DebugImages/"
    path_image = path + im_name + "_mask.png"
    im_mask_color = cv2.imread(path_image)
    im_mask_gray = cv2.cvtColor(im_mask_color, cv2.COLOR_BGR2GRAY)

    ret_masks, ret_labels = split_mask(im_mask_gray, b_one_mask)

    if b_output_debug:
        cv2.imwrite(path_debug + im_name + "_" + "labels.png", ret_labels)

    for i, m in enumerate(ret_masks):
        im_mask_name = path + im_name + "_mask" + str(i) + ".png"
        cv2.imwrite(im_mask_name, m)


if __name__ == '__main__':
    path = "./data/predictions/"
    #path = "./forcindy/"
    for im_i in range(0, 49):
        name = str(im_i)
        print(name)
        split_masks(path, name, b_one_mask=True, b_output_debug=True)
