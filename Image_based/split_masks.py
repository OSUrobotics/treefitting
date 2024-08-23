#!/usr/bin/env python3

# Split masks
#  1) if there's one mask image with multiple masks, split it into mask image 0, 1, etc

import os
import sys
sys.path.insert(0, os.path.abspath('./Image_based'))

import numpy as np
import cv2
from os.path import exists


def create_optical_flow_edge_image(of_image_name, of_edge_image_name):
    """Create an edge image for the optical flow
    @param of_image_name - optical flow image name
    @param of_edge_image_name - output file name"""
    im_of_color = cv2.imread(of_image_name)
    im_of_gray = cv2.cvtColor(im_of_color, cv2.COLOR_BGR2GRAY)
    im_of_edge = cv2.Canny(im_of_gray, 50, 150, apertureSize=3)
    cv2.imwrite(of_edge_image_name, im_of_edge)


def create_optical_flow_edge_images(handle_fnames, b_use_calculated):
    """ Convert an entire set of optical flow images to edge images
    @param andle_fnames - handle file name file
    @param b_use_calculated - are optical flow images in calculated or the main directory?"""
    all_files = HandleFileNames.read_filenames(handle_fnames)

    for ind in all_files.loop_images():
        # Not putting tags on optical flow names
        of_fname = all_files.get_flow_image_name(path=all_files.path, index=ind, b_add_tag=True)
        if b_use_calculated:
            of_edge_fname = all_files.get_edge_image_name(path=all_files.path_calculated, index=ind, b_optical_flow=True, b_add_tag=b_add_tag)
        else:
            of_edge_fname = all_files.get_edge_image_name(path=all_files.path, index=ind, b_optical_flow=True, b_add_tag=True)

        if not exists(of_fname):
            raise ValueError(f"Error, file {of_fname} does not exist")
        create_optical_flow_edge_image(of_fname, of_edge_fname)


def convert_jet_to_grey(img):
    (height, width) = img.shape[:2]

    im_size = 16
    n_pixs = im_size * im_size
    im_all = np.zeros((im_size, im_size), dtype='uint8')
    for i in range(0, im_size * im_size):
        im_all[i // im_size, i % im_size] = i
    im_rgb = cv2.applyColorMap(im_all, cv2.COLORMAP_JET)
    im_rgb_linear = im_rgb.reshape((im_size * im_size, 3))

    r_indx = 2
    g_indx = 1
    b_indx = 0
    b_check = np.logical_and(np.logical_and(im_rgb_linear[:, r_indx] == 0, im_rgb_linear[:, g_indx] == 0), im_rgb_linear[:, b_indx] < 255)
    b_offset = np.count_nonzero(b_check)
    c_check = im_rgb_linear[:, b_indx] == 255
    c_offset = b_offset + np.count_nonzero(c_check)
    g_check = im_rgb_linear[:, g_indx] == 255
    g_offset = c_offset + np.count_nonzero(g_check)
    y_check = np.logical_and(im_rgb_linear[:, b_indx] == 0, im_rgb_linear[:, r_indx] == 255)
    y_offset = g_offset + np.count_nonzero(y_check)
    r_check = np.logical_and(np.logical_and(im_rgb_linear[:, b_indx] == 0, im_rgb_linear[:, g_indx] == 0), im_rgb_linear[:, r_indx] < 255)
    r_offset = y_offset + np.count_nonzero(r_check)
    b_c_check = np.count_nonzero(np.logical_and(b_check, c_check))
    b_g_check = np.count_nonzero(np.logical_and(b_check, g_check))
    b_y_check = np.count_nonzero(np.logical_and(b_check, y_check))
    b_r_check = np.count_nonzero(np.logical_and(b_check, r_check))

    c_g_check = np.count_nonzero(np.logical_and(c_check, g_check))
    g_y_check = np.count_nonzero(np.logical_and(g_check, y_check))
    y_r_check = np.count_nonzero(np.logical_and(y_check, r_check))
    # c = img[:, :, b_indx] == 255
    #r = np.logical_and(img[:, :, g_indx] == 0, img[:, :, b_indx] == 0)
    #b  = np.logical_and(np.logical_and(img_r == 0, img_g == 0), img_b < 255)
    #y = np.logical_and(img[:, :, r_indx] == 255, img[:, :, b_indx] == 0)
    #c = img[:, :, g_indx] == 255
    #b = np.logical_and(img[:, :, g_indx] == 255, img[:, :, r_indx] == 0)

    im_gray = np.zeros((height, width), dtype='uint8')
    div = 255 // 3
    for r in range(0, height):
        for c in range(1, width):
            bgr = img[r, c]
            if bgr[2] > bgr[1] and bgr[2] > bgr[0]:
                gray = 2 * div + bgr[2] // 3
                im_gray[r, c] = gray
            elif bgr[0] > bgr[1] and bgr[0] > bgr[2]:
                gray = bgr[0] // 3
                im_gray[r, c] = gray
            else:
                gray = div + bgr[1] // 3
                im_gray[r, c] = gray
                """
            bgr = img[r, c]
            if bgr[0] < 20 and bgr[1] < 20 and bgr[2] < 255:
                bgr[0] = 0
                bgr[1] = 0
            if bgr[1] < 20 and bgr[2] < 20 and bgr[0] < 255:
                bgr[1] = 0
                bgr[2] = 0
            if bgr[r_indx] == 0 and bgr[g_indx] == 0 and bgr[b_indx] < 255:
                gray = bgr[b_indx] // 8
                gray = min(gray, b_offset)
                gray = max(gray, 0)
                im_gray[r, c] = gray
            elif bgr[b_indx] == 255 and bgr[r_indx] == 0:
                gray = b_offset + bgr[g_indx] // 4
                gray = min(gray, c_offset)
                gray = max(gray, b_offset)
                im_gray[r, c] = gray
            elif bgr[g_indx] == 255:
                gray = c_offset + bgr[r_indx] // 4
                gray = min(gray, g_offset)
                gray = max(gray, c_offset)
                im_gray[r, c] = gray
            elif bgr[b_indx] == 0 and bgr[r_indx] == 255:
                gray = g_offset + (255 - bgr[g_indx]) // 4
                gray = min(gray, y_offset)
                gray = max(gray, g_offset)
                im_gray[r, c] = gray
            elif bgr[b_indx] == 0 and bgr[g_indx] == 0:
                gray = y_offset + (255 - bgr[r_indx]) // 4
                gray = min(gray, r_offset)
                gray = max(gray, y_offset)
                im_gray[r, c] = gray
            else:
                im_gray[r, c] = 0 #gray
    """
    """            
    im_gray_linear = np.zeros((im_size * im_size), dtype='uint8')
    b_offset = np.count_nonzero(b_check)
    c_offset = b_offset + 256 // 4
    #c_offset = y_offset + 256 // 4
    #b_end = c_offset + 256 // 4
    ugh = np.zeros((height, width)) + 128
    foo = img[b, b_indx]
    im_gray[b] = (img_b[b] - 128) // 4
    im_gray_linear[b_check] = (im_rgb_linear[b_check, b_indx] - 128) // 4
    im_gray[c] = b_offset + img_g[c] // 4
    im_gray_linear[b_check] = b_offset + im_rgb_linear[c_check, g_indx] // 4
    #im_gray[y] = r_offset + img[y, g_indx] // 4
    #im_gray[c] = y_offset + img[c, b_indx] // 4
    #im_gray[b] = c_offset + (255 - img[b, b_indx]) // 4
    #np.max((252 - img[b2, b_indx]) // 4)
    #np.min((252 - img[b2, b_indx]) // 4)
    #im_gray[b2] = b_end + (252 - img[b2, b_indx]) // 4


    # cm = LinearSegmentedColormap("jet", _jet_data, N=2 ** 8)
    # cm = colormaps['turbo'] swap with jet if you use turbo colormap instead

    # cm._init()  # Must be called first. cm._lut data field created here

    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    #search_params = dict(checks=50)
    #fm = cv2.FlannBasedMatcher(index_params, search_params)

    # JET, BGR order, excluding special palette values (>= 256)
    #fm.add(255 * np.float32([im_rgb[:256, (2, 1, 0)]]))  # jet
    #fm.train()

    # look up all pixels
    #query = img.reshape((-1, 3)).astype(np.float32)
    #matches = fm.match(query)

    # statistics: `result` is palette indices ("grayscale image")
    #output = np.uint16([m.trainIdx for m in matches]).reshape(height, width)
    #result = np.where(output < 256, output, 0).astype(np.uint8)
    # dist = np.uint8([m.distance for m in matches]).reshape(height, width)
    """
    return im_gray  # , dist uncomment if you wish accuracy image


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
    path = "./data/forcindy/"

    create_optical_flow_edge_images("./data/forcindy_fnames.json", b_use_calculated=True)
    fname_img_depth = path + "0_depth.png"
    im = cv2.imread(fname_img_depth)
    im_grey = convert_jet_to_grey(im)
    cv2.imwrite(path + "0_depth_back.png", im_grey)
    im_rgb = cv2.applyColorMap(im_grey, cv2.COLORMAP_JET)
    cv2.imwrite(path + "0_depth_back_rgb.png", im_rgb)

    path = "./data/predictions/"
    #path = "./forcindy/"
    for im_i in range(0, 49):
        name = str(im_i)
        print(name)
        split_masks(path, name, b_one_mask=True, b_output_debug=True)
