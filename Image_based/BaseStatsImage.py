#!/usr/bin/env python3

# Eigen value stats for a mask image
import os

import numpy as np
from glob import glob
import cv2
import json
from os.path import exists

class BaseStatsImage:
    _width = 0
    _height = 0

    _x_grid = None
    _y_grid = None

    @staticmethod
    def _init_grid_(in_im):
        """ INitialize width, height, xgrid, etc so we don't have to keep re-making it
        :param in_im: Input image
        """
        if BaseStatsImage._width == in_im.shape[1] and BaseStatsImage._height == in_im.shape[0]:
            return
        BaseStatsImage._width = in_im.shape[1]
        BaseStatsImage._height = in_im.shape[0]

        BaseStatsImage._x_grid, BaseStatsImage._y_grid = np.meshgrid(np.linspace(0.5, BaseStatsImage._width - 0.5, BaseStatsImage._width), np.linspace(0.5,  BaseStatsImage._height -  0.5,  BaseStatsImage._height))

    def __init__(self, path, image_name, mask_id=-1, b_output_debug=True, b_recalc=False):
        """ Read in the image, mask image, flow image, 2 rgb images
        @param path: Directory where files are located
        @param image_name: image number/name as a string
        @param image_mask: the actual mask image
        @param b_recalc: Force recalculate the result, y/n"""

        self.path = path
        self.path_debug = path + "DebugImages/"
        self.path_calculated = path + "CalculatedData/"

        self.b_output_debug = b_output_debug
        self.b_recalc = b_recalc

        if not exists(self.path_debug):
            os.mkdir(self.path_debug)
        if not exists(self.path_calculated):
            os.mkdir(self.path_calculated)

        self.name = image_name
        self.stats_dict = None
        self.mask_id = str(mask_id)
        if mask_id == -1:
            self.mask_id = ""
        self.mask_image_name = self.path + image_name + "_mask" + self.mask_id + ".png"
        self.image_mask = cv2.imread(self.mask_image_name)
        if len(self.image_mask.shape) == 3:
            im_mask = cv2.cvtColor(self.image_mask, cv2.COLOR_BGR2GRAY)
        self._init_grid_(self.image_mask)

        # Calculate the stats for this image
        print("Calculating stats")
        fname_stats = self.path_calculated + self.name + f"_mask{mask_id}.json"
        if b_recalc or not exists(fname_stats):
            self.stats_dict = self.stats_image(self.image_mask)
            for k, v in self.stats_dict.items():
                try:
                    # Convert any numpy arrays to lists
                    if v.size == 2:
                        self.stats_dict[k] = [v[0], v[1]]
                except:
                    pass
            # If this fails, make a CalculatedData and DebugImages folder in the data/forcindy folder
            with open(fname_stats, 'w') as f:
                json.dump(self.stats_dict, f)
        elif exists(fname_stats):
            with open(fname_stats, 'r') as f:
                self.stats_dict = json.load(f)

        for k, v in self.stats_dict.items():
            try:
                if len(v) == 2:
                    self.stats_dict[k] = np.array([v[0], v[1]])
            except:
                pass

    def stats_image(self, in_im):
        """ Add statistics (bounding box, left right, orientation, radius] to image
        Note: Could probably do this without transposing image, but...
        @param im image
        @returns stats as a dictionary of values"""

        pixs_in_mask = in_im > 0

        xs = BaseStatsImage._x_grid[pixs_in_mask]
        ys = BaseStatsImage._y_grid[pixs_in_mask]

        stats = {}
        stats["x_min"] = np.min(xs)
        stats["y_min"] = np.min(ys)
        stats["x_max"] = np.max(xs)
        stats["y_max"] = np.max(ys)
        stats["x_span"] = stats["x_max"] - stats["x_min"]
        stats["y_span"] = stats["y_max"] - stats["y_min"]

        avg_width = 0.0
        count_width = 0
        if stats["x_span"] > stats["y_span"]:
            stats["Direction"] = "left_right"
            stats["Length"] = stats["x_span"]
            for r in range(0, BaseStatsImage._width):
                if sum(pixs_in_mask[:, r]) > 0:
                    avg_width += sum(pixs_in_mask[:, r] > 0)
                    count_width += 1
        else:
            stats["Direction"] = "up_down"
            stats["Length"] = stats["y_span"]
            for c in range(0, BaseStatsImage._height):
                if sum(pixs_in_mask[c, :]) > 0:
                    avg_width += sum(pixs_in_mask[c, :] > 0)
                    count_width += 1
        stats["width"] = avg_width / count_width
        stats["center"] = np.array([np.mean(xs), np.mean(ys)])

        x_matrix = np.zeros([2, xs.shape[0]])
        x_matrix[0, :] = xs.transpose() - stats["center"][0]
        x_matrix[1, :] = ys.transpose() - stats["center"][1]
        covariance_matrix = np.cov(x_matrix)
        eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
        if eigen_values[0] < eigen_values[1]:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[1, :]
        else:
            stats["EigenValues"] = [np.min(eigen_values), np.max(eigen_values)]
            stats["EigenVector"] = eigen_vectors[0, :]
        eigen_ratio = stats["EigenValues"][1] / stats["EigenValues"][0]
        stats["EigenVector"][1] *= -1
        stats["EigenRatio"] = eigen_ratio
        stats["lower_left"] = stats["center"] - stats["EigenVector"] * (stats["Length"] * 0.5)
        stats["upper_right"] = stats["center"] + stats["EigenVector"] * (stats["Length"] * 0.5)
        print(stats)
        print(f"Eigen ratio {eigen_ratio}")
        return stats


if __name__ == '__main__':
    path = "./data/trunk_segmentations/"
    #path = "./forcindy/"
    for in_r in range(0, 16):
        path_row = path + "row_" + str(in_r) + "/"
        search_path = f"{path_row}*_mask*.png"
        fnames = glob(search_path)
        if fnames is None:
            raise ValueError(f"No files in directory {search_path}")

        for fname in fnames:
            full_img_name = str.split(fname, "/")[-1]
            img_name = str.split(full_img_name, "_")[0]
            mask_id = str.split(full_img_name, ".")[0]
            mask_id = int(mask_id[-1])

            print(f"Image name {img_name}, mask id {mask_id}")
            im_mask = cv2.imread(fname)
            if len(im_mask.shape) == 3:
                im_mask = cv2.cvtColor(im_mask, cv2.COLOR_BGR2GRAY)

            bp = BaseStatsImage(path, img_name, im_mask, mask_id, b_output_debug=True, b_recalc=True)
