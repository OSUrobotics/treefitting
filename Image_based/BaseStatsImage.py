#!/usr/bin/env python3

# Given a mask image name, read the image in and perform the following statistical calculations
# PCA - find major and minor axes
#  Find the bounding box, max width along the major axis
#
# Caches data in image name.json
#


import numpy as np
import cv2
from os.path import exists
import json
from line_seg_2d import LineSeg2D
from HandleFileNames import HandleFileNames


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

        BaseStatsImage._x_grid, BaseStatsImage._y_grid = np.meshgrid(np.linspace(0.5, BaseStatsImage._width - 0.5, BaseStatsImage._width),
                                                                     np.linspace(0.5,  BaseStatsImage._height -  0.5,  BaseStatsImage._height))

    def __init__(self, fname_mask_image, fname_calculated=None, fname_debug=None, b_recalc=False):
        """ Given an image, calculate the main axis and width along that axis
          If fname_calculated is given, check to see if the data is already calculated - if so, read it in,
          otherwise, calculate and write out
          If fname_debug is given, the write out a debug image with the main axis and end points marked
        @param fname_mask_image: name of mask file, rgb or gray scale image with white where the mask is
        @param fname_calculated: the file name for the saved .json file
        @param fname_debug: the file name for a debug image showing the bounding box, etc
        @param b_recalc: Force recalculate the result, y/n"""

        self.stats_dict = None
        mask_image_rgb = cv2.imread(fname_mask_image)
        if len(mask_image_rgb.shape) == 3:
            self.mask_image = cv2.cvtColor(mask_image_rgb, cv2.COLOR_BGR2GRAY)
        else:
            self.mask_image = mask_image_rgb
        self._init_grid_(self.mask_image)

        # Calculate the stats for this image
        if b_recalc or not fname_calculated or not exists(fname_calculated):
            # Cached data doesn't exist, or we need to re-calculated
            self.stats_dict = self.stats_image(self.mask_image)

            try:
                # Convert any numpy arrays to lists prior to writing out
                for k, v in self.stats_dict.items():
                    try:
                        if v.size == 2:
                            self.stats_dict[k] = [v[0], v[1]]
                    except AttributeError:
                        pass
                # If this fails, make a CalculatedData and DebugImages folder in the data/forcindy folder
                with open(fname_calculated, 'w') as f:
                    json.dump(self.stats_dict, f, indent=2)
            except FileNotFoundError:
                if fname_calculated:
                    print(f"BaseStats Image: File not found {fname_calculated}")
        else:
            with open(fname_calculated, 'r') as f:
                self.stats_dict = json.load(f)

        # Undo the numpy array -> list
        for k, v in self.stats_dict.items():
            try:
                if len(v) == 2:
                    self.stats_dict[k] = np.array([v[0], v[1]])
            except TypeError:
                pass

        if fname_debug:
            im_debug = mask_image_rgb
            if len(mask_image_rgb.shape) == 1:
                im_debug = cv2.cvtColor(mask_image_rgb, cv2.COLOR_GRAY2RGB)
            # Add the lines showing the eigen vecs
            self.debug_image(im_debug)
            cv2.imwrite(fname_debug, im_debug)

    @staticmethod
    def stats_image(in_im):
        """ Add statistics (bounding box, left right, orientation, radius] to image
        Note: Could probably do this without transposing image, but...
        Doing this as a static method so it can be used in a stand-alone pipeline
        @param in_im gray scale image with mask non-zero
        @returns stats as a dictionary of values"""

        BaseStatsImage._init_grid_(in_im)

        pixs_in_mask = in_im > 0

        xs = BaseStatsImage._x_grid[pixs_in_mask]
        ys = BaseStatsImage._y_grid[pixs_in_mask]

        stats = {"x_min": np.min(xs), "y_min": np.min(ys), "x_max": np.max(xs), "y_max": np.max(ys)}
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
        stats["EigenMinorVector"] = np.array([stats["EigenVector"][1], -stats["EigenVector"][0]])
        stats["EigenRatio"] = eigen_ratio
        stats["lower_left"] = stats["center"] - stats["EigenVector"] * (stats["Length"] * 0.5)
        stats["upper_right"] = stats["center"] + stats["EigenVector"] * (stats["Length"] * 0.5)
        stats["left_edge"] = stats["center"] - stats["EigenMinorVector"] * (stats["width"] * 0.5)
        stats["right_edge"] = stats["center"] + stats["EigenMinorVector"] * (stats["width"] * 0.5)
        print(stats)
        print(f"Eigen ratio {eigen_ratio}")
        return stats

    def debug_image(self, in_image):
        """ Draw the eigen vectors/points on the image. Note, this edits the image
        @param in_image: The image to draw on top of"""
        p1 = self.stats_dict["lower_left"]
        p2 = self.stats_dict["upper_right"]
        LineSeg2D.draw_line(in_image, p1, p2, (220, 128, 220), 2)

        p1 = self.stats_dict["left_edge"]
        p2 = self.stats_dict["right_edge"]
        LineSeg2D.draw_line(in_image, p1, p2, (128, 128, 128), 2)

        pc = self.stats_dict["center"]
        LineSeg2D.draw_cross(in_image, pc, (256, 256, 128), 1, 2)

        xmin = self.stats_dict["x_min"]
        xmax = self.stats_dict["x_max"]
        ymin = self.stats_dict["y_min"]
        ymax = self.stats_dict["y_max"]
        LineSeg2D.draw_rect(in_image, [[xmin, xmax], [ymin, ymax]], (256, 128, 128), 2)


if __name__ == '__main__':
    path_bpd = "./data/trunk_segmentation_names.json"
    #path_bpd = "./data/forcindy_fnames.json"
    all_files = HandleFileNames.read_filenames(path_bpd)

    b_do_debug = True
    b_do_recalc = False
    for ind in all_files.loop_masks():
        mask_fname = all_files.get_mask_name(path=all_files.path, index=ind, b_add_tag=True)
        mask_fname_debug = all_files.get_mask_name(path=all_files.path_debug, index=ind, b_add_tag=True)
        if not b_do_debug:
            mask_fname_debug = ""

        mask_fname_calculate = all_files.get_mask_name(path=all_files.path_calculated, index=ind, b_add_tag=False)
        mask_fname_calculate = mask_fname_calculate + ".json"

        if not exists(mask_fname):
            raise ValueError(f"Error, file {mask_fname} does not exist")
        b_stats = BaseStatsImage(mask_fname, mask_fname_calculate, mask_fname_debug, b_recalc=b_do_recalc)
