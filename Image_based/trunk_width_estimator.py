import os
import sys
import cv2
import numpy as np

# from trunk_segmenter import TrunkSegmenter
import skimage.morphology
import pandas as pd

# from env_vars import *

class TrunkWidthEstimator:
    def __init__(self):

        # Initialize predictor using the configuration object
        # self.segmenter = TrunkSegmenter()
        pass

    #function that takes slices angled by the PA and returns the medial axis
    def get_st(self, medial_axis, return_distance, image):
        """
        Get the indices of the portion of the medial axis that will be kept

        Args:
            medial_axis (): Medial axis of the mask.
            return_distance (): Distance between medial axis and the mask boundary.

        Returns:
            real_idx (): Indices of the portion of the medial axis that will be kept.
            return_distance1 (): Distance between medial axis and the mask boundary for the portion of the medial axis that will be kept.
        """

        # Get the number of medial axes in each row
        axes_per_row = medial_axis.sum(axis=1)

        # Find the longest medial axis, where there is only one medial axis in the row for the duration
        pre = 0
        mlen, mstart, mend = 0, -1, -1
        tlen, tstart, tend = 0, -1, -1
        for i in range(axes_per_row.shape[0]):
            # If there is one medial axis in the row, then start counting the length of the medial axis
            if axes_per_row[i] == 1:
                if pre == 0:
                    pre = 1
                    tstart = i
                    tlen = 0
                tlen += 1
            else:
                if axes_per_row[max(i - 1, 0)] == 1 and axes_per_row[min(i + 1, image.shape[0] - 1)] == 1:
                    medial_axis[i] = 0
                elif pre == 1:
                    pre = 0
                    # print(tlen,tstart,i)
                    # print(mlen,mstart,mend)
                    # print('---')
                    if tlen > mlen:
                        mlen = tlen
                        mend = i
                        mstart = tstart
                    # tlen=0

        if tlen > mlen:
            mlen = tlen
            mend = axes_per_row.shape[0]
            mstart = tstart
            tlen = 0

        # Make 1d mask of the longest medial axis
        b2 = np.zeros_like(axes_per_row)
        b2[mstart:mend] = 1

        # Make 2d mask of the longest medial axis
        medial_axis = medial_axis * b2[:,np.newaxis].astype(bool)
        # Get the distance from the medial axis to the edge of the mask for each row along the medial axis
        return_distance0 = return_distance * medial_axis
        return_distance1 = np.max(return_distance0,axis=1)
        return_distance2 = return_distance1[mstart:mend]

        # Take the cumulative sum, then cut off the first 20% and last 20% of the medial axis
        diff = mend - mstart
        diff2 = int(diff * 0.2)
        return_distance3 = np.cumsum(return_distance2)[:-diff2]
        return_distance3 = return_distance3[diff2:]

        # Take the difference between the cumulative sum and the cumulative sum shifted by 20 pixels
        return_distance4 = return_distance3[20:] - return_distance3[:-20]

        # Find the indices of the 40% of the remaining medial axis with the smallest distance to the edge of the mask
        k = int(return_distance4.shape[0] * 0.4)
        idx1 = np.argpartition(return_distance4, k)[:k]
        real_idx = idx1 + mstart + 10
        real_idx += diff2

        return real_idx, return_distance1

    def calculate_width(self, mask, pc, image):

        # The function returns two arrays: medial_axis containing the boolean mask of the medial axis, and return_distance
        # containing the distance transform of the binary mask, which assigns to each pixel the distance to the closest background pixel.
        medial_axis, return_distance = skimage.morphology.medial_axis(mask, return_distance=True)
        #convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, medial_axis1 = self.get_st(medial_axis, return_distance, image)

        # Replace all nan values with 0
        pc2 = np.where(np.isnan(pc), 0, pc)

        #
        pc3 = medial_axis * pc2

        pc4 = pc3.sum(axis=1)

        depth2 = pc4[height]

        linelength2 = medial_axis1[height] * 2

        depth3 = depth2[np.nonzero(depth2)]

        d_med = np.median(depth3)

        depth4 = np.where(abs(depth2 - d_med) > 1, 0, depth2)

        if np.sum(depth4) == 0:
            depth5 = pc3[np.nonzero(pc3)]
            depth4 = np.median(depth5)

        imheight2 = depth4 * np.tan(np.deg2rad(21)) * 2

        distperpix2 = imheight2 / image.shape[0]

        width4 = linelength2 * distperpix2

        self.width = np.max(width4)

    def get_width(self, image, pc, mask):
        """
        Get the width of the trunk in the image
        Args:
            image (): Image to get the width of the trunk from.
            pc (): Pointcloud of the image, in this case actually just the distance from the camera to the point,
            so the z values from the pointcloud.

        Returns:
            width (float): Width of the trunk in the image.
            vis_imgs2 (): Image with the mask drawn on it.
            mask (): Mask of the trunk.
        """

        # Get the width of the trunk
        self.calculate_width(mask, pc, image)
        return self.width











