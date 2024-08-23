#!/usr/bin/env python3

# A 2D radius extraction method

import numpy as np
import json
import cv2

@staticmethod
def image_cutout(im, rect, step_size, height):
    """Cutout a warped bit of the image and return it
    @param im - the image rect is in
    @param rect - four corners of the rectangle to cut out
    @param step_size - the length of the destination rectangle
    @param height - the height of the destination rectangle
    @returns an image, and the reverse transform"""
    rect_destination = np.array([[0, 0], [step_size, 0], [step_size, height], [0, height]], dtype="float32")
    tform3 = cv2.getPerspectiveTransform(rect, rect_destination)
    tform3_back = np.linalg.pinv(tform3)
    return cv2.warpPerspective(im, tform3, (step_size, height)), tform3_back