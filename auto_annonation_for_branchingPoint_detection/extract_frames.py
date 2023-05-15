import sys
import argparse

import cv2
print(cv2.__version__)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def extractImages(pathIn, pathOut):
    count = 0
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    success = True
    while success:
        #BGR
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count*100))    # added this line
        success, image = vidcap.read()
        print ('Read a new frame: ', success)
        if not success:
            return 0
        # cv2.imshow('image',image)
        ind = np.logical_and(image[:, :, 0] >= 200, image[:, :, 1] <= 15, image[:, :, 2] <= 15)
        image[ind, 0] = 255
        image[ind, 1] = 0
        image[ind, 2] = 0
        ind1 = np.logical_and(image[:, :, 0] >= 90, image[:, :, 1] == 0, image[:, :, 2] == 0)
        image[ind1, 0] = 255
        image[ind1, 1] = 0
        image[ind1, 2] = 0
        image = np.asarray(image, dtype=np.uint8)
        #convert to RGB to show image
        # plt.imshow(image[:,:,::-1])
        # plt.show()
        # cv2.imwrite( pathOut + "img%d.jpg" % count, image)     # save frame as JPEG file
        dim = (640, 480)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        img = Image.fromarray(image[:,:,::-1], 'RGB')

        img.save(pathOut + "img%d.jpg" % count)
        count = count + 1


if __name__=="__main__":
    # a = argparse.ArgumentParser()
    # a.add_argument("--pathIn", help="path to video")
    # a.add_argument("--pathOut", help="path to images")
    pathsIn = ["/home/josyula/Documents/DataAndModels/ufo_trees_labelled/render/unlabelled.mkv",
             "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/render/unlabelled1.mkv",
               "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/render/unlabelled2.mkv",]
    pathsOut = ["/home/josyula/Documents/DataAndModels/ufo_trees_labelled/unlabelled/",
                "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/unlabelled1/",
                "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/unlabelled2/"]
    # pathIn = "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/render/labelled2.mkv"
    # pathOut = "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/labelled3/"
    # args = a.parse_args()
    # print(args)
    for i in range(len(pathsIn)):
        extractImages(pathsIn[i], pathsOut[i])