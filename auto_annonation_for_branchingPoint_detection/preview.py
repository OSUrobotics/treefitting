from PIL import Image
from pycocotools.coco import COCO
from lxml import etree
import os
import cv2
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


sys.path.insert(0, '../libs')

from imantics import Dataset

with open('./annotations.json', 'r') as f:
    data = json.load(f)

# https://stackoverflow.com/questions/50805634/how-to-create-mask-images-from-coco-dataset
dataset = Dataset.from_coco(data)
coco = COCO('./annotations.json')
image_id = 21
img_dir = "/home/josyula/Documents/DataAndModels/ufo_trees_labelled/labelled/"
img = coco.imgs[image_id]
cat_ids = coco.getCatIds()
image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(image, interpolation='nearest')
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
coco.showAnns(anns)
# for i, ann in enumerate(anns):
#     ax.text(anns[i]['bbox'][0], anns[i]['bbox'][1], anns[i]['category_id'], style='italic',
#             bbox={'facecolor': 'red', 'alpha': 1, 'pad': 15})

for ann in anns:
    boxes = ann['bbox']
    for box in boxes:
        bb = patches.Rectangle((box[0],box[1]), box[2],box[3], linewidth=2, edgecolor="blue", facecolor="none")
        ax.add_patch(bb)
# mask = coco.annToMask(anns[0])>0
# for i in range(len(anns)):
#      mask += coco.annToMask(anns[i])>0

# plt.imshow(mask,cmap='gray')
plt.show()
