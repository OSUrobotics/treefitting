import glob
from collections import defaultdict
from copy import deepcopy

import numpy as np
import cv2
import json
from PIL import Image # (pip install Pillow)
import numpy as np                                 # (pip install numpy)
from skimage import measure                        # (pip install scikit-image)
from shapely.geometry import Polygon, MultiPolygon # (pip install Shapely)
import matplotlib.pyplot  as plt

image_dict = {
            "id": 0,
            "license": 1,
            "file_name": "",
            "height": 480,
            "width": 640,
            "date_captured": "2020-07-20T19:39:26+00:00",
            "annotated":"true",
            "category_ids":[0],
            "metadata":{}
}
def create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    sub_mask1 = np.array(sub_mask)
    contours = measure.find_contours(sub_mask1, 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    bbox = []
    area = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)

        # Make a polygon and simplify it
        poly = Polygon(contour)
        # poly = poly.simplify(0.5, preserve_topology=False)
        polygons.append(poly)
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)
        x, y, max_x, max_y = poly.bounds
        w = max_x - x
        h = max_y - y
        bbox.append([x, y, w, h])
        area.append(poly.area)

    # Combine the polygons to calculate the bounding box and area
    # multi_poly = MultiPolygon(polygons)
    # x, y, max_x, max_y = multi_poly.bounds
    # width = max_x - x
    # height = max_y - y
    # bbox = (x, y, width, height)
    # area = multi_poly.area

    annotation = {
        'segmentation': segmentations,
        'iscrowd': is_crowd,
        'image_id': image_id,
        'category_id': category_id,
        'id': annotation_id,
        'bbox': bbox,
        'area': area
    }

    return annotation

def create_sub_masks(mask_image):
    width, height = mask_image.size
    # m_img = np.array(mask_image.getdata())
    # m_img = m_img.reshape((height, width, 3))
    # plt.imshow(m_img)
    # plt.show()
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x,y))[:3]
            # if any(np.array(pixel)==255):
            #     print(pixel)
            # If the pixel is not black...
            if pixel[0] >= 230 and pixel[1] <= 10 and pixel[2] <= 10:
                # Check to see if we've created a sub-mask...
                pixel_str = str(pixel)
                sub_mask = sub_masks.get(pixel_str)
                if sub_mask is None:
                   # Create a sub-mask (one bit per pixel) and add to the dictionary
                    # Note: we add 1 pixel of padding in each direction
                    # because the contours module doesn't handle cases
                    # where pixels bleed to the edge of the image
                    sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

                # Set the pixel value to 1 (default is 0), accounting for padding
                sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks

def read_image(path):
    image = Image.open(path)
    image = image.convert('RGBA')
    return image

def main():

    labelled_images = glob.glob("/home/josyula/Documents/DataAndModels/ufo_trees_labelled/labelled/*.jpg")
    unlabelled_images = glob.glob("/home/josyula/Documents/DataAndModels/ufo_trees_labelled/unlabelled/*.jpg")
    # for image in images:
    #     mask_image = read_image(image)
    #     sub_masks = create_sub_masks(mask_image)
    #     print(sub_masks)
    #
    # import json
    import json_dict
    mask_images =  []
    json_dict1 = deepcopy(json_dict.json_dict)
    for image_ in labelled_images:
        mask_image = Image.open(image_)
        image = np.array(mask_image)
        # image = image[:,:,::-1]
        ind = np.logical_and(image[:, :, 2] >= 200, image[:, :, 0] <= 15, image[:, :, 1] <= 15)
        image[ind, 0] = 255
        image[ind, 1] = 0
        image[ind, 2] = 0
        ind1 = np.logical_and(image[:, :, 0] >= 90, image[:, :, 1] == 0, image[:, :, 2] == 0)
        image[ind1, 0] = 255
        image[ind1, 1] = 0
        image[ind1, 2] = 0
        image = np.asarray(image, dtype=np.uint8)
        mask_images.append(Image.fromarray(image))

    # Define which colors match which categories in the images
    cat_id = 0
    category_ids = {
            '(255,0,0)': cat_id,
    }

    annotations = []
    images = []

    is_crowd = 0
    annotation_id = 0
    # category_ids = defaultdict(dict)
    image_id = 0
    for mask_image in mask_images:
        sub_masks = create_sub_masks(mask_image)
        for pixel_str, sub_mask in sub_masks.items():
            category_ids.update({pixel_str: 0})
        for color, sub_mask in sub_masks.items():
            category_id = category_ids[color]
            annotation = create_sub_mask_annotation(sub_mask, image_id, category_id, annotation_id, is_crowd)
            annotations.append(annotation)
            annotation_id += 1
        image_dict.update({'file_name': labelled_images[image_id].split("/")[-1], 'id': image_id, "path": labelled_images[image_id]})
        images.append(deepcopy(image_dict))
        image_id += 1
    # print(category_ids)
    # print(json.dumps(annotations))
    json_dict1["annotations"]+=annotations
    json_dict1["images"]+=images

    with open('annotations.json', 'w') as outfile:
        json.dump(json_dict1, outfile)
if __name__ == '__main__':
    main()