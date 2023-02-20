import os
import torch
import random
import cv2
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from detectron2.data.datasets import load_sem_seg, register_coco_panoptic_separated, register_coco_panoptic, register_coco_instances

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, GenericMask
from detectron2.data import MetadataCatalog, DatasetCatalog

# for mask2former
from detectron2.projects.deeplab import add_deeplab_config
import sys
project_path='/nfs/stak/users/wangtie/2022/8/13/AgAID2022/'
project_path1=project_path+'Mask2Former/'
sys.path.insert(1, project_path1)
from mask2former import add_maskformer2_config
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, BoxMode
import pycocotools.mask as mask_util

from shapely.geometry import LineString, Point
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

# pretrained model


config_file=project_path1+"/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
weight_file=project_path1+"/output20220612_5instances/model_final.pth"

dataset_root = project_path+'/datasets/tree_trunk_20220612/datasets'
train_gt_root = os.path.join(dataset_root, "train_gt")
train_image_root = os.path.join(dataset_root, "train_rgb")

coco_metadata = MetadataCatalog.get('coco_2017_train_panoptic_with_sem_seg').as_dict()
for d in ["train", "val"]:
    dataset_name = "treepruning2_"+d
    image_root = os.path.join(dataset_root, "train_rgb")

    MetadataCatalog.get(dataset_name).set(thing_classes=["trunk"])

    metadata = MetadataCatalog.get(dataset_name).as_dict()
    metadata['stuff_dataset_id_to_contiguous_id'] = coco_metadata['stuff_dataset_id_to_contiguous_id']
    instances_json = os.path.join(dataset_root, "tree_trunk_ann_instance_train.json")


    register_coco_instances(dataset_name, {}, instances_json, image_root)
MetadataCatalog.get("treeprunung2_val").set(evaluator_type='coco_panoptic_seg')

# from detectron2.engine import DefaultTrainer
from train_net8 import Trainer

# pretrained model
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file(config_file)
cfg.DATASETS.TRAIN = ("treepruning2_train",)
cfg.DATASETS.TEST  = ("treepruning2_val",)
cfg.DATALOADER.NUM_WORKERS = 1
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES=1
cfg.MODEL.RESNETS.NORM="BN"
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.MODEL.WEIGHTS = weight_file
cfg.freeze()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

predictor = DefaultPredictor(cfg)

from skimage.measure import find_contours
def draw_mask(img, pts, color, alpha=0.5):
    h, w, _ = img.shape

    overlay = img.copy()
    output = img.copy()

    cv2.fillPoly(overlay, pts, color)
    output = cv2.addWeighted(overlay, alpha, output, 1 - alpha,
                    0, output)
    return output
    
def get_mask_contours(mask):
    #mask = masks[:, :, i]
    # Mask Polygon
    # Pad to ensure proper polygons for masks that touch image edges.
    contours_mask = []
    padded_mask = np.zeros(
        (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask
    contours = find_contours(padded_mask, 0.5)
    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        contours_mask.append(np.array(verts, np.int32))
    return contours_mask

def test(predictions, score_thresh,v,image):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None

    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if score_thresh != 0:
#         keep = ((scores > score_thresh)&(classes==1))
        keep = scores > score_thresh
        scores = scores[keep]
        boxes= boxes[keep]
        classes = classes[keep]
        
        if predictions.has("pred_keypoints"):
            keypoints = keypoints[keep]
    classes = classes.tolist()
    
#     labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        print(masks.shape)
        if score_thresh != 0:
            masks = masks[keep]
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
    else:
        masks = None
#     print('AAAAAAA')
#     print(type(masks))
    print("2fdskljfds")
    print(len(masks))
#     print(masks[0].mask.shape)
    id=0
    lenmax=0
#     mask_r=np.zeros_like(masks[0].mask)
    for id in range(len(masks)):
    
        colors = [0, 0, 255]
    #     for i in range(2):
        contours = get_mask_contours(masks[id].mask)
        print('#######################')
        print(type(contours))
        print(len(contours))
        #draw the mask, good for debugging
        cnt2=0
        for cnt in contours:
            cv2.polylines(image, [cnt], True, colors[0], 2)
            image = draw_mask(image, [cnt], colors[0])
#         mask_r|=masks[id].mask
#     mask_r*=255
#     mask_r.dtype='uint8'
    return image#,mask_r



#function that takes slices angled by the PCA
class PCAslice2(object):
    #initialize
    def __init__(self,mask,pc,image):
        #height at which the slice is taken
#         self.height = height
        #width of the image
#         self.width=width
        #mask for the image
        self.mask=mask
        #point cloud for the image
        self.pc=pc
        #the image
        self.image=image
        self.width1=1
        self.width2=[]
        self.width3=[]
        self.imheight=[]
        self.depth=[]
        self.slicewidth=[]
    def get_st(self,medial_axis,return_distance):
        # self.image[medial_axis] = (0, 200, 255)
        medial_axis1=medial_axis.sum(axis=1)
        # print(medial_axis1)
        
        a=np.array(medial_axis1)
        pre=0
        mlen,mstart,mend=0,-1,-1
        tlen,tstart,tend=0,-1,-1
        for i in range(a.shape[0]):
            if a[i]==1:
                if pre==0:
                    pre=1
                    tstart=i
                    tlen=0
                tlen+=1
            else:
                if a[max(i-1,0)]==1 and a[min(i+1,self.image.shape[0]-1)]==1:
                    medial_axis[i]=0
                elif pre==1:
                    pre=0
                    # print(tlen,tstart,i)
                    # print(mlen,mstart,mend)
                    # print('---')
                    if tlen>mlen:
                        mlen=tlen
                        mend=i
                        mstart=tstart
                    # tlen=0
                    
        if tlen>mlen:
            mlen=tlen
            mend=a.shape[0]
            mstart=tstart
            tlen=0
        
        mlen=a.shape[0]
        # a=np.arange(18).reshape(6,3)
        # b=np.arange(mlen)
        # b1=np.where((b>=mstart)&(b<mend))
        # print(mstart,mend)
        b2=np.zeros_like(a)
        b2[mstart:mend]=1
        # a2=a*b2[:,np.newaxis]
        # print(b2)
        medial_axis=medial_axis*b2[:,np.newaxis].astype(bool)
        return_distance0=return_distance*medial_axis
        return_distance1=np.max(return_distance0,axis=1)
        return_distance2=return_distance1[mstart:mend]
        
        # mrd=np.median(return_distance2)
        # while 1:
        #     if return_distance1[mstart]*3<mrd:
        #         mstart+=1
        #     else:
        #         break
        # while 1:
        #     if return_distance1[mend-1]*3<mrd:
        #         mend-=1
        #     else:
        #         break
        # return_distance2=return_distance1[mstart:mend]
        
        diff=mend-mstart
        # if diff>80:
        diff2=int(diff*0.2)
        return_distance3=np.cumsum(return_distance2)[:-diff2]
        return_distance3=return_distance3[diff2:]

        # else:
        #     return_distance3=np.cumsum(return_distance2)
        return_distance4=return_distance3[20:]-return_distance3[:-20]
        # print(return_distance4)
        k=int(return_distance4.shape[0]*0.4)
        idx1 = np.argpartition(return_distance4, k)[:k] 
        # print(idx1)
        idx=idx1[0]
        # print(id0x)
        # print(return_distance4[idx])
        real_idx=idx1+mstart+10
        # if diff>200:
        real_idx+=diff2
        # print(real_idx)
        return_distance5=return_distance4[idx]/20*2
        # print(return_distance5)
        # self.image[real_idx,:] = (0, 0, 255)
        
        # self.image[medial_axis] = (0, 200, 255)
        return real_idx,return_distance1
    
    def GetWidth(self):
        #pull up the mask
        # print(np.max(self.mask))

        medial_axis,return_distance=skimage.morphology.medial_axis(self.mask, return_distance=True)
        # print(medial_axis.shape)
        # medial_axis0=medial_axis.sum(axis=0)
        # medial_axis1=medial_axis.sum(axis=1)
        # print(medial_axis1.shape)
        height,medial_axis1=self.get_st(medial_axis,return_distance)
        # print('fdajkljfdaljfdklajklfdjaklj')
        # print(medial_axis1.shape)
        zs2=[]
        #         print(self.pc[height,:,2])
        # print(self.pc[:,:,2].shape)
        pc2=np.where(np.isnan(self.pc),0,self.pc)
        # print(self.pc2)
        # print(self.pc2.min())
        pc3=medial_axis*pc2
        pc4=pc3.sum(axis=1)
        depth2 = pc4[height]
        linelength2=medial_axis1[height]*2
        
        depth3=depth2[np.nonzero(depth2)]
        d_med=np.median(depth3)
        depth4=np.where(abs(depth2-d_med)>1,0,depth2)
        
        if np.sum(depth4)==0:
            depth5=pc3[np.nonzero(pc3)]
            depth4=np.median(depth5)
        
#         print(self.image.shape)
        imheight2=depth4*np.tan(np.deg2rad(21))*2
        distperpix2=imheight2/self.image.shape[0]
        self.width4=linelength2*distperpix2       
        self.width5=np.max(self.width4)
        print(self.width5) 

import skimage.morphology

#function that gets the widths from the .csv
import pandas as pd
def getwidths3(row):
    df=pd.read_csv('row_{}_diameters.csv'.format(row))  
    if row!=96:
        m1=df['m1'].to_numpy()/100.0
        m2=df['m2'].to_numpy()/100.0
        m3=(m1+m2)/2.0
    else:
        m3=df['Average'].to_numpy()/1000.0
    return m3

def test2(predictions, score_thresh,v,image):
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

    if score_thresh != 0:
        keep = scores > score_thresh
        scores = scores[keep]
        boxes= boxes[keep]
        classes = classes[keep]
        
        if predictions.has("pred_keypoints"):
            keypoints = keypoints[keep]
    classes = classes.tolist()

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
        print(masks.shape)
        if score_thresh != 0:
            masks = masks[keep]
        masks = [GenericMask(x, v.output.height, v.output.width) for x in masks]
    else:
        masks = None
    id=0
    lenmax=0
    print(len(masks))
    if len(masks)==0:
        return image,np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8),-1,-1,-1

    # (720, 1280, 3)
    h,w,_=image.shape
    mid_w=w/2.0
    m_distance=1000000
    for i in range(len(masks)):
        a=np.max(masks[i].mask,axis=0)
        a1=np.nonzero(a)
        a2=np.mean(a1)
        b=np.max(masks[i].mask,axis=1) 
        t_distance=abs(mid_w-a2)
        if sum(b)<image.shape[0]*0.35:continue
        if t_distance<m_distance:
            m_distance=t_distance
            id=i


    colors = [255, 255, 255]
    contours = get_mask_contours(masks[id].mask)
    cnt2=0
    for cnt in contours:
        cv2.polylines(image, [cnt], True, colors, 2)
        image = draw_mask(image, [cnt], colors)

    bt=boxes.tensor[id].numpy().astype('int')
    start_point = (bt[0],bt[1])
    end_point = (bt[2],bt[3])
    delta=20
    btxm=bt[0]
    if min(bt[3],bt[1])>image.shape[0]-max(bt[3],bt[1]):
        btym=int((bt[3]+bt[1])/2)
        btxm=bt[2]
    else:
        btym=max(bt[3],bt[1])+delta
    return image,masks[id].mask,scores[id],btxm,btym

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--rownum', type=int,  default=96)
args = parser.parse_args()
def main_test(test_num):
    # Visualize the stump training set

    dataset_dicts = DatasetCatalog.get("treepruning2_train")
    print(len(dataset_dicts))
    dataset_dicts = DatasetCatalog.get("treepruning2_val")
    my_dataset_val_metadata = MetadataCatalog.get("treepruning2_val")

    my_dataset_train_metadata = MetadataCatalog.get("treepruning2_train")
    # save_num=5
    saved_folder_path="row{}_1".format(test_num)
    check_mkdir(saved_folder_path)
    saved_folder_path2="row{}_2".format(test_num)
    check_mkdir(saved_folder_path2)

    training_sample_path =project_path+ "/datasets/row"+str(test_num)
    truewidths=getwidths3(test_num)
    min_t=min(truewidths)
    max_t=max(truewidths)
    #empty arrays for each width
    w5=np.zeros_like(truewidths)
    for idx, data_sample in enumerate(os.listdir(training_sample_path)):
        if data_sample.startswith("."):
            continue
        if data_sample.endswith(".npy"):
            continue
        if data_sample.endswith(".ply"):
            continue
        
        info=data_sample.split('.')[0]
        
        img_path = os.path.join(training_sample_path, data_sample)
        pc_path = os.path.join(training_sample_path, info+'.npy')
    #     print(info)
    #     break
        if test_num==96:
            info2=int(info)-200
        else:
            info2=int(info.split('_')[1])
    #     if info2!=28:continue
        print(data_sample)
        # if data_sample!='1659633226725213528_74.png':continue
        print(info2)
        pc = np.load(pc_path)
        
    #     print(img_path)
        
        im = cv2.imread(img_path)
        outputs = predictor(im) 
        pred_classes = outputs['instances'].pred_classes
        
        # outputs['instances'].scores[pred_classes != 1 ] = 0
        # print(outputs['instances'].pred_classes)
        empty_img = np.zeros_like(im[:, :, ::-1])
        
        # v = Visualizer(im[:,:,::-1], metadata=my_dataset_val_metadata, scale=1.0)
        v = Visualizer(empty_img, metadata=my_dataset_val_metadata, scale=1.0,)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"),score_thresh=0.3)
        vis_imgs2,mask,scores,x,y =test2(outputs["instances"].to("cpu"),0.2,v,im)
        if scores!=-1:
            a = PCAslice2(mask, pc, im)
            a.GetWidth()
  
        print(truewidths[info2-1])
        w5[info2-1]=a.width5
        save_path = os.path.join(saved_folder_path, data_sample)
        save_path2 = saved_folder_path2+'/'+info+'.npy'
        cv2.imwrite(save_path,vis_imgs2)
        np.save(save_path2, mask)
        # break

    linex=[min_t,max_t]
    liney=[min_t,max_t]
    saved_folder_path3="row{}_3".format(test_num)
    check_mkdir(saved_folder_path3)
    np.save(saved_folder_path3+"/w5", w5)
    w5error = np.sum(np.abs(np.subtract(w5, truewidths)))
    print("w5 total error:", w5error)
    plt.plot(truewidths,w5,'g*',linex,liney)
    plt.ylabel('Estimates (m)')
    plt.xlabel('Ground Truth (m)')
    plt.show()
main_test(args.rownum)