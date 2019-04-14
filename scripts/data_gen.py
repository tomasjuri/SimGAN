
#%%
import os

import numpy as np
import pandas as pd
import random
import tables
import cv2

from config import *

H5 = '/srv/workplace/tjurica/tasks/1544-ANY_defects_detection/pc_dataset/ann_train.h5'

IMG_TABLE = 'images'
ANNOT_TABLE = 'annotations'

DF_IMG_ID = 'img_id'
DF_IMG_PATH = 'img_path'
CENTER_X = 'center_x'
BBOX_LEFT = 'bbox_left'
BBOX_RIGHT = 'bbox_right'
CENTER_Y = 'center_y'
BBOX_UP = 'bbox_up'
BBOX_DOWN = 'bbox_down'
DF_INDICES = 'indices'
DF_CLS_ID = 'cls_id'

class Cropper:
    def __init__(self, h5, dim=(64,64), crops_per_img=50,
        ann_ids=[0,1], driver="H5FD_CORE"):    
        
        self.h5 = h5
        self.dim = dim
        self.crops_per_img = crops_per_img
        self.ith_crop = 0
        self.ann_ids = ann_ids
        self.dirname = os.path.dirname(h5)

        self.pos_radius = int(min(self.dim)/4) #int(min(self.dim)/2)
        self.neg_radius = int(2*np.sqrt((self.dim[0]/2) * (self.dim[0]/2) \
                                   + (self.dim[1]/2) * (self.dim[1]/2)))

        with pd.HDFStore(h5, "r") as hdf_store:
            self.img_df = hdf_store[IMG_TABLE]
            self.ann_df = hdf_store[ANNOT_TABLE]

        self.img_ids = self.img_df[DF_IMG_ID].unique()

        # filter ann ids
        self.ann_df = self.ann_df[
            self.ann_df[DF_CLS_ID].isin(self.ann_ids)]
        self.ann_df = self.ann_df[
            self.ann_df[DF_IMG_ID].isin(self.img_ids)]
        
        self.ann_df[CENTER_X] = (self.ann_df[BBOX_RIGHT] \
                                 + self.ann_df[BBOX_LEFT])//2
        self.ann_df[CENTER_Y] = (self.ann_df[BBOX_DOWN] \
                                 + self.ann_df[BBOX_UP])//2

        self.h5 = tables.open_file(h5, mode='r', driver=driver)
        self.reload_img()

    def reload_img(self):
        rnd_id = random.choice(self.img_ids)
        
        self.img_row = self.img_df[self.img_df[DF_IMG_ID] == rnd_id]
        self.img_annots = self.ann_df[self.ann_df[DF_IMG_ID] == rnd_id]
        
        rel_path = getattr(self.img_row, DF_IMG_PATH).values[0]
        abs_path = os.path.join(self.dirname, rel_path)

        self.img = cv2.imread(abs_path, cv2.IMREAD_UNCHANGED)/255.0

        self.init_annots()
    
    def init_annots(self):
        n = len(self.img_annots)

        mask_shape = (self.img.shape[0], self.img.shape[1], 1)
        pos_mask = np.zeros(mask_shape, dtype=np.float32)
        neg_mask = np.ones(mask_shape, dtype=np.float32)
        bin_mask = np.zeros(mask_shape, dtype=np.float32)
        
        for ann_row in self.img_annots.itertuples():
            ctr_x = int(getattr(ann_row, CENTER_X))
            ctr_y = int(getattr(ann_row, CENTER_Y))
            
            indices_node = getattr(ann_row, DF_INDICES)           
            idcs = self.h5.get_node(indices_node)
            idcs = np.array(idcs)

            bin_mask[idcs[0], idcs[1], 0] = 1.0
            cv2.circle(pos_mask, (ctr_x, ctr_y), self.pos_radius, (1.0), -1)
            cv2.circle(neg_mask, (ctr_x, ctr_y), self.neg_radius, (0.0), -1)

        # remove borders from imgs
        pos_mask[:self.dim[0]//2, :,:] = 0.0
        pos_mask[pos_mask.shape[0]-self.dim[0]//2:, :,:] = 0.0
        pos_mask[:,:self.dim[1]//2,:] = 0.0
        pos_mask[:,pos_mask.shape[1]-self.dim[1]//2:,:] = 0.0
        
        neg_mask[:self.dim[0]//2, :,:] = 0.0
        neg_mask[neg_mask.shape[0]-self.dim[0]//2:, :,:] = 0.0
        neg_mask[:,:self.dim[1]//2,:] = 0.0
        neg_mask[:,neg_mask.shape[1]-self.dim[1]//2:,:] = 0.0

        self.pos_indices = pos_mask.nonzero()
        self.neg_indices = neg_mask.nonzero()
        self.bin_mask = bin_mask

        if len(self.pos_indices[0]) == 0 or len(self.neg_indices[0]) == 0:
            self.reload_img()
        
    def get_positive(self):
        self.ith_crop += 1
        if self.ith_crop > self.crops_per_img:
            self.ith_crop = 0
            self.reload_img()
        
        # get random pos
        n = self.pos_indices[0].shape[0]
        rnd = random.randint(0, n-1)
        ctr = (self.pos_indices[0][rnd],
               self.pos_indices[1][rnd],
               self.pos_indices[2][rnd])

        # crop mask
        crop_mask = self.bin_mask[
            ctr[0]-self.dim[0]//2:ctr[0]+self.dim[0]//2,
            ctr[1]-self.dim[1]//2:ctr[1]+self.dim[1]//2,:]
        
        # crop image
        crop_img = self.img[
            ctr[0]-self.dim[0]//2:ctr[0]+self.dim[0]//2,
            ctr[1]-self.dim[1]//2:ctr[1]+self.dim[1]//2,:]

        # stack together
        return np.dstack((crop_img, crop_mask))
        
    def get_negative(self):
        self.ith_crop += 1
        if self.ith_crop > self.crops_per_img:
            self.ith_crop = 0
            self.reload_img()
        
        # get random neg
        n = self.neg_indices[0].shape[0]
        rnd = random.randint(0, n-1)
        ctr = (self.neg_indices[0][rnd],
               self.neg_indices[1][rnd],
               self.neg_indices[2][rnd])

        # crop mask
        crop_mask = self.bin_mask[
            ctr[0]-self.dim[0]//2:ctr[0]+self.dim[0]//2,
            ctr[1]-self.dim[1]//2:ctr[1]+self.dim[1]//2,:]
        
        # crop image
        crop_img = self.img[
            ctr[0]-self.dim[0]//2:ctr[0]+self.dim[0]//2,
            ctr[1]-self.dim[1]//2:ctr[1]+self.dim[1]//2,:]

        # stack together
        return np.dstack((crop_img, crop_mask))


class PosGenerator:
    def __init__(self, cropper, batch_size):
        self.cropper = cropper
        self.batch_size = batch_size
        
    def next(self):
        batch = np.array([self.cropper.get_positive() \
            for i in range(self.batch_size)])
        return batch

class NegGenerator:
    def __init__(self, cropper, batch_size):
        self.cropper = cropper
        self.batch_size = batch_size
        
    def next(self):
        batch = np.array([self.cropper.get_negative() \
            for i in range(self.batch_size)])
        return batch 
