import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from scipy import ndimage

thr = 0.25

def train_generator(data, batch_size=32):
#     rows, cols, channels = 480, 640, 5
#     rows, cols, channels = 480, 640, 3
    rows, cols, channels = 480, 640, 2
    
    img_batch = np.zeros((batch_size, rows, cols, channels))
    lbl_batch = np.zeros(batch_size)
#     files = sorted(glob.glob('./train/*.npz')) == data
    nfiles = len(data)
    
    while 1:
        data_ids = np.random.choice(nfiles, size=batch_size, replace=False).astype('int')
        data_files = [np.load(data[i]) for i in data_ids]
        for i, d in enumerate(data_files):
#             img_batch[i, :, :, 0:3] = d['color']
#             img_batch[i, :, :, 3] = d['dvs'] * (d['dvs'] > 0)
#             img_batch[i, :, :, 4] = -d['dvs'] * (d['dvs'] < 0)

            tmp = d['dvs'] * (d['dvs'] > 0)
            tmp = ndimage.gaussian_filter(tmp, 1)
            sx = ndimage.sobel(tmp, axis=0, mode='constant')
            sy = ndimage.sobel(tmp, axis=1, mode='constant')
            img_batch[i, :, :, 0] = np.hypot(sx, sy)
            img_batch[i, :, :, 0] *= (dvs > thr)

            
            tmp = -d['dvs'] * (d['dvs'] < 0)
            tmp = ndimage.gaussian_filter(tmp, 1)
            sx = ndimage.sobel(tmp, axis=0, mode='constant')
            sy = ndimage.sobel(tmp, axis=1, mode='constant')
            img_batch[i, :, :, 1] = np.hypot(sx, sy)
            img_batch[i, :, :, 1] *= (dvs > thr)
        
#             img_batch[i, :, :, 0] = d['dvs'] * (d['dvs'] > 0)        
#             img_batch[i, :, :, 1] = -d['dvs'] * (d['dvs'] < 0)

#             img_batch[i, :, :, 0] = d['gray']
#             img_batch[i, :, :, 1] = d['dvs'] * (d['dvs'] > 0)
#             img_batch[i, :, :, 2] = -d['dvs'] * (d['dvs'] < 0)
            
#             for j in range(3):
#                 isum = np.sum(img_batch[i, :, :, j])
#                 if isum > 0.0:
#                     img_batch[i, :, :, j] *= 1.0/isum

#                 mean = img_batch[i, :, :, j].mean()
#                 norm = img_batch[i, :, :, j].std()
#                 norm = np.sqrt(np.sum(img_batch[i, :, :, j]**2))
#                 img_batch[i, :, :, j] = (img_batch[i, :, :, j] - mean) * (1.0 / norm)

#             mean = img_batch[i, :, :, :].mean()
#             norm = img_batch[i, :, :, :].std()
#             img_batch[i, :, :, :] = (img_batch[i, :, :, :] - mean) * (1.0 / norm)

            lbl_batch[i] = np.round(d['label'], decimals=2)

        yield (img_batch, lbl_batch)
            
def test_generator(data, batch_size=32):
    rows, cols, channels = 480, 640, 5
    
    img_batch = np.zeros((batch_size, rows, cols, channels))
    lbl_batch = np.zeros(batch_size)
#     files = sorted(glob.glob('./test/*.npz')) == data
    nfiles = len(data)
    
    while 1:
        data_ids = np.random.choice(nfiles, size=batch_size, replace=False)
        data_files = [np.load(data[i]) for i in data_ids]
        for i, d in enumerate(data_files):
            img_batch[i, :, :, 0:3] = d['color']
            img_batch[i, :, :, 3] = d['dvs'] * (d['dvs'] > 0)
            img_batch[i, :, :, 4] = -d['dvs'] * (d['dvs'] < 0)
            lbl_batch[i] = np.nan

        yield (img_batch, lbl_batch)
            
