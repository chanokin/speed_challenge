import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

def train_generator(data, batch_size=32):
    rows, cols, channels = 480, 640, 4
    
    img_batch = np.zeros((batch_size, rows, cols, channels))
    lbl_batch = np.zeros(batch_size)
#     files = sorted(glob.glob('./train/*.npz')) == data
    nfiles = len(data)
    
    while 1:
        data_ids = np.random.choice(nfiles, size=batch_size, replace=False)
        data_files = [np.load(data[i]) for i in data]
        for i, d in enumerate(data_files):
            img_batch[i, :, :, 0:3] = d['color']
            img_batch[i, :, :, 3] = d['dvs']
            lbl_batch[i] = d['label']

        yield (img_batch, lbl_batch)
            
def test_generator(data, batch_size=32):
    rows, cols, channels = 480, 640, 4
    
    img_batch = np.zeros((batch_size, rows, cols, channels))
    lbl_batch = np.zeros(batch_size)
#     files = sorted(glob.glob('./train/*.npz')) == data
    nfiles = len(data)
    
    while 1:
        data_ids = np.random.choice(nfiles, size=batch_size, replace=False)
        data_files = [np.load(data[i]) for i in data]
        for i, d in enumerate(data_files):
            img_batch[i, :, :, 0:3] = d['color']
            img_batch[i, :, :, 3] = d['dvs']
            lbl_batch[i] = np.nan

        yield (img_batch, lbl_batch)
            
