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
    files = sorted(glob.glob('./train/*.npz'))
    nfiles = len(files)
    
    while 1:
        for i in range(batch_size):
            f = 
