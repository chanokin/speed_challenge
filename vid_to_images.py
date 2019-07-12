# coding: utf-8
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

labels_file = open('./data/train.txt', 'r')
labels = [float(line) for line in labels_file]

vid = cv2.VideoCapture('./data/train.mp4')
nframes = float(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

ret, frame = vid.read()
rows, cols, channels = frame.shape

thresh = 0.20
color = np.zeros_like(frame, dtype='float')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float')/255.0
ref = gray.copy()
diff = np.zeros_like(gray)
dvs = np.zeros_like(gray)
out_path = './train_%03dp'%(np.round(thresh*100.0))
os.makedirs(out_path, exist_ok=True)


print("Processing training data")
frame_num = 0
while(1):
    try:
        sys.stdout.write("\r%06.2f%%"%((100.0*frame_num)/nframes))
        sys.stdout.flush()

        ret, frame[:] = vid.read()
        color[:] = frame
        color *= (1.0/255.0)

        gray[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float')
        gray *= (1.0/255.0)
        
        diff[:] = gray - ref
        dvs[:] = diff * (np.abs(diff) > thresh)
        
        ref += dvs

        fname = os.path.join(out_path, 'data_%010d.npz'%frame_num)
        np.savez_compressed(fname, 
            color=color, gray=gray, dvs=dvs, label=labels[frame_num])

        frame_num += 1
    except:
        break

sys.stdout.write("\n")
sys.stdout.flush()


vid = cv2.VideoCapture('./data/test.mp4')
nframes = float(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

ret, frame = vid.read()
rows, cols, channels = frame.shape

color = np.zeros_like(frame, dtype='float')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float')/255.0
ref = gray.copy()
diff = np.zeros_like(gray)
dvs = np.zeros_like(gray)
out_path = './test_%03dp'%(np.round(thresh*100.0))
os.makedirs(out_path, exist_ok=True)

print("\nProcessing test data")
frame_num = 0
while(1):
    try:
        sys.stdout.write("\r%06.2f%%"%((100.0*frame_num)/nframes))
        sys.stdout.flush()

        ret, frame[:] = vid.read()
        color[:] = frame
        color *= (1.0/255.0)

        gray[:] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float')
        gray *= (1.0/255.0)
        
        diff[:] = gray - ref
        dvs[:] = diff * (np.abs(diff) > thresh)
        
        ref += dvs

        fname = os.path.join(out_path, 'data_%010d.npz'%frame_num)
        np.savez_compressed(fname, 
            color=color, gray=gray, dvs=dvs, label=np.nan)

        frame_num += 1
    except:
        break

print("")

