# Demo of the salient object detector
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import glob2
from functions.SOD_class import SOD
import cv2
import os
from tqdm import tqdm

sod_model = SOD(weights_path = '/home/lgleznah/Doctorado/SOD-weights-dataset/sod_cnn_weights.h5',
                center_path = '/home/lgleznah/Doctorado/SOD-weights-dataset/center100.npy')

fns = ['birds.jpg']
res_all = sod_model.predict(fns, refine = True, verbose = True)

save_dir = 'results'
if not os.path.isdir(save_dir):
  os.mkdir(save_dir)

for i in tqdm(range(len(res_all))):
    fname = fns[i]
    I = cv2.imread(fname)
    res = res_all[i].astype(int)
    if res.size == 0:
        cv2.imwrite(save_dir + '/' + fname, I)
    else:
        for j in range(res.shape[1]):
            rect = res[:, j]
            I = cv2.rectangle(I, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), thickness = 5)
        cv2.imwrite(save_dir + '/' + fname, I)
