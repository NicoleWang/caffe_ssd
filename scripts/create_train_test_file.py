#! /usr/bin/python

import os, sys
import glob
import random
#trainval_dir = "/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/trainval"
#test_dir = "/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/test"
#image_dir = "/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/part_images"
image_dir = "/home/wangyuzhuo/Data/train_data/chn_data/JPEGImages"

all_images = os.listdir(image_dir)
random.shuffle(all_images)
trainval_img_lists = all_images[201:]
trainval_img_names = []
for item in trainval_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    trainval_img_names.append(temp1)

test_img_lists = all_images[0:201]
test_img_names = []
for item in test_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_img_names.append(temp1)

dist_img_dir = "chn/JPEGImages"
dist_anno_dir = "chn/XMLAnnotations"

trainval_fd = open("/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/trainval.txt", 'w')
test_fd = open("/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/test.txt", 'w')

for item in trainval_img_names:
    trainval_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')

for item in test_img_names:
    test_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')
