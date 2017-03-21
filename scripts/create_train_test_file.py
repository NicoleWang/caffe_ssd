#! /usr/bin/python

import os, sys
import glob
import random
#trainval_dir = "/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/trainval"
#test_dir = "/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/test"
#image_dir = "/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/part_images"
image_dir = "/home/wangyuzhuo/Data/train_data/chn_data/JPEGImages"
xml_dir = "/home/wangyuzhuo/Data/train_data/chn_data/LineXMLAnnotations"

all_file = os.listdir(xml_dir)
all_images = []
for fname in all_file:
    imname = fname[:-4] + ".jpg"
    all_images.append(imname)

random.shuffle(all_images)
trainval_img_lists = all_images[20:]
trainval_img_names = []
for item in trainval_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    trainval_img_names.append(temp1)

test_img_lists = all_images[0:20]
test_img_names = []
for item in test_img_lists:
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    print temp1, temp2
    test_img_names.append(temp1)

dist_img_dir = "chn/JPEGImages"
dist_anno_dir = "chn/LineXMLAnnotations"

trainval_fd = open("/home/wangyuzhuo/Experiments/ssd_git/caffe_ssd/data/chn/trainval.txt", 'w')
test_fd = open("/home/wangyuzhuo/Experiments/ssd_git/caffe_ssd/data/chn/test.txt", 'w')

for item in trainval_img_names:
    trainval_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')

for item in test_img_names:
    test_fd.write(dist_img_dir + '/' + str(item) + '.jpg' + ' ' + dist_anno_dir + '/' + str(item) + '.xml\n')
