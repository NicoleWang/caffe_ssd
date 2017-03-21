#! /usr/bin/python

import os, sys
import glob
from PIL import Image

img_dir = "/home/wangyuzhuo/Data/train_data/chn_data/JPEGImages"
xml_dir = "/home/wangyuzhuo/Data/train_data/chn_data/LineXMLAnnotations"
file_list = os.listdir(xml_dir)
img_list = []
for fname in file_list:
    imname = fname[:-4] + ".jpg"
    impath = os.path.join(img_dir, imname)
    img_list.append(impath)
#img_lists = glob.glob(img_dir + '/*.jpg')

test_name_size = open('/home/wangyuzhuo/Experiments/ssd_git/caffe_ssd/data/chn/test_name_size.txt', 'w')

for item in img_list:
    img = Image.open(item)
    width, height = img.size
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_name_size.write(temp1 + ' ' + str(height) + ' ' + str(width) + '\n')
