#! /usr/bin/python

import os, sys
import glob
from PIL import Image

img_dir = "/home/wangyuzhuo/Data/train_data/chn_data/JPEGImages"

img_lists = glob.glob(img_dir + '/*.jpg')

test_name_size = open('/home/wangyuzhuo/Experiments/ssd/caffe/data/chn/test_name_size.txt', 'w')

for item in img_lists:
    img = Image.open(item)
    width, height = img.size
    temp1, temp2 = os.path.splitext(os.path.basename(item))
    test_name_size.write(temp1 + ' ' + str(height) + ' ' + str(width) + '\n')
