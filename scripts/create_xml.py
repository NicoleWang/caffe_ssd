import os, sys
import glob
import cv2
import string

src_img_dir = "/home/wangyuzhuo/Data/train_data/chn_data/JPEGImages"
src_txt_dir = "/home/wangyuzhuo/Data/train_data/chn_data/Line_Annotations"
out_xml_dir = "/home/wangyuzhuo/Data/train_data/chn_data/LineXMLAnnotations"

'''
img_Lists = glob.glob(src_img_dir + '/*.jpg')
img_basenames = [] # e.g. 100.jpg
for item in img_Lists:
    print item
    img_basenames.append(os.path.basename(item))

img_names = [] # e.g. 100
for item in img_basenames:
    temp1, temp2 = os.path.splitext(item)
    img_names.append(temp1)
'''
img_names = []
filename_list = os.listdir(src_txt_dir)
for tname in filename_list:
    imprefix = tname[:-8]
    img_names.append(imprefix)
    print imprefix

for img in img_names:
    im = cv2.imread((src_img_dir + '/' + img + '.jpg'))
    height, width, depth = im.shape

    # open the crospronding txt file
    gt = open(src_txt_dir + '/' + img + '.jpg.txt').read().splitlines()

    # write in xml file
    #os.mknod(out_xml_dir + '/' + img + '.xml')
    xml_file = open((out_xml_dir + '/' + img + '.xml'), 'w')
    xml_file.write('<annotation>\n')
    xml_file.write('    <folder>VOC2007</folder>\n')
    xml_file.write('    <filename>' + str(img) + '.jpg' + '</filename>\n')
    xml_file.write('    <size>\n')
    xml_file.write('        <width>' + str(width) + '</width>\n')
    xml_file.write('        <height>' + str(height) + '</height>\n')
    xml_file.write('        <depth>3</depth>\n')
    xml_file.write('    </size>\n')

    # write the region of text on xml file
    for img_each_label in gt:
        spt = img_each_label.split()
        left = string.atoi(spt[0])
        top = string.atoi(spt[1])
        right = string.atoi(spt[2]) + left - 1
        bottom = string.atoi(spt[3]) + top - 1
        xml_file.write('    <object>\n')
        xml_file.write('        <name>text</name>\n')
        xml_file.write('        <pose>Unspecified</pose>\n')
        xml_file.write('        <truncated>0</truncated>\n')
        xml_file.write('        <difficult>0</difficult>\n')
        xml_file.write('        <bndbox>\n')
        xml_file.write('            <xmin>' + str(left) + '</xmin>\n')
        xml_file.write('            <ymin>' + str(top) + '</ymin>\n')
        xml_file.write('            <xmax>' + str(right) + '</xmax>\n')
        xml_file.write('            <ymax>' + str(bottom) + '</ymax>\n')
        xml_file.write('        </bndbox>\n')
        xml_file.write('    </object>\n')

    xml_file.write('</annotation>')
