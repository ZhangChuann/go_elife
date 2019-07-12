#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : img_demo.py
# @Author: Zhang Chuan
# @Date  : 19-7-5
# @Desc  : input img_path, return and show the result
import os
import getopt
import math
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import sys
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

from ssd_300_model import SSD_300_MODEL

def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

# Test on some demo image and visualize output.
if __name__ == '__main__':
    # TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
    img_name = ''
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o", "ifile=")
    except getopt.GetoptError:
        print('img_demp.py -i <img_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('img_demp.py -i <img_name>')
            sys.exit()
        elif opt == '-i':
            img_name = arg
    print('file_name:' + img_name)
    classes_file = '../dataset//VOC_2007/voc2007_classes.txt'
    all_classes = get_classes(classes_file)
    sys.path.append('./')

    ssd = SSD_300_MODEL(0.5, 0.45, (300,300))
    img_filepath, img_filename = os.path.split(img_name)
    img_real_name, img_ext = os.path.splitext(img_filename)
    img = mpimg.imread(img_name)
    rclasses, rscores, rbboxes =  ssd._process_image(img)
    #visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
    res_img = visualization.bboxes_draw_on_img(img, rclasses, rscores, rbboxes, visualization.colors_plasma, all_classes)
    cv2.namedWindow('ssd_detect', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('ssd_detect', res_img)
    cv2.waitKey(0)
    cv2.imwrite(img_filepath + '/' + img_real_name + '_ssd_res' + img_ext, res_img)


