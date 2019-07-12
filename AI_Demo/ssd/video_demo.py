#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : video_demo.py
# @Author: Zhang Chuan
# @Date  : 19-7-12
# @Desc  : capture you camera and detect using ssd
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
from img_demo import get_classes
from ssd_300_model import SSD_300_MODEL


def detect_video(video, all_classes, ssd):
    """Use ssd to detect video.
    # Argument:
        video: video file.
        all_classes: all classes name.
    """
    video_path = os.path.join("videos", "test", video)
    if (os.path.exists(video_path) and video != ''):
        camera = cv2.VideoCapture(video_path)
    else:
        camera = cv2.VideoCapture(0)
        video = 'your_camera.mp4'
    res, frame = camera.read()
    if not res:
        print("file open failed and camera can not open")
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()

        if not res:
            break

        rclasses, rscores, rbboxes =  ssd._process_image(frame)
        image = visualization.bboxes_draw_on_img(frame, rclasses, rscores, rbboxes, visualization.colors_plasma, all_classes)
        cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:       #press ESC to quit
                break

    vout.release()
    camera.release()


if __name__ == '__main__':
    classes_file = '../dataset//VOC_2007/voc2007_classes.txt'
    all_classes = get_classes(classes_file)

    # Define the SSD model
    ssd = SSD_300_MODEL(0.5, 0.45, (300,300))
    # detect videos one at a time in videos/test folder
    #video = 'AI.mp4'
    video = ''
    detect_video(video, all_classes, ssd)