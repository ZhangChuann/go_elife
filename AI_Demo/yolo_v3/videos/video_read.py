#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : video_read.py.py
# @Author: Zhang Chuan
# @Date  : 18-12-5
# @Desc  : read video tool

import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

def video_read(file_path):
    if(os.path.exists(file_path)):
        cv2_video = cv2.VideoCapture(file_path)
    else:
        cv2_video = cv2.VideoCapture(0)
    i=0
    print(cv2_video.isOpened())
    # 获得码率及尺寸

    while True:
        res, frame = cv2_video.read()
        if not res:
            break
        cv2.imshow("detection", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    cv2_video.release()

if __name__ == '__main__':
    #video_read("0")
    video_read("../../dataset/video/AI.mp4")
