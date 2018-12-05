#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : camera_capture.py
# @Author: Zhang Chuan
# @Date  : 18-10-18
# @Desc  : capture img & video from camera

import cv2

def camera_capture_img():
    cap = cv2.VideoCapture(0)
    while(1):
        ret, frame = cap.read()
        #show a frame
        cv2.imshow("capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame


if __name__ == '__main__':
    img = camera_capture_img()

