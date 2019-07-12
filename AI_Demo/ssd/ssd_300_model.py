#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : ssd_300_model.py
# @Author: Zhang Chuan
# @Date  : 19-7-13
# @Desc  : SSD 300 VGG Model
import os
import getopt
import math
import tensorflow as tf
import cv2

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

class SSD_300_MODEL:
    def __init__(self, select_threshold = 0.5, nms_threshold =  0.45, net_shape = (300,300)):
        self.select_threshold = select_threshold
        self.nms_threshold = nms_threshold
        self.net_shape = net_shape

        slim = tf.contrib.slim
        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        self.isess = tf.InteractiveSession(config=config)

        # Input placeholder
        net_shape = (300, 300)
        data_format = 'NHWC'
        self.img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        self.image_pre, self.labels_pre, self.bboxes_pre, self.bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
            self.img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
        self.image_4d = tf.expand_dims(self.image_pre, 0)

        # Define the SSD model
        reuse = True if 'ssd_net' in locals() else None
        ssd_net = ssd_vgg_300.SSDNet()
        with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
            self.predictions, self.localisations, _, _ = ssd_net.net(self.image_4d, is_training=False, reuse=reuse)

        # Restore SSD Model
        ckpt_filename = 'checkpoints/ssd_300_vgg.ckpt'

        self.isess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.isess, ckpt_filename)
        self.ssd_anchors = ssd_net.anchors(net_shape)

    def _process_image(self, img):
        '''
        Main image processing routine
        :param img:
        :param select_threshold:
        :param nms_threshold:
        :param net_shape:
        :return: rclasses, rscores, rbboxes
        '''
        # Run SSD network.
        print(self.nms_threshold)
        rimg, rpredictions, rlocalisations, rbbox_img = self.isess.run([self.image_4d, self.predictions, self.localisations, self.bbox_img],
                                                                  feed_dict={self.img_input: img})

        # Get classes and bboxes from the net outputs.
        rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, self.ssd_anchors,
            select_threshold=self.select_threshold, img_shape=self.net_shape, num_classes=21, decode=True)

        rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
        rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
        rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=self.nms_threshold)
        # Resize bboxes to original image shape. Note: useless for Resize.WARP!
        rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
        return rclasses, rscores, rbboxes



