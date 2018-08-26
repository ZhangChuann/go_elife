#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : save_pic.py
# @Author: Zhang Chuan
# @Date  : 18-8-12
# @Desc  : dataset to pic
from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

WORK_DIRECTOTY = "../dataset/mnist"
SAVE_DIRECTOTY = "../dataset/mnist/raw/"
mnist = input_data.read_data_sets(WORK_DIRECTOTY, one_hot=True)

if os.path.exists(SAVE_DIRECTOTY) is False:
    os.makedirs(SAVE_DIRECTOTY)

# 保存前20张图片
for i in range(20):
    # 请注意，mnist.train.images[i, :]就表示第i张图片（序号从0开始）
    image_array = mnist.train.images[i, :]
    # TensorFlow中的MNIST图片是一个784维的向量，我们重新把它还原为28x28维的图像。
    image_array = image_array.reshape(28, 28)
    # 保存文件的格式为 mnist_train_0.jpg, mnist_train_1.jpg, ... ,mnist_train_19.jpg
    filename = SAVE_DIRECTOTY + 'mnist_train_%d.jpg' % i
    # 将image_array保存为图片
    # 先用scipy.misc.toimage转换为图像，再调用save直接保存。
    scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print('Please check: %s ' % SAVE_DIRECTOTY)

