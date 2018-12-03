#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : mnist_cnn_predict.py
# @Author: Zhang Chuan
# @Date  : 18-8-23
# @Desc  : predict image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from tool import camera_capture


#model path
model_path = './mnist_cnn_saved_model/1534953492'
signature_key = 'classify'
input_key = 'image'
output_key = 'classes'

def mnist_predict(file_path, show = False):

    if file_path is '':
        testImage = camera_capture.camera_capture_img()
    else:
        testImage = cv.imread(file_path)

    #print(testImage.shape)
    #testImage = cv.resize(testImage,(32,32),interpolation=cv.INTER_CUBIC)

    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, ['serve'], model_path)
        signature = meta_graph_def.signature_def
        x_tensor_name =  signature[signature_key].inputs[input_key].name
        y_tensor_name =  signature[signature_key].outputs[output_key].name

        #获取tensor
        x = sess.graph.get_tensor_by_name(x_tensor_name)
        y = sess.graph.get_tensor_by_name(y_tensor_name)
        #对图片进行测试
        cv.imwrite("temp0.jpg", testImage)
        testImage=cv.cvtColor(testImage, cv.COLOR_RGB2GRAY)
        cv.imwrite("temp1.jpg", testImage)
        testImage=cv.resize(testImage,dsize=(28, 28))
        cv.imwrite("temp2.jpg", testImage)
        test_input=np.array(testImage)
        test_input = test_input.reshape(1, 28,  28)
        #图像反色处理, when my input img is test_img/example*.png
        test_input = 255 - test_input
        pre_num = sess.run(y, feed_dict={x: test_input})#利用训练好的模型预测结果
        if show:
            print('模型预测结果为：', pre_num)
            # cv.imshow("image",testImage)
            # cv.waitKey(0)
            #显示测试的图片
            fig = plt.figure(), plt.imshow(testImage,cmap='binary')  # 显示图片
            plt.title("prediction result:"+str(pre_num))
            plt.show()
        return pre_num


if  __name__ == '__main__':
    file_path = 'test_img/example3.png'
    pre_num = mnist_predict(file_path,True)
    #pre_num = mnist_predict(file_path='',show=True)
