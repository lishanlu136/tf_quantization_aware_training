#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/12/17 16:01
@Author     : Li Shanlu
@File       : test_tflite.py
@Software   : PyCharm
@Description: 测试tflite模型
"""
import os
import sys
f_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f_path)
sys.path.append(f_path+"/..")
import numpy as np
import tensorflow as tf
import scipy
from utils import facenet
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="train_results/freeze_models/tflite_model/inception_v3/inception_v3_converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image_origin = scipy.misc.imread("../../dataset/test/151050622675_0_80.895_0.png", mode='RGB')
image_w = facenet.prewhiten(image_origin)
image_q = (image_w + 7.288) * 10.006671114
image_ = np.array([image_q.astype('uint8')])

print(image_.shape)
print(type(image_))
print(input_details)
print(output_details)

interpreter.set_tensor(input_details[0]['index'], image_)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[1]['index'])
print(output_data)
print(output_data.shape, type(output_data))

def f(x):
    return 0.0078125*(x-128)

output_data_new = map(f, output_data[0])
#embeddings_after = output_data_new/np.linalg.norm(output_data_new)
print("convert after:")
print(output_data_new)
print(np.linalg.norm(output_data_new))


with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        facenet.load_model("train_results/freeze_models/tflite_model/inception_v3/inception_v3_eval_model.pb")
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        images = np.zeros((1, 160, 160, 3))
        images[0,:,:,:] = image_w
        embeddings_ = sess.run(embeddings, feed_dict={images_placeholder: images})
        print("converted before:")
        print(embeddings_[0])
        print(np.linalg.norm(embeddings_[0]))
        similar = np.sum(output_data_new*embeddings_[0])
        print("similar: %1.3f" % similar)

