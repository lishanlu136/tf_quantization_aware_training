#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/2/13 13:48
@Author     : Li Shanlu
@File       : tflite_convert.py
@Software   : PyCharm
@Description:
"""
import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
graph_def_file = "train_results/freeze_models/inception_v3_eval_model.pb"
input_arrays = ["input"]
output_arrays = ["InceptionV3/PreLogits/SpatialSqueeze","embeddings"]
converter = tf.contrib.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
converter.inference_type = tf.contrib.lite.constants.QUANTIZED_UINT8
converter.quantized_input_stats = {input_arrays[0]: (73.0, 10.00667)}
#converter.default_ranges_stats = (-1.0, 10.0)
tflite_model = converter.convert()
open("train_results/freeze_models/inception_v3_converted_model.tflite", "wb").write(tflite_model)