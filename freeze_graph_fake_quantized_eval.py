#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2018/12/26 17:09
@Author     : Li Shanlu
@File       : freeze_graph_fake_quantized_eval.py
@Software   : PyCharm
@Description: freeze graph for fake quantized eval inference
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import sys
import argparse
import importlib
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph


def create_inference_graph(network):
    image_input = tf.placeholder(dtype=tf.float32, shape=[1, 160, 160, 3], name="input")
    keep_probability = 1.0
    phase_train = False
    embedding_size = 128
    weight_decay = 0.0

    # for inception_v3
    prelogits, _ = network.inference(image_input,
                                     num_classes=None,
                                     is_training=phase_train,
                                     dropout_keep_prob=keep_probability,
                                     bottleneck_layer_size=embedding_size,
                                     weight_decay=weight_decay)
    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    network = importlib.import_module(args.model_def)
    ckpt_model_path = args.ckpt_model_path
    save_pb_path = args.save_pb_path

    create_inference_graph(network)
    g = tf.get_default_graph()
    tf.contrib.quantize.create_eval_graph(input_graph=g)
    all_vars = tf.global_variables()
    print("eval global vars: \n +++++++++++++++++++++\n")
    for var in all_vars:
        print(var.name)
    print("++++++++++++++++++++++++++")
    saver = tf.train.Saver(tf.global_variables())
    sess = tf.Session()
    with sess.as_default():
        saver.restore(sess, ckpt_model_path)
        """
        # Turn all the variables into inline constants inside the graph and save it.
        frozen_graph_def = graph_util.convert_variables_to_constants(
            sess, sess.graph_def, ['embeddings'])
        """
        input_saver_def = saver.as_saver_def()

        frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(), input_saver_def=input_saver_def,
            input_checkpoint=ckpt_model_path, output_node_names='embeddings', restore_op_name='',
            filename_tensor_name='', clear_devices=True, output_graph='', initializer_nodes='')

        binary_graph = save_pb_path
    
        with tf.gfile.GFile(binary_graph, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

        """
        print("frozen graph def:")
        for node in frozen_graph_def.node:
            print(node)
        
        tf.train.write_graph(
            frozen_graph_def,
            os.path.dirname(save_pb_path),
            os.path.basename(save_pb_path),
            as_text=False)
        """


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_model_path', type=str,
                        help='Load the wight from this ckpt model.')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='network.inception_v3')
    parser.add_argument('--gpu_idx', type=str,
                        help='gpu indexs', default='0')
    parser.add_argument('--save_pb_path', type=str,
                        help='freeze pb model save path', default='./freeze_model')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
