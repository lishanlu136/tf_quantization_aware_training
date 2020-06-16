#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/2/13 17:49
@Author     : Li Shanlu
@File       : evaluate_acc_fake_quantized_eval_from_ckpt.py
@Software   : PyCharm
@Description: freeze eval pb and evaluate acc with gave pairs.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import sys
import argparse
import importlib
from utils import facenet, lfw, get_test_path
import math
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import freeze_graph


# my_data_dir下面是已经配好的pair
def get_paths(my_data_dir):
    path_list = []
    issame_list = []
    for pair in os.listdir(my_data_dir):
        if int(pair)>99999:
            continue
        img_pair = os.path.join(my_data_dir,pair)
        #imgs = []
        issame = True
        for img in os.listdir(img_pair):
            path = os.path.join(img_pair, img)
            #imgs.append(path)
            path_list.append(path)
        issame_list.append(issame)
    print("all of pairs:%d" %(len(path_list)))
    return path_list, issame_list


def create_inference_graph(network):
    image_input = tf.placeholder(dtype=tf.float32, shape=[None, 160, 160, 3], name="input")
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

    # eval from ckpt(fake_ckpt)
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
        input_saver_def = saver.as_saver_def()
        frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(), input_saver_def=input_saver_def,
            input_checkpoint=ckpt_model_path, output_node_names='embeddings', restore_op_name='',
            filename_tensor_name='', clear_devices=True, output_graph='', initializer_nodes='')

        binary_graph = save_pb_path

        with tf.gfile.GFile(binary_graph, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

        # evaluate test pairs acc
        paths, actual_issame = get_test_path.get_paths(os.path.expanduser(args.my_data_dir))
        path_all = {"paths": paths, "actual_issame": actual_issame, "type": "MY_DATA"}

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        image_size = args.image_size
        embedding_size = embeddings.get_shape()[1]
        paths = path_all["paths"]
        actual_issame = path_all["actual_issame"]
        batch_size = args.batch_size
        nrof_images = len(paths)
        nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
        emb_array = np.zeros((nrof_images, embedding_size))
        # ipdb.set_trace()
        for i in range(nrof_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, nrof_images)
            print("get embedding,idx start %d,idx end %d" %
                  (start_index, end_index))
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(
                paths_batch, False, False, image_size)
            feed_dict = {images_placeholder: images}
            emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)
        # ipdb.set_trace()
        tpr, fpr, accuracy, val, val_std, far, acc_dot, recall, fpr_dot, percision_dot, dot_product_all, fp_idxs, fn_idxs, recall_th, percision_th, acc_th = lfw.evaluate(emb_array, actual_issame)
        """
        embeddings1 = emb_array[0::2]
        embeddings2 = emb_array[1::2]
        dot_product_all = np.sum(embeddings1 * embeddings2, 1)
        f = open("/data1/lishanlu/raw_data/dot_score.txt","w")
        f.write(str(dot_product_all))
        f.close()
        """
        print('Accuracy: %1.3f+-%1.3f' %
              (np.mean(accuracy), np.std(accuracy)))
        print('tpr: %1.3f+-%1.3f' % (np.mean(tpr), np.std(tpr)))
        print('fpr: %1.3f+-%1.3f' % (np.mean(fpr), np.std(fpr)))
        print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' %
              (val, val_std, far))

        print("\nacc_dot:%1.3f,recall:%1.3f,fpr_dot:%1.3f,percision_dot:%1.3f" %
              (acc_dot, recall, fpr_dot, percision_dot))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt_model_path', type=str,
                        help='Load the wight from this ckpt model.')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='network.inception_v3')
    parser.add_argument('--gpu_idx', type=str,
                        help='gpu indexs', default='2')
    parser.add_argument('--save_pb_path', type=str,
                        help='freeze pb model save path', default='./freeze_model')
    parser.add_argument('--my_data_dir', type=str,
                        help='test data dir', default='./test_data')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=100)

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))