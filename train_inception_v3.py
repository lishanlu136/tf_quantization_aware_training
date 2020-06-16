#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2019/1/2 9:40
@Author     : Li Shanlu
@File       : train_inception_v3.py
@Software   : PyCharm
@Description: train fake quantized use inception_v3
              inception_v3能够量化训练成功
              1. fake quantized training, 得到ckpt model
              2. freeze eval model， 得到eval.pb
              3. eval.pb model convert to tflite model
              4. test tflite model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import os
import time
import sys
f_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f_path)
sys.path.append(f_path+"/..")
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
from utils import facenet, lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import graph_util

train_image_max = []
train_image_min = []


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
    network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    model_good_dir = os.path.join(os.path.expanduser(args.models_good_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    if not os.path.isdir(model_good_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_good_dir)
    # Write arguments to a text file
    facenet.write_arguments_to_file(
        args, os.path.join(log_dir, 'arguments.txt'))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = facenet.get_dataset(args.data_dir)

    if args.increased_data_dir:
        increased_set = facenet.get_dataset(args.increased_data_dir)
        train_set += increased_set
    if args.filter_filename:
        nrof_classes_before_filter = len(train_set)
        train_set = filter_dataset(train_set, os.path.expanduser(args.filter_filename),
                                   args.filter_percentile, args.filter_min_nrof_images_per_class)
        print(str(len(train_set) - nrof_classes_before_filter)+" classes filtered")

    nrof_classes = len(train_set)

    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)

    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(
            os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)

    train_graph = tf.Graph()
    with train_graph.as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list) > 0, 'The dataset should not be empty'

        # Create a queue that produces indices into the image_list and label_list
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]

        index_queue = tf.train.range_input_producer(range_size, num_epochs=None, shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')

        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

        image_paths_placeholder = tf.placeholder(tf.string, shape=(None, 1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None, 1), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=150000,
                                              dtypes=[tf.string, tf.int64],
                                              shapes=[(1,), (1,)],
                                              shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
                if args.random_brightness:
                    image = tf.image.random_brightness(image, max_delta=0.5)
                if args.random_contrast:
                    image = tf.image.random_contrast(image, 0.8, 1.2)
                if args.random_saturation:
                    image = tf.image.random_saturation(image, 0.3, 0.5)
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder,
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        image_max = tf.reduce_max(image_batch, name='image_max')  #统计最大值
        image_min = tf.reduce_min(image_batch, name='image_min')  #统计最小值

        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))

        print('Building training graph')

        # Build the inference graph
        # for inception_v3
        # Note that is_training be set to True or False, not be placeholder.
        prelogits, _ = network.inference(image_batch,
                                         num_classes=None,
                                         is_training=True,
                                         dropout_keep_prob=args.keep_probability,
                                         bottleneck_layer_size=args.embedding_size)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(args.weight_decay),
                                      scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if args.center_loss_factor > 0.0:
            prelogits_center_loss, _ = facenet.center_loss(
                prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
                                                   args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # print("regularization_losses:"+str(regularization_losses))
        total_loss = tf.add_n([cross_entropy_mean] +
                              regularization_losses, name='total_loss')

        # for fake quantized train, add quantized nodes automatically.
        train_graph = tf.get_default_graph()
        tf.contrib.quantize.create_training_graph(input_graph=train_graph, quant_delay=0)

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = facenet.train(total_loss, global_step, args.optimizer,
                                     learning_rate, args.moving_average_decay, tf.global_variables(),
                                     args.log_histograms, args.train_mode)

        # print all vars
        g_vars = tf.global_variables()
        print("++++++++++++++++++++")
        for g in g_vars:
            print(g.name)
        print("++++++++++++++++++++")

        if pretrained_model:
            # 指定加载某些变量的权重
            all_vars = tf.trainable_variables()
            #all_vars = tf.global_variables()
            # 跳过加载某些层的权重
            var_to_skip = [v for v in all_vars if v.name.startswith('Logits') or v.name.startswith('centers')]
            print("got pretrained_mode, var_to_skip:\n" + " \n".join([x.name for x in var_to_skip]))
            var_to_restore = [v for v in all_vars if not (v.name.startswith('Logits') or v.name.startswith('centers'))]
            """
            g_list = tf.global_variables()
            #bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
            #bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
            #act_quant_vars = [g for g in g_list if "act_quant" in g.name]
            #var_to_restore += act_quant_vars
            """
            saver = tf.train.Saver(all_vars, max_to_keep=20)
        else:
            var_list = tf.trainable_variables()
            #var_list = tf.global_variables()
            """
            g_list = tf.global_variables()
            bn_moving_vars = [g for g in g_list if "moving_mean" in g.name]
            bn_moving_vars += [g for g in g_list if "moving_variance" in g.name]
            print("++++++++++++++++++++")
            for g in bn_moving_vars:
                print(g.name)
            print("++++++++++++++++++++")
            var_list += bn_moving_vars
            """
            saver = tf.train.Saver(var_list, max_to_keep=20)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()
        train_global_init = tf.global_variables_initializer()
        train_local_init = tf.local_variables_initializer()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(train_global_init)
        sess.run(train_local_init)
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                saver.restore(sess, pretrained_model)
                """
                if args.lfw_dir:
                    print('evaluate on lfw with pretrained model: %s' %
                          pretrained_model)
                    evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                             batch_size_placeholder, embeddings, label_batch, lfw_paths, actual_issame,
                             args.lfw_batch_size, args.lfw_nrof_folds, log_dir, 0, summary_writer)
                """

            #tf.train.write_graph(sess.graph_def, model_dir, 'inception_v3_fake_quantized_train.pbtxt')
            saver1 = tf.train.Saver(tf.global_variables(), max_to_keep=20)
            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                print("step:", step)
                epoch = step // args.epoch_size
                #import ipdb; ipdb.set_trace()
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                      total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file, image_max, image_min)
                # Save variables and the metagraph if it doesn't exist already
                ckpt_model = save_variables_and_metagraph(sess, saver1, summary_writer, model_dir, subdir, step)
                print("ckpt_model name:", ckpt_model)
                """
                if args.lfw_dir:
                    acc_dot = evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder,
                                       batch_size_placeholder, embeddings, label_batch, lfw_paths, actual_issame,
                                       args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer)
                    if acc_dot > 0.97:
                        print("get a good model,save it!!!,step:%d,acc_dot:%.3f" %(step, acc_dot))
                        ckpt_model = save_variables_and_metagraph(sess, saver1, summary_writer, model_good_dir, subdir, step)
                """

    return model_dir


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename, 'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(
            distance_to_center, percentile)
        indices = np.where(distance_to_center >=
                           distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths) < min_nrof_images_per_class:
                removelist.append(label)
        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset


def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
          learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
          loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file, image_max, image_min):
    batch_number = 0
    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(
            learning_rate_schedule_file, epoch)
    epoch_start_t = time.time()
    print("train epoch: "+str(epoch)+" start ,learning_rate: "+str(lr))
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch), 1)
    image_paths_array = np.expand_dims(np.array(image_epoch), 1)
    sess.run(enqueue_op, {
             image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr,
                     #phase_train_placeholder: True,
                     batch_size_placeholder: args.batch_size}
        #import ipdb
        #ipdb.set_trace()
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str, image_max_, image_min_ = sess.run(
                [loss, train_op, global_step, regularization_losses, summary_op, image_max, image_min], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss, image_max_, image_min_ = sess.run(
                [loss, train_op, global_step, regularization_losses, image_max, image_min], feed_dict=feed_dict)
        train_image_max.append(image_max_)
        train_image_min.append(image_min_)
        duration = time.time() - start_time
        print('Epoch:[%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    print("train epoch:"+str(epoch)+" end,cost:"+str(time.time()-epoch_start_t))
    print("train image min: %1.3f" % min(train_image_min))
    print("train image max: %1.3f" % max(train_image_max))
    return step


def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder,
             embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer,  type="LFW"):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on %s images' % (type))

    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0, len(image_paths)), 1)
    image_paths_array = np.expand_dims(np.array(image_paths), 1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})

    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame)*2
    assert nrof_images % batch_size == 0, 'The number of %s images must be an integer multiple of the LFW batch size' % (
        type)
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder: False,
                     batch_size_placeholder: batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb

    assert np.array_equal(lab_array, np.arange(
        nrof_images)) == True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    tpr, fpr, accuracy, val, val_std, far, acc_dot, recall, fpr_dot, percision_dot, dot_product_all, fp_idxs, fn_idxs, recall_th, precision_th, acc_th = lfw.evaluate(
        emb_array, actual_issame, nrof_folds=nrof_folds)

    print('%s Accuracy: %1.3f+-%1.3f,tpr: %1.3f+-%1.3f,fpr: %1.3f+-%1.3f' % (type, np.mean(accuracy), np.std(accuracy),
                                                                             np.mean(tpr), np.std(tpr), np.mean(fpr), np.std(fpr)))
    """
    meaning: https://github.com/davidsandberg/facenet/issues/288
    If True Positive(TP) is defined as "two faces are validated as the same person, and they are indeed the same", False Negative(FN) is defined as "two faces are judged as different people, but the truth is that they are the same person"
    "Val@FAL=0.001" means the rate that faces are successfully accepted (TP/(TP+FN)) when the rate that faces are incorrectly accepted (FP/(TN+FP)) is 0.001.
    """
    print('%s Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' %
          (type, val, val_std, far))

    print("\nacc_dot:%1.3f,recall:%1.3f,percision_dot:%1.3f" %
          (acc_dot, recall, percision_dot))

    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='%s/accuracy' % type, simple_value=np.mean(accuracy))
    summary.value.add(tag='%s/val_rate' % type, simple_value=val)
    summary.value.add(tag='time/%s' % type, simple_value=lfw_time)

    summary.value.add(tag='%s/acc_dot' % type, simple_value=acc_dot)
    summary.value.add(tag='%s/recall' % type, simple_value=recall)
    summary.value.add(tag='%s/fpr_dot' % type, simple_value=fpr_dot)
    summary.value.add(tag='%s/percision_dot' %
                      type, simple_value=percision_dot)
    summary.value.add(tag='%s/recall_0.65' % type, simple_value=recall_th)
    summary.value.add(tag='%s/precision_0.65' % type, simple_value=precision_th)
    summary.value.add(tag='%s/acc_0.65' % type, simple_value=acc_th)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir, '%s_result.txt' % type), 'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    return acc_dot


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables',
                      simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph',
                      simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
    ckpt_model = checkpoint_path + '-' + str(step)
    return ckpt_model


def parse_arguments(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--logs_base_dir', type=str,
                        help='Directory where to write event logs.', default='train_results/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='train_results/models/facenet')
    parser.add_argument('--models_good_dir', type=str,
                        help='Directory where to write trained models and checkpoints.', default='train_results/models/facenet_good')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
                        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
                        help='Path to the data directory containing aligned face patches.',
                        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
                        help='Model definition. Points to a module containing the definition of the inference graph.', default='network.inception_v3')
    parser.add_argument('--max_nrof_epochs', type=int,
                        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
                        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
                        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop',
                        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
                        'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip',
                        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate',
                        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--random_saturation',
                        help='Performs random_saturation of training images.', action='store_true', default=False)
    parser.add_argument('--random_contrast',
                        help='Performs random_contrast of training images.', action='store_true', default=False)
    parser.add_argument('--random_brightness',
                        help='Performs random_brightness of training images.', action='store_true', default=False)
    parser.add_argument('--keep_probability', type=float,
                        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
                        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
                        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
                        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
                        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
                        help='Initial learning rate. If set to a negative value a learning rate ' +
                        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
                        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
                        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
                        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
                        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
                        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
                        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
                        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
                        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
                        help='Keep only the classes with this number of examples or more', default=0)

    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
                        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
                        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
                        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
                        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
                        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--train_mode',
                        help='train_mode', type=int, default=0)
    parser.add_argument('--gpu_idx', type=str,
                        help='gpu indexs', default='0')
    parser.add_argument('--increased_data_dir', type=str,
                        help='Path to the another one data directory containing aligned face patches.', default='')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))