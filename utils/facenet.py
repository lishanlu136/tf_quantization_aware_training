# -*- coding: utf-8 -*-
"""Functions for building the face recognition network.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from subprocess import Popen, PIPE
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
from scipy import misc
from sklearn.model_selection import KFold
from scipy import interpolate
from tensorflow.python.training import training
import random
import re
from tensorflow.python.platform import gfile
from six import iteritems
import math


def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper

    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.

    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss


def decov_loss(xs):
    """Decov loss as described in https://arxiv.org/pdf/1511.06068.pdf
    'Reducing Overfitting In Deep Networks by Decorrelating Representation'
    """
    x = tf.reshape(xs, [int(xs.get_shape()[0]), -1])
    m = tf.reduce_mean(x, 0, True)
    z = tf.expand_dims(x-m, 2)
    corr = tf.reduce_mean(tf.matmul(z, tf.transpose(z, perm=[0, 2, 1])), 0)
    corr_frob_sqr = tf.reduce_sum(tf.square(corr))
    corr_diag_sqr = tf.reduce_sum(tf.square(tf.diag_part(corr)))
    loss = 0.5*(corr_frob_sqr - corr_diag_sqr)
    return loss


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def shuffle_examples(image_paths, labels):
    shuffle_list = list(zip(image_paths, labels))
    random.shuffle(shuffle_list)
    image_paths_shuff, labels_shuff = zip(*shuffle_list)
    return image_paths_shuff, labels_shuff


def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_image(file_contents, channels=3)
    return example, label


def random_rotate_image(image):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic')


def read_and_augment_data(image_list, label_list, image_size, batch_size, max_nrof_epochs,
                          random_crop, random_flip, random_rotate, nrof_preprocess_threads, shuffle=True):

    images = ops.convert_to_tensor(image_list, dtype=tf.string)
    labels = ops.convert_to_tensor(label_list, dtype=tf.int32)

    # Makes an input queue
    input_queue = tf.train.slice_input_producer([images, labels],
                                                num_epochs=max_nrof_epochs, shuffle=shuffle)

    images_and_labels = []
    for _ in range(nrof_preprocess_threads):
        image, label = read_images_from_disk(input_queue)
        if random_rotate:
            image = tf.py_func(random_rotate_image, [image], tf.uint8)
        if random_crop:
            image = tf.random_crop(image, [image_size, image_size, 3])
        else:
            image = tf.image.resize_image_with_crop_or_pad(
                image, image_size, image_size)
        if random_flip:
            image = tf.image.random_flip_left_right(image)
        # pylint: disable=no-member
        image.set_shape((image_size, image_size, 3))
        image = tf.image.per_image_standardization(image)
        images_and_labels.append([image, label])

    image_batch, label_batch = tf.train.batch_join(
        images_and_labels, batch_size=batch_size,
        capacity=4 * nrof_preprocess_threads * batch_size,
        allow_smaller_final_batch=True)

    return image_batch, label_batch


def _add_loss_summaries(total_loss):
    """Add summaries for losses.

    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.

    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name + ' (raw)', l)
        tf.summary.scalar(l.op.name, loss_averages.average(l))

    return loss_averages_op


def train(total_loss, global_step, optimizer, learning_rate, moving_average_decay, update_gradient_vars, log_histograms=True, train_mode=0):
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        if optimizer == 'ADAGRAD':
            opt = tf.train.AdagradOptimizer(learning_rate)
        elif optimizer == 'ADADELTA':
            opt = tf.train.AdadeltaOptimizer(
                learning_rate, rho=0.9, epsilon=1e-6)
        elif optimizer == 'ADAM':
            opt = tf.train.AdamOptimizer(
                learning_rate, beta1=0.9, beta2=0.999, epsilon=0.1)
        elif optimizer == 'RMSPROP':
            opt = tf.train.RMSPropOptimizer(
                learning_rate, decay=0.9, momentum=0.9, epsilon=1.0)
        elif optimizer == 'MOM':
            opt = tf.train.MomentumOptimizer(
                learning_rate, 0.9, use_nesterov=True)
        else:
            raise ValueError('Invalid optimization algorithm')

        grads = opt.compute_gradients(total_loss, update_gradient_vars)

        # fine tune
        if train_mode == 1:  # 按变量衰减学习率
            print("train_mode==1，learning_rate_decrease by variable")
            idx = 0
            grads_decayed = []
            active_grads = [grad for grad in grads if grad[0] is not None]
            for i in range(len(grads)):
                grad = grads[i]
                if grad[0] is None:
                    grads_decayed.append(grad)
                    print(" grad for "+str(grad[1])+" is None")
                    continue
                else:
                    grads_decayed.append(
                        (tf.multiply(
                            grad[0], 1.0/math.exp((len(active_grads)-idx)*0.01)), grad[1]))
                    print(" grad for "+str(grad[1])+" multiply "+str(1.0 / math.exp((len(active_grads)-idx)*0.01)))
                    idx += 1
            grads = grads_decayed
        elif train_mode == 11:  # 按层衰减学习率
            #re_str = r"(.*Conv2d_.*?/)|(.*Bottleneck/)|(Logits/)|(centers)"
            #re_str = r"(.*Conv2d_.*?/)|(.*Bottleneck/)|(Logits/)|(arcface_loss/)"
            re_str = r"(InceptionResnetV1/)|(arcface_loss/)"
            #re_str = r"(.*Conv2d_.*?/)|(.*Conv_.*?/)|(.*PreBottleneck_.*?/)|(Logits/)|(centers)"
            #re_str = r"(.*Conv2d_.*?/)|(.*prelu/)|(.*Bottleneck/)|(.*BatchNorm/)|(Logits/)|(arcface_loss/)"
            #re_str = r"(.*/Conv/.*)|(.*/expand/.*)|(.*/depthwise/.*)|(.*/project/.*)|(.*/Conv_1/.*)|(.*/Logits/.*)|(centers)"
            #re_str = r"(.*/Conv/.*)|(.*/expand/.*)|(.*/depthwise/.*)|(.*/project/.*)|(.*/Conv_1/.*)|(.*/Logits/.*)|(Logits/)|(Bottleneck/)|(centers)"
            #re_str = r"(.*Conv.*?/)|(.*prelu_alphas)|(.*BatchNorm/)|(Logits/)|(arcface_loss/)"
            # import ipdb
            # ipdb.set_trace()
            print("train_mode==11, learning_rate_decrease by layer")
            layer_idx = 0
            grads_decayed = []
            active_layers = set([re.search(re_str, grad[1].name).group()
                                 for grad in grads if grad[0] is not None])
            # print("active_layers:"+str(active_layers))
            print("active layers:")
            for layer in active_layers:
                print(layer)
            last_layer = ""
            for grad in grads:
                if grad[0] is None:
                    grads_decayed.append(grad)
                    print(" grad for "+str(grad[1])+" is None")
                    continue
                else:
                    cur_layer = re.search(re_str, grad[1].name).group()
                    if cur_layer != last_layer:
                        layer_idx += 1
                        last_layer = cur_layer
                    decay = 1.0/math.exp((len(active_layers)-layer_idx)*0.01)
                    grads_decayed.append(
                        (tf.multiply(
                            grad[0], decay), grad[1]))
                    print("cur_layer:"+cur_layer+" grad for "+str(grad[1].name)+" "+str(grad[0])+" multiply "+str(decay))
            grads = grads_decayed
        elif train_mode == 2 or train_mode == 3:  # 只训练顶层
            if train_mode == 2:
                train_layers_re = r"(Block8)|(Bottleneck)|(centers:0)"

            if train_mode == 3:
                train_layers_re = r"(Bottleneck)|(centers:0)|(Logits)"
            print("train_mode==%d train_layers_re: %s" %
                  (train_mode, str(train_layers_re)))
            grads_top = []
            for grad in grads:
                if re.search(train_layers_re, grad[1].name) is not None:
                    print(str(grad[1].name)+" match")
                    grads_top.append(grad)
                else:
                    print(str(grad[1].name)+" not match!")
            grads = grads_top
            # grads = opt.compute_gradients(total_loss, var_list)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    if log_histograms:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    if log_histograms:
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op


def learning_rate_decay_by_layer(lenth):
    return [1.0/math.exp((lenth-i)*0.01) for i in range(lenth)]


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1]//2)
        sz2 = int(image_size//2)
        if random_crop:
            diff = sz1-sz2
            (h, v) = (np.random.randint(-diff, diff+1),
                      np.random.randint(-diff, diff+1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1-sz2+v):(sz1+sz2+v), (sz1-sz2+h):(sz1+sz2+h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


def load_data(image_paths, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    for i in range(nrof_samples):
        img = misc.imread(image_paths[i])
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i, :, :, :] = img
    return images


def get_label_batch(label_data, batch_size, batch_index):
    nrof_examples = np.size(label_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size <= nrof_examples:
        batch = label_data[j:j+batch_size]
    else:
        x1 = label_data[j:nrof_examples]
        x2 = label_data[0:nrof_examples-j]
        batch = np.vstack([x1, x2])
    batch_int = batch.astype(np.int64)
    return batch_int


def get_batch(image_data, batch_size, batch_index):
    nrof_examples = np.size(image_data, 0)
    j = batch_index*batch_size % nrof_examples
    if j+batch_size <= nrof_examples:
        batch = image_data[j:j+batch_size, :, :, :]
    else:
        x1 = image_data[j:nrof_examples, :, :, :]
        x2 = image_data[0:nrof_examples-j, :, :, :]
        batch = np.vstack([x1, x2])
    batch_float = batch.astype(np.float32)
    return batch_float


def get_triplet_batch(triplets, batch_index, batch_size):
    ax, px, nx = triplets
    a = get_batch(ax, int(batch_size/3), batch_index)
    p = get_batch(px, int(batch_size/3), batch_index)
    n = get_batch(nx, int(batch_size/3), batch_index)
    batch = np.vstack([a, p, n])
    return batch


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate
        raise Exception(
            "ERROR:learning_rate for epoch  %d in file %s is not set" % (epoch, filename))


class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = os.listdir(path_exp)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset


def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir, img) for img in images]
    return image_paths


def split_dataset(dataset, split_ratio, mode):
    if mode == 'SPLIT_CLASSES':
        nrof_classes = len(dataset)
        class_indices = np.arange(nrof_classes)
        np.random.shuffle(class_indices)
        split = int(round(nrof_classes*split_ratio))
        train_set = [dataset[i] for i in class_indices[0:split]]
        test_set = [dataset[i] for i in class_indices[split:-1]]
    elif mode == 'SPLIT_IMAGES':
        train_set = []
        test_set = []
        min_nrof_images = 2
        for cls in dataset:
            paths = cls.image_paths
            np.random.shuffle(paths)
            split = int(round(len(paths)*split_ratio))
            if split < min_nrof_images:
                continue  # Not enough images for test set. Skip class...
            train_set.append(ImageClass(cls.name, paths[0:split]))
            test_set.append(ImageClass(cls.name, paths[split:-1]))
    else:
        raise ValueError('Invalid train/test split mode "%s"' % mode)
    return train_set, test_set


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(),
                      os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError(
            'No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError(
            'There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)
    # code.interact("test roc1", local=locals())
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            t, f, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])

        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tpr, fpr, acc = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
            tprs[fold_idx, threshold_idx] = tpr
            fprs[fold_idx, threshold_idx] = fpr
            print("threshold_idx:%d,threhold:%1.3f,fold_idx:%d,tpr:%1.3f,fpr:%1.3f,acc:%1.3f" %
                  (threshold_idx, threshold, fold_idx,  tpr, fpr, acc))
        t, f, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
        print("[best_threshold_index-%d]:threhold:%1.3f,fold_idx:%d,tpr:%1.3f,fpr:%1.3f,acc:%1.3f" %
              (best_threshold_index, thresholds[best_threshold_index], fold_idx,  t, f, accuracy[fold_idx]))

    # code.interact("test roc2", local=locals())
    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    # print("true_accept", true_accept, "false_accept", false_accept,"n_same", n_same, "n_diff", n_diff)
    if n_diff == 0 or n_same == 0:
        print("threshold:%.2f error?: n_diff==0 or n_same==0 when calculate_val_far" % threshold)
        return 0, 0
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def calculate_acc_dot_product(thresholds, embeddings1, embeddings2, actual_issame):
    """
    根据线上逻辑 根据余弦相似度 计算准确率和召回率
    """
    # code.interact("test", local=locals())
    accs, recalls, fprs, precisions = np.zeros(len(thresholds)), np.zeros(
        len(thresholds)), np.zeros(len(thresholds)), np.zeros(len(thresholds))

    dot_res_same = np.sum(
        embeddings1[actual_issame] * embeddings2[actual_issame], 1)
    dot_res_nsame = np.sum(
        embeddings1[~actual_issame] * embeddings2[~actual_issame], 1)
    
    for idx, threshold in enumerate(thresholds):
        tp = np.sum(dot_res_same >= threshold)
        fn = np.sum(actual_issame) - tp  # 事实同一人 预测为不同人

        tn = np.sum(dot_res_nsame < threshold)
        fp = np.sum(~actual_issame) - tn  # 事实不同人 预测为同一人

        accs[idx] = acc = (tp + tn)*1.0 / (tp+fp+tn+fn)
        recalls[idx] = recall = tp * 1.0/(tp+fn)  # 同一个人的人脸被判定正确的比率
        fprs[idx] = fpr = fp*1.0/(fp+tn)  # 不是同一个人被判定为同一人的比率
        precisions[idx] = precision = tp*1.0/(tp+fp+0.000001)

        # code.interact("test calculate_acc_dot_product", local=locals())
        print("[calculate_acc_dot_product-%d]:threshold:%1.3f,recall:%1.3f,precision:%1.3f,fpr:%1.3f,acc:%1.3f"
              % (idx, threshold, recall, precision, fpr, acc))
    max_acc_idx = np.argmax(accs)
    print("[best_threshold_index:%d],threshold:%1.3f,acc:%1.3f,recall:%1.3f,precision:%1.3f,fpr:%1.3f" %
          (max_acc_idx, thresholds[max_acc_idx], accs[max_acc_idx], recalls[max_acc_idx], precisions[max_acc_idx], fprs[max_acc_idx]))
    print("[threshold== %1.3f], acc:%1.3f,recall:%1.3f,precision:%1.3f" %
          (thresholds[650], accs[650], recalls[650], precisions[650]))
    # 返回准确率最大时预测错样本的idx
    best_threshold = thresholds[max_acc_idx]
    # ipdb.set_trace()
    dot_product_all = np.sum(embeddings1 * embeddings2, 1)
    same_idxs = np.where(actual_issame)
    n_same_idxs = np.where(~actual_issame)
    bigger_idxs = np.where(dot_product_all >= best_threshold) 
    less_idxs = np.where(dot_product_all < best_threshold)
    fp_idxs = np.intersect1d(bigger_idxs[0], n_same_idxs[0])  # fp 不同人 大于阈值的索引
    fn_idxs = np.intersect1d(less_idxs[0], same_idxs[0])  # fn 同一个人 小于阈值的索引
    return thresholds[max_acc_idx], accs[max_acc_idx], recalls[max_acc_idx], fprs[max_acc_idx], precisions[max_acc_idx],\
           dot_product_all, fp_idxs, fn_idxs, recalls[650], precisions[650], accs[650]


def store_revision_info(src_path, output_dir, arg_string):
    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    rev_info_filename = os.path.join(output_dir, 'revision_info.txt')
    with open(rev_info_filename, "w") as text_file:
        text_file.write('arguments: %s\n--------------------\n' % arg_string)
        text_file.write('tensorflow version: %s\n--------------------\n' %
                        tf.__version__)  # @UndefinedVariable
        text_file.write('git hash: %s\n--------------------\n' % git_hash)
        text_file.write('%s' % git_diff)


def list_variables(filename):
    reader = training.NewCheckpointReader(filename)
    variable_map = reader.get_variable_to_shape_map()
    names = sorted(variable_map.keys())
    return names


def put_images_on_grid(images, shape=(16, 8)):
    nrof_images = images.shape[0]
    img_size = images.shape[1]
    bw = 3
    img = np.zeros((shape[1]*(img_size+bw)+bw, shape[0]
                    * (img_size+bw)+bw, 3), np.float32)
    for i in range(shape[1]):
        x_start = i*(img_size+bw)+bw
        for j in range(shape[0]):
            img_index = i*shape[0]+j
            if img_index >= nrof_images:
                break
            y_start = j*(img_size+bw)+bw
            img[x_start:x_start+img_size, y_start:y_start +
                img_size, :] = images[img_index, :, :, :]
        if img_index >= nrof_images:
            break
    return img


def write_arguments_to_file(args, filename):
    with open(filename, 'w') as f:
        for key, value in iteritems(vars(args)):
            f.write('%s: %s\n' % (key, str(value)))
