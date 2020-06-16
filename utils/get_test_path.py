#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 20-6-16 下午4:57
@Author     : lishanlu
@File       : get_test_path.py
@Software   : PyCharm
@Description:
"""

from __future__ import division, print_function, absolute_import
import os
import numpy as np
import random


def get_paths(unit_data_dir, batch_size=100, num=1):
    path_list = []
    issame_list = []
    dir_ary = unit_data_dir.split(",")
    imgs_all_correct = []
    imgs_all_error = []
    for unit_dir in dir_ary:
        print("got dir :"+unit_dir)
        for dir in os.listdir(unit_dir):
            if dir == "correct":
                img_pairs = get_sub_img(os.path.join(unit_dir, dir))
                print("imgs in correct dir:" +
                      os.path.join(unit_dir, dir)+" len:"+str(len(img_pairs)))
                imgs_all_correct += img_pairs
            elif dir == "error":
                img_pairs = get_sub_img(os.path.join(unit_dir, dir))
                print("imgs in error dir:" +
                      os.path.join(unit_dir, dir)+" len:"+str(len(img_pairs)))
                imgs_all_error += img_pairs
            else:
                # os.rmdir(os.path.join(unit_dir, dir))
                print("remove unit image path:"+os.path.join(unit_dir, dir))

    # 取相同样本数
    min_list_len = min(len(imgs_all_correct), len(imgs_all_error))
    print("imgs_all_correct:"+str(len(imgs_all_correct))+" imgs_all_error:"+
          str(len(imgs_all_error))+" min_list_len:"+str(min_list_len))
    # 在取样本之前打乱correct和error的pair顺序
    # correct
    correct_idx = range(int(len(imgs_all_correct)/2))
    random.shuffle(correct_idx, random.seed(20))
    shuffled_correct = []
    for x in correct_idx:
        shuffled_correct += [imgs_all_correct[x * 2], imgs_all_correct[x * 2 + 1]]
    # error
    error_idx = range(int(len(imgs_all_error)/2))
    random.shuffle(error_idx, random.seed(30))
    shuffled_error = []
    for x in error_idx:
        shuffled_error += [imgs_all_error[x * 2], imgs_all_error[x * 2 + 1]]
    actual_len = int(min_list_len/(batch_size*num)) * (batch_size*num)  # batch 的倍数
    print("actual batch len:"+str(actual_len))
    path_list += shuffled_correct[:actual_len]
    issame_list += [True]*(int(actual_len/2))
    path_list += shuffled_error[:actual_len]
    issame_list += [False]*(int(actual_len/2))
    #shuffle good和bad混在一起
    #shuffled_idx = np.random.permutation(len(issame_list))
    shuffled_idx = range(len(issame_list))
    random.shuffle(shuffled_idx, random.seed(10))
    path_list_final = []
    for x in shuffled_idx:
        path_list_final += [path_list[x*2], path_list[x*2+1]]
    print("path list:", len(path_list_final), "issame_list:", len(issame_list))
    return path_list_final, np.array(issame_list)[shuffled_idx]


def get_sub_img(cur_dir):
    imgs_all = []
    for img_dir in os.listdir(cur_dir):
        imgs = []
        if os.path.isfile(os.path.join(cur_dir, img_dir)):
            print("skip file:"+os.path.join(cur_dir, img_dir))
            continue
        for img in os.listdir(os.path.join(cur_dir, img_dir)):
            imgs.append(os.path.join(cur_dir, img_dir, img))
        if len(imgs) == 2:
            # print("got pair:"+str(imgs))
            imgs_all += imgs
        else:
            print("skip path:"+os.path.join(cur_dir, img_dir, img))
    return imgs_all
