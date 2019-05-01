#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import argparse
import numpy as np
from easydict import EasyDict

parser = argparse.ArgumentParser(
    description='Pyramidbox face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--basenet',
                    # default='vgg16_reducedfc.pth',
                    default='pyramidbox_120000_99.02.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size',
                    default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=False, type=bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

_C = EasyDict()
cfg = _C

# data augument config
_C.expand_prob = 0.5
_C.expand_max_ratio = 4
_C.hue_prob = 0.5
_C.hue_delta = 18
_C.contrast_prob = 0.5
_C.contrast_delta = 0.5
_C.saturation_prob = 0.5
_C.saturation_delta = 0.5
_C.brightness_prob = 0.5
_C.brightness_delta = 0.125
_C.data_anchor_sampling_prob = 0.5
_C.min_face_size = 6.0
_C.apply_distort = True
_C.apply_expand = False
_C.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
    'float32')
_C.resize_width = 640
_C.resize_height = 640
_C.scale = 1 / 127.0
_C.anchor_sampling = True
_C.filter_min_face = True

# train config
_C.LR_STEPS = (80000,100000,120000)
_C.MAX_STEPS = 150000
_C.EPOCHES = 100

# anchor config
_C.FEATURE_MAPS = [160, 80, 40, 20, 10, 5]
_C.INPUT_SIZE = 640
_C.STEPS = [4, 8, 16, 32, 64, 128]
_C.ANCHOR_SIZES = [16, 32, 64, 128, 256, 512]
_C.CLIP = False
_C.VARIANCE = [0.1, 0.2]

# loss config
_C.NUM_CLASSES = 2
_C.OVERLAP_THRESH = 0.35
_C.NEG_POS_RATIOS = 3
# DSN loss weight
_C.ALPHA = 0.01
_C.BETA = 0.05
_C.GAMMA = 0.25

# detection config
_C.NMS_THRESH = 0.3
_C.TOP_K = 5000
_C.KEEP_TOP_K = 750
_C.CONF_THRESH = 0.05


# dataset config
_C.HOME = os.environ['LLFD_ROOT']

# face config
_C.FACE = EasyDict()
_C.FACE.TRAIN_FILE_S = './data/face_train.txt'
_C.FACE.VAL_FILE_S = './data/face_val.txt'
_C.FACE.TRAIN_FILE_T = './data/face_train_df.txt'
_C.FACE.VAL_FILE_T = './data/face_val_df.txt'
_C.FACE.FDDB_DIR = '/home/data/lj/FDDB'
_C.FACE.WIDER_DIR = '/home/data/lj/WIDER'
_C.FACE.AFW_DIR = '/home/data/lj/AFW'
_C.FACE.PASCAL_DIR = '/home/data/lj/PASCAL_FACE'
_C.FACE.OVERLAP_THRESH = 0.35

