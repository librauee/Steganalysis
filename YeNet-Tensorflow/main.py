import argparse
import numpy as np
import tensorflow as tf
from functools import partial

from utils import *
from generator import *
from queues import *
from YeNet import YeNet

parser = argparse.ArgumentParser(description='PyTorch implementation of YeNet')
parser.add_argument('train_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training cover images')
parser.add_argument('train_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'training stego images or beta maps')
parser.add_argument('valid_cover_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation cover images')
parser.add_argument('valid_stego_dir', type=str, metavar='PATH',
                    help='path of directory containing all ' +
                    'validation stego images or beta maps')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=4e-1, metavar='LR',
                    help='learning rate (default: 4e-1)')
parser.add_argument('--use-batch-norm', action='store_true', default=False,
                    help='use batch normalization after each activation,' +
                    ' also disable pair constraint (default: False)')
parser.add_argument('--embed-otf', action='store_true', default=False,
                    help='use beta maps and embed on the fly instead' +
                    ' of use stego images (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=0,
                    help='index of gpu used (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='how many batches to wait ' +
                    'before logging training status')
parser.add_argument('--log-path', type=str, default='logs/',
                    metavar='PATH', help='path to generated log file')
args = parser.parse_args()

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.no_cuda else str(args.gpu)

tf.set_random_seed(args.seed)
train_ds_size = len(glob(args.train_cover_dir + '/*')) * 2
if args.embed_otf:
    train_gen = partial(gen_embedding_otf, args.train_cover_dir, \
                        args.train_stego_dir, args.use_batch_norm)
else:
    train_gen = partial(gen_flip_and_rot, args.train_cover_dir, \
                        args.train_stego_dir,  args.use_batch_norm)

valid_ds_size = len(glob(args.valid_cover_dir + '/*')) * 2
valid_gen = partial(gen_valid, args.valid_cover_dir, \
                    args.valid_stego_dir)

if valid_ds_size % 32 != 0:
    raise ValueError("change batch size for validation")
    
optimizer = tf.train.AdadeltaOptimizer(args.lr)


train(YeNet, train_gen, valid_gen, args.batch_size, \
      args.test_batch_size, valid_ds_size, \
      optimizer, args.log_interval, train_ds_size, \
      args.epochs * train_ds_size, train_ds_size, args.log_path, 8)
