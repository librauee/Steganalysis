# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:59:14 2019

@author: Lee
"""

import os
import sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # set a GPU (with GPU Number)
home = os.path.expanduser("~")
sys.path.append(home + '/tflib/')        # path for 'tflib' folder
import matplotlib.pyplot as plt
from scipy.io import loadmat
from SCA_SRNet_Spatial import *         # use  'SCA_SRNet_JPEG' for JPEG domain


def trnGen(cover_path, stego_path, cover_beta_path, stego_beta_path, thread_idx=0, n_threads=1):
    IL=os.listdir(cover_path)
    img_shape = plt.imread(cover_path +IL[0]).shape
    batch = np.empty((2, img_shape[0], img_shape[1], 2), dtype='float32')
    while True:
        indx = np.random.permutation(len(IL))
        for i in indx:
            batch[0,:,:,0] =  plt.imread(cover_path + IL[i])  # use loadmat for loading JPEG decompressed images 
            batch[0,:,:,1] =  loadmat(cover_beta_path + IL[i].replace('pgm','mat'))['Beta'] # adjust for JPEG images
            batch[1,:,:,0] =  plt.imread(stego_path + IL[i])  # use loadmat for loading JPEG decompressed images 
            batch[1,:,:,1] =  loadmat(stego_beta_path + IL[i].replace('pgm','mat'))['Beta'] # adjust for JPEG images
            rot = random.randint(0,3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]   
                

def valGen(cover_path, stego_path, cover_beta_path, stego_beta_path, thread_idx=0, n_threads=1):
    IL=os.listdir(cover_path)
    img_shape = plt.imread(cover_path +IL[0]).shape
    batch = np.empty((2, img_shape[0], img_shape[1], 2), dtype='float32')
    while True:
        for i in range(len(IL)):
            batch[0,:,:,0] =  plt.imread(cover_path + IL[i])  # use loadmat for loading JPEG decompressed images 
            batch[0,:,:,1] =  loadmat(cover_beta_path + IL[i].replace('pgm','mat'))['Beta'] # adjust for JPEG images
            batch[1,:,:,0] =  plt.imread(stego_path + IL[i])  # use loadmat for loading JPEG decompressed images 
            batch[1,:,:,1] =  loadmat(stego_beta_path + IL[i].replace('pgm','mat'))['Beta'] # adjust for JPEG images
            yield [batch, np.array([0,1], dtype='uint8') ]
            
            
train_batch_size = 32
valid_batch_size = 40
max_iter = 500000
train_interval=100
valid_interval=5000
save_interval=5000
num_runner_threads=10

# save Betas as '.mat' files with variable name "Beta" and put them in thier corresponding directoroies. Make sure 
# all mat files in the directories can be loaded in Python without any errors.

TRAIN_COVER_DIR = '/media/Cover_TRN/'
TRAIN_STEGO_DIR = '/media/Stego_WOW_0.5_TRN/'
TRAIN_COVER_BETA_DIR =  '/media/Beta_Cover_WOW_0.5_TRN/'
TRAIN_STEGO_BETA_DIR = '/media/Beta_Stego_WOW_0.5_TRN/'

VALID_COVER_DIR = '/media/Cover_VAL/'
VALID_STEGO_DIR = '/media/Stego_WOW_0.5_VAL/'
VALID_COVER_BETA_DIR = '/media/Beta_Cover_WOW_0.5_VAL/'
VALID_STEGO_BETA_DIR = '/media/Beta_Stego_WOW_0.5_VAL/'

train_gen = partial(trnGen, \
                    TRAIN_COVER_DIR, TRAIN_STEGO_DIR, TRAIN_COVER_BETA_DIR, TRAIN_STEGO_BETA_DIR) 
valid_gen = partial(valGen, \
                    VALID_COVER_DIR, VALID_STEGO_DIR, VALID_COVER_BETA_DIR, VALID_STEGO_BETA_DIR)

LOG_DIR= '/media/LogFiles/SCA_WOW_0.5'  # path for a log direcotry
# load_path= LOG_DIR + 'Model_460000.ckpt'   # continue training from a specific checkpoint
load_path=None                               # training from scratch

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    
train_ds_size = len(glob(TRAIN_COVER_DIR + '/*')) * 2
valid_ds_size = len(glob(VALID_COVER_DIR +'/*')) * 2
print 'train_ds_size: %i'%train_ds_size
print 'valid_ds_size: %i'%valid_ds_size

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")

optimizer = AdamaxOptimizer
boundaries = [400000]     # learning rate adjustment at iteration 400K
values = [0.001, 0.0001]  # learning rates
train(SCA_SRNet, train_gen, valid_gen , train_batch_size, valid_batch_size, valid_ds_size, \
      optimizer, boundaries, values, train_interval, valid_interval, max_iter,\
      save_interval, LOG_DIR,num_runner_threads, load_path)


# Testing 
TEST_COVER_DIR = '/media/Cover_TST/'
TEST_STEGO_DIR = '/media/Stego_WOW_0.5_TST/'
TEST_COVER_BETA_DIR = '/media/Beta_Cover_WOW_0.5_TST/'
TEST_STEGO_BETA_DIR = '/media/Beta_Stego_WOW_0.5_TST/'

test_batch_size=40
LOG_DIR = '/media/LogFiles/SCA_WOW_0.5/' 
LOAD_DIR = LOG_DIR + 'Model_435000.ckpt'        # loading from a specific checkpoint

test_gen = partial(gen_valid, \
                    TEST_COVER_DIR, TEST_STEGO_DIR)

test_ds_size = len(glob(TEST_COVER_DIR + '/*')) * 2
print 'test_ds_size: %i'%test_ds_size

if test_ds_size % test_batch_size != 0:
    raise ValueError("change batch size for testing!")

test_dataset(SCA_SRNet, test_gen, test_batch_size, test_ds_size, LOAD_DIR)