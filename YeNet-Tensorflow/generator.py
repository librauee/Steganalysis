import numpy as np
from scipy import misc, io
from glob import glob
import random
from itertools import izip
from random import random as rand
from random import shuffle
import h5py

def gen_embedding_otf(cover_dir, beta_dir, shuf_pair, \
                      thread_idx=0, n_threads=1):
    cover_list = sorted(glob(cover_dir + '/*'))
    beta_list = sorted(glob(beta_dir + '/*'))
    nb_data = len(cover_list)
    assert len(beta_list) != 0, "the beta directory '%s' is empty" % beta_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(beta_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(beta_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='uint8')
    beta_map = np.empty(img_shape, dtype='<f8')
    inf_map = np.empty(img_shape, dtype='bool')
    rand_arr = np.empty(img_shape, dtype='float64')
    shuf_cov = np.empty(img_shape, dtype='uint8')
    while True:
        if shuf_pair:
            list_i = np.random.permutation(nb_data)
            list_j = np.random.permutation(nb_data)
            for i, j in izip(list_i, list_j):
                batch[0,:,:,0] = misc.imread(cover_list[i])
                beta_map[:,:] = io.loadmat(beta_list[j])['pChange']
                shuf_cov[:,:] = misc.imread(cover_list[j])
                rand_arr[:,:] = np.random.rand(img_shape[0], img_shape[1])
                inf_map[:,:] = rand_arr < (beta_map / 2.)
                batch[1,:,:,0] = np.copy(shuf_cov)
                batch[1,np.logical_and(shuf_cov != 255, inf_map),0] += 1
                batch[1,np.logical_and(shuf_cov != 0, \
                      np.logical_and(np.logical_not(inf_map), \
                      rand_arr < beta_map)), 0] -= 1
                rot = random.randint(0,3)
                if rand() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]
        else:
            list_i = np.random.permutation(nb_data)
            for i in list_i:
                batch[0,:,:,0] = misc.imread(cover_list[i])
                beta_map[:,:] = io.loadmat(beta_list[i])['pChange']
                rand_arr[:,:] = np.random.rand(img_shape[0], img_shape[1])
                inf_map[:,:] = rand_arr < (beta_map / 2.)
                batch[1,:,:,0] = np.copy(batch[0,:,:,0])
                batch[1,np.logical_and(batch[0,:,:,0] != 255, inf_map),0] += 1
                batch[1,np.logical_and(batch[0,:,:,0] != 0, \
                      np.logical_and(np.logical_not(inf_map), \
                      rand_arr < beta_map)), 0] -= 1
                rot = random.randint(0,3)
                if rand() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]

def gen_all_flip_and_rot(cover_dir, stego_dir, thread_idx, n_threads):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the beta directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    iterable = zip(cover_list, stego_list)
    for cover_path, stego_path in iterable:
        batch[0,:,:,0] = misc.imread(cover_path)
        batch[1,:,:,0] = misc.imread(stego_path)
        for rot in range(4):
            yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
        for rot in range(4):
            yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]

def gen_flip_and_rot(cover_dir, stego_dir, shuf_pair=False, thread_idx=0, n_threads=1):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the beta directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, + \
                                      len(stego_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    if not shuf_pair:
        iterable = zip(cover_list, stego_list)
    while True:
        if shuf_pair:
            shuffle(cover_list)
            shuffle(stego_list)
            iterable = izip(cover_list, stego_list)
        else:
            shuffle(iterable)
        for cover_path, stego_path in iterable:
            batch[0,:,:,0] = misc.imread(cover_path)
            batch[1,:,:,0] = misc.imread(stego_path)
            rot = random.randint(0,3)
            if rand() < 0.5:
                yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
            else:
                yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]

def gen_valid(cover_dir, stego_dir, thread_idx, n_threads):
    cover_list = sorted(glob(cover_dir + '/*'))
    stego_list = sorted(glob(stego_dir + '/*'))
    nb_data = len(cover_list)
    assert len(stego_list) != 0, "the beta directory '%s' is empty" % stego_dir
    assert nb_data != 0, "the cover directory '%s' is empty" % cover_dir
    assert len(stego_list) == nb_data, "the cover directory and " + \
                                      "the beta directory don't " + \
                                      "have the same number of files " + \
                                      "respectively %d and %d" % (nb_data, \
                                      len(stego_list))
    img = misc.imread(cover_list[0])
    img_shape = img.shape
    batch = np.empty((2,img_shape[0],img_shape[1],1), dtype='uint8')
    labels = np.array([0, 1], dtype='uint8')
    while True:
        for cover_path, stego_path in zip(cover_list, stego_list):
            batch[0,:,:,0] = misc.imread(cover_path)
            batch[1,:,:,0] = misc.imread(stego_path)
            yield [batch, labels]


# def trainGen(thread_idx, n_threads):
#     batch = np.empty((2,256,256,1), dtype='uint8')
#     beta_map = np.empty((256, 256), dtype='<f8')
#     inf_map = np.empty((256, 256), dtype='bool')
#     rand_arr = np.empty((256, 256), dtype='float64')

#     cover_ds_path = LOG_DIR + '/cover_train' + str(thread_idx) +'.txt'
#     shuf_cover_ds_path = LOG_DIR + '/shuf_cover_train' + str(thread_idx) +'.txt'
#     shuf_beta_ds_path = LOG_DIR + '/shuf_beta_train' + str(thread_idx) +'.txt'
#     createSubFile(DS_PATH + 'train_cover.txt', \
#                   cover_ds_path, thread_idx, n_threads)
#     createSubFile(DS_PATH + 'train_beta.txt', \
#                   shuf_beta_ds_path, thread_idx, n_threads)
#     createSubFile(DS_PATH + 'train_cover.txt', \
#                   shuf_cover_ds_path, thread_idx, n_threads)
#     while True:
#         with open(cover_ds_path, 'r') as f_cover, \
#                 open(shuf_cover_ds_path, 'r') as f_shuf_cover, \
#                 open(shuf_beta_ds_path, 'r') as f_shuf_beta:
#             for cov_path, shuf_cov_path, shuf_beta_path in \
#                     izip(f_cover, f_shuf_cover, f_shuf_beta):
#                 batch[0,:,:,0] = misc.imread(str.strip(cov_path))
#                 shuf_cov = misc.imread(str.strip(shuf_cov_path))
#                 beta_map[:,:] = io.loadmat(str.strip(shuf_beta_path))['pChange']
#                 rand_arr[:,:] = np.random.rand(256, 256)
#                 inf_map[:,:] = rand_arr < (beta_map / 2.)
#                 batch[1,:,:,0] = np.copy(shuf_cov)
#                 batch[1,np.logical_and(shuf_cov != 255, inf_map),0] += 1
#                 batch[1,np.logical_and(shuf_cov != 0, \
#                       np.logical_and(np.logical_not(inf_map), rand_arr < beta_map)), 0] -= 1
#                 rot = random.randint(0,3)
#                 if rand() < 0.5:
#                     yield [np.rot90(batch, rot, axes=[1,2]), np.array([0,1], dtype='uint8')]
#                 else:
#                     yield [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,1], dtype='uint8')]
