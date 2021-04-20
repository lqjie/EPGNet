#coding=utf-8
import numpy as np
from glob import glob
import random
from random import random as rand
from random import shuffle
from scipy.io import loadmat
import cv2

def trnGen_jpeg(data_list, is_sca,  cover_path, stego_path, cover_beta_path, stego_beta_path, thread_idx=0, n_threads=1):
    img_list = [name.replace('tif', 'mat') for name in data_list]

    if is_sca:
        beta_list = [name.replace('tif', 'mat') for name in data_list]
        img_shape = loadmat(cover_path + img_list[0],verify_compressed_data_integrity=False)['img'].shape
        batch = np.empty((2, img_shape[0], img_shape[1], 2), dtype='float32')
        while True:
            indx = np.random.permutation(len(img_list))
            for i in indx:
                batch[0, :, :, 0] = loadmat(cover_path + img_list[i],verify_compressed_data_integrity=False)['img']
                batch[0, :, :, 1] = loadmat(cover_beta_path + beta_list[i])['Beta']

                batch[1, :, :, 0] = loadmat(stego_path + img_list[i],verify_compressed_data_integrity=False)['img']
                batch[1, :, :, 1] = loadmat(stego_beta_path + beta_list[i])['Beta']

                rot = random.randint(0, 3)
                if rand() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1, 2]), np.array([0, 1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), np.array([0, 1], dtype='uint8')]
    else:
        img_shape = loadmat(cover_path + img_list[0],verify_compressed_data_integrity=False)['img'].shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
        while True:
            indx = np.random.permutation(len(data_list))
            for i in indx:
                batch[0, :, :, 0] = loadmat(cover_path + img_list[i], verify_compressed_data_integrity=False)['img']
                batch[1, :, :, 0] = loadmat(stego_path + img_list[i], verify_compressed_data_integrity=False)['img']

                rot = random.randint(0, 3)
                if rand() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1, 2]), np.array([0, 1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), np.array([0, 1], dtype='uint8')]
def ValGen_jpeg(data_list, is_sca, cover_path, stego_path, cover_beta_path, stego_beta_path, thread_idx=0, n_threads=1):
    img_list = [name.replace('tif', 'mat') for name in data_list]

    if is_sca:
        beta_list = [name.replace('tif', 'mat') for name in data_list]
        img_shape = loadmat(cover_path + img_list[0], verify_compressed_data_integrity=False)['img'].shape
        batch = np.empty((2, img_shape[0], img_shape[1], 2), dtype='float32')
        while True:
            indx = np.random.permutation(len(img_list))
            for i in indx:
                batch[0, :, :, 0] = loadmat(cover_path + img_list[i], verify_compressed_data_integrity=False)['img']
                batch[0, :, :, 1] = loadmat(cover_beta_path + beta_list[i])['Beta']

                batch[1, :, :, 0] = loadmat(stego_path + img_list[i], verify_compressed_data_integrity=False)['img']
                batch[1, :, :, 1] = loadmat(stego_beta_path + beta_list[i])['Beta']

                yield [batch, np.array([0,1], dtype='uint8')]
    else:
        img_shape = loadmat(cover_path + img_list[0], verify_compressed_data_integrity=False)['img'].shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
        while True:
            for i in range(len(data_list)):
                batch[0, :, :, 0] = loadmat(cover_path + img_list[i], verify_compressed_data_integrity=False)['img']
                batch[1, :, :, 0] = loadmat(stego_path + img_list[i], verify_compressed_data_integrity=False)['img']
                yield [batch, np.array([0,1], dtype='uint8')]



def trnGen_tif(data_list, is_sca, cover_path, stego_path, cover_beta_path, stego_beta_path, thread_idx=0, n_threads=1):
    img_list = data_list
    if is_sca:
        beta_list = [name.replace('tif', 'mat') for name in data_list]
        img_shape = cv2.imread(cover_path + img_list[0], -1).shape
        batch = np.empty((2, img_shape[0], img_shape[1], 2), dtype='float32')
        while True:
            indx = np.random.permutation(len(img_list))
            for i in indx:
                
                batch[0, :, :, 0] = cv2.imread(cover_path + img_list[i],-1)
                batch[0, :, :, 1] = loadmat(cover_beta_path + beta_list[i])['Beta']

                batch[1, :, :, 0] = cv2.imread(stego_path + img_list[i],-1)
                batch[1, :, :, 1] = loadmat(stego_beta_path + beta_list[i])['Beta']

                rot = random.randint(0, 3)
                if rand() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1, 2]), np.array([0, 1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), np.array([0, 1], dtype='uint8')]
    else:
        img_shape = cv2.imread(cover_path + img_list[0], -1).shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
        while True:
            indx = np.random.permutation(len(data_list))
            for i in indx:
                batch[0, :, :, 0] = cv2.imread(cover_path + img_list[i],-1)
                batch[1, :, :, 0] = cv2.imread(stego_path + img_list[i],-1)
                rot = random.randint(0, 3)
                if rand() < 0.5:
                    yield [np.rot90(batch, rot, axes=[1, 2]), np.array([0, 1], dtype='uint8')]
                else:
                    yield [np.flip(np.rot90(batch, rot, axes=[1, 2]), axis=2), np.array([0, 1], dtype='uint8')]
def ValGen_tif(data_list, is_sca, cover_path, stego_path, cover_beta_path, stego_beta_path, thread_idx=0, n_threads=1):
    img_list = data_list
    if is_sca:
        beta_list = [name.replace('tif', 'mat') for name in data_list]
        img_shape = cv2.imread(cover_path + img_list[0], -1).shape
        batch = np.empty((2, img_shape[0], img_shape[1], 2), dtype='float32')
        while True:
            for i in range(len(data_list)):
                batch[0, :, :, 0] = cv2.imread(cover_path + img_list[i],-1)
                batch[0, :, :, 1] = loadmat(cover_beta_path + beta_list[i])['Beta']

                batch[1, :, :, 0] = cv2.imread(stego_path + img_list[i],-1)
                batch[1, :, :, 1] = loadmat(stego_beta_path + beta_list[i])['Beta']

                yield [batch, np.array([0,1], dtype='uint8')]
    else:
        img_shape = cv2.imread(cover_path + img_list[0], -1).shape
        batch = np.empty((2, img_shape[0], img_shape[1], 1), dtype='float32')
        while True:
            for i in range(len(data_list)):
                batch[0, :, :, 0] = cv2.imread(cover_path + img_list[i],-1)
                batch[1, :, :, 0] = cv2.imread(stego_path + img_list[i],-1)
                yield [batch, np.array([0,1], dtype='uint8')]



