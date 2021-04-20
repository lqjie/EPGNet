# coding=utf-8
## This code is based on offical SRNet code (http://dde.binghamton.edu/download/feature_extractors/)
import os
import sys
import argparse
import numpy as np
import random
import tensorflow as tf
sys.path.append('./tflib/')
from SCA_SRNet_JPEG import *
from SCA_SRNet_Spatial import *
from EPG_SRNet import *
from SRNet import *
from datetime import datetime
import json

parser = argparse.ArgumentParser(description='TF implementation of SRNet')
parser.add_argument('--cover_path', type=str,
                    metavar='PATH', help='path of directory containing all cover images')
parser.add_argument('--stego_path', type=str,
                    metavar='PATH', help='path of directory containing all stego images')
parser.add_argument('--cover_beta_path', type=str,
                    metavar='PATH', help='path of directory containing all cover beta images')
parser.add_argument('--stego_beta_path', type=str,
                    metavar='PATH', help='path of directory containing all stego beta images')
parser.add_argument('--model_name', type=str, default='SRNet', metavar='PATH',
                    help='model_name')
parser.add_argument('--data_split', type=str, default='data_split_v1', metavar='PATH',
                    help='data_split file name (json file)')
parser.add_argument('--LOG_DIR', type=str, default='s1234_1', metavar='PATH',
                    help='LOG_DIR')
parser.add_argument('--load_path', type=str, default=None, metavar='PATH',
                    help='resume path')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--is_jpeg', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--valid_batch_size', type=int, default=40, metavar='N',
                    help='input batch size for validation (default: 40)')
parser.add_argument('--max_iter', type=int, default=125000, metavar='N',
                    help='max iterations (default: 125000)')
parser.add_argument('--step_iter', type=int, default=100000, metavar='N',
                    help='step_iter (default: 100000)')
parser.add_argument('--valid_interval', type=int, default=2500, metavar='N',
                    help='valid_interval (default: 2500)')
parser.add_argument('--init_lr', type=float, default=1e-3, metavar='LR',
                    help='inital learning rate (default: 1e-3)')
parser.add_argument('--step_lr', type=float, default=1e-4, metavar='LR',
                    help='step learning rate (default: 1e-4)')
parser.add_argument('--gpu', type=int, default=0,
                    help='index of gpu used (default: 0)')
args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
train_batch_size = args.batch_size
valid_batch_size = args.valid_batch_size
max_iter = args.max_iter

train_interval = 500
valid_interval = args.valid_interval
save_interval = 5000
num_runner_threads = 10
start_save_iter = 100000

LOG_DIR = os.path.join('./ckpt', args.LOG_DIR)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_info_file = os.path.join('./ckpt/', 'info.txt')
load_path = args.load_path
best_ckpt_path = os.path.join(LOG_DIR, 'best_model.ckpt')

with open(args.data_split + '.json', "r") as f:
    data_list = json.load(f)
train_list = data_list['train']
valid_list = data_list['valid']
test_list = data_list['test']

# The format of data_split.json is similar to the following:
# {"train": ["20690.tif", "65494.tif", ..],
#  "valid": ["49460.tif", "18854.tif", ...],
#  "test": ["67702.tif", "18080.tif", ...]}

if args.model_name == 'SRNet':
    model_class = SRNet
elif args.model_name == 'SCA_SRNet_Spatial':
    model_class = SCA_SRNet_Spatial
elif args.model_name == 'SCA_SRNet_JPEG':
    model_class = SCA_SRNet_JPEG
elif args.model_name == 'EPG_SRNet':
    model_class = EPG_SRNet


if args.model_name == 'SRNet':
    is_sca = 0
else:
    is_sca = 1
if args.is_jpeg:
    train_gen = partial(trnGen_jpeg, train_list, is_sca, args.cover_path, args.stego_path,\
                        args.cover_beta_path, args.stego_beta_path)
    valid_gen = partial(ValGen_jpeg, valid_list, is_sca, args.cover_path, args.stego_path,\
                        args.cover_beta_path, args.stego_beta_path)
    test_gen = partial(ValGen_jpeg, test_list, is_sca, args.cover_path, args.stego_path,
                    args.cover_beta_path, args.stego_beta_path)
else:
    train_gen = partial(trnGen_tif, train_list, is_sca, args.cover_path, args.stego_path,\
                        args.cover_beta_path, args.stego_beta_path)
    valid_gen = partial(ValGen_tif, valid_list, is_sca, args.cover_path, args.stego_path,\
                        args.cover_beta_path, args.stego_beta_path)
    test_gen = partial(ValGen_tif, test_list, is_sca, args.cover_path, args.stego_path,
                    args.cover_beta_path, args.stego_beta_path)

train_ds_size = len(train_list) * 2
valid_ds_size = len(valid_list) * 2
test_ds_size = len(test_list) * 2
print('train_ds_size: %i'%train_ds_size)
print('valid_ds_size: %i'%valid_ds_size)
print('test_ds_size: %i'%test_ds_size)

if valid_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for validation")
if test_ds_size % valid_batch_size != 0:
    raise ValueError("change batch size for test")

optimizer = AdamaxOptimizer
boundaries = [args.step_iter]
values = [args.init_lr, args.step_lr]
exp_info_str = datetime.now().strftime(r'%m%d_%H%M%S') + ' model: ' + args.LOG_DIR + ' split: ' + args.data_split + '\n'
best_iter, best_val_acc = train(model_class, train_gen, valid_gen, train_batch_size, valid_batch_size, valid_ds_size, \
      optimizer, boundaries, values, train_interval, valid_interval, max_iter, \
      save_interval, start_save_iter, LOG_DIR, num_runner_threads, load_path)

test_acc = test_dataset(model_class, test_gen, valid_batch_size, test_ds_size, best_ckpt_path)
print('test_accucy:{:.4f}'.format(test_acc))
file = open(log_info_file, 'a')
file.write(exp_info_str)
file.write('best_iter: {0},best_valid:{1:.4f} test_last: {2:.4f}\n'.format(best_iter, best_val_acc, test_acc))
file.close()
