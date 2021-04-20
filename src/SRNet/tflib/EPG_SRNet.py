import tensorflow as tf
from functools import partial
from tensorflow.contrib import layers
from tensorflow.contrib.framework import arg_scope
import functools
from queues import *
from generator import *
from utils_multistep_lr import *

import numpy as np
def bn_layer(input_layer, is_training, scope=None):
    bn = layers.batch_norm(inputs=input_layer, decay=0.9, center=True, scale=True,
                           updates_collections=None, is_training=is_training,
                           fused=True)
    return bn
def fc_layer(input, out_dims):
    ip = layers.fully_connected(input, num_outputs=out_dims,
                                activation_fn=None, normalizer_fn=None,
                                weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                biases_initializer=tf.constant_initializer(0.))
    return ip
def GAP(x):
    x = tf.reduce_mean(x, [1,2], keep_dims=True)
    x = layers.flatten(x)
    return x

def conv_layer(input_layer, num_outputs=16, kernel_size=3, stride=1, bias=True):
    if bias:
        bias_init = tf.constant_initializer(0.2)
    else:
        bias_init = None
    conv = layers.conv2d(inputs=input_layer,
                         num_outputs=num_outputs,
                         kernel_size=kernel_size, stride=stride, padding='SAME',
                         activation_fn=None,
                         weights_initializer=layers.variance_scaling_initializer(),
                         weights_regularizer=layers.l2_regularizer(5e-4),
                         biases_initializer=bias_init,
                         biases_regularizer=None
                         )
    return conv


def conv_bn_relu(input, is_training, num_outputs=16, kernel_size=3,stride=1):
    output = conv_layer(input, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride)

    output = tf.nn.relu(bn_layer(output, is_training))
    return output
def conv_bn(input, is_training, num_outputs=16, kernel_size=3,stride=1):
    output = conv_layer(input, num_outputs=num_outputs, kernel_size=kernel_size, stride=stride)

    output = bn_layer(output, is_training)
    return output

def EPGM(img, beta, is_training, num_outputs=16, pool_stride=1):

    beta_mask = layers.avg_pool2d(inputs=beta, kernel_size=[3, 3], stride=[pool_stride, pool_stride], padding='SAME')

    beta_mask = conv_layer(beta_mask, num_outputs=num_outputs, kernel_size=3, stride=1)
    beta_mask = tf.nn.sigmoid(beta_mask)
    output = tf.multiply(img, beta_mask)

    output = conv_layer(output, num_outputs=num_outputs, kernel_size=1, stride=1)

    output = tf.nn.relu(bn_layer(output, is_training))+img
    return output

class EPG_SRNet(Model):
    def _build_model(self, input_batch):
        inputs_image, inputs_Beta = tf.split(input_batch, num_or_size_splits=2, axis=3)
        if self.data_format == 'NCHW':
            reduction_axis = [2, 3]
            _inputs_image = tf.cast(tf.transpose(inputs_image, [0, 3, 1, 2]), tf.float32)
            _inputs_Beta = tf.cast(tf.transpose(inputs_Beta, [0, 3, 1, 2]), tf.float32)
        else:
            reduction_axis = [1, 2]
            _inputs_image = tf.cast(inputs_image, tf.float32)
            _inputs_Beta = tf.cast(inputs_Beta, tf.float32)
        with arg_scope([layers.conv2d], num_outputs=16,
                       kernel_size=3, stride=1, padding='SAME',
                       data_format=self.data_format,
                       activation_fn=None,
                       weights_initializer=layers.variance_scaling_initializer(),
                       weights_regularizer=layers.l2_regularizer(2e-4),
                       biases_initializer=tf.constant_initializer(0.2),
                       biases_regularizer=None), \
             arg_scope([layers.batch_norm],
                       decay=0.9, center=True, scale=True,
                       updates_collections=None, is_training=self.is_training,
                       fused=True, data_format=self.data_format), \
             arg_scope([layers.avg_pool2d],
                       kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                       data_format=self.data_format):
            with tf.variable_scope('Layer1'):
                beta = _inputs_Beta
                conv = layers.conv2d(_inputs_image, num_outputs=64, kernel_size=3)
                actv = tf.nn.relu(layers.batch_norm(conv))
                low_fusion = EPGM(actv, beta, self.is_training, num_outputs=64, pool_stride=1)
            with tf.variable_scope('Layer2'):
                conv = layers.conv2d(low_fusion)
                actv = tf.nn.relu(layers.batch_norm(conv))
            with tf.variable_scope('Layer3'):  # 256*256
                conv1 = layers.conv2d(actv)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(actv, bn2)
            with tf.variable_scope('Layer4'):  # 256*256
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn2 = layers.batch_norm(conv2)
                res = tf.add(res, bn2)
            with tf.variable_scope('Layer5'):  # 256*256
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer6'):  # 256*256
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer7'):  # 256*256
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                res = tf.add(res, bn)
            with tf.variable_scope('Layer8'):  # 256*256
                convs = layers.conv2d(res, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
                mid_fusion = EPGM(res, beta, self.is_training, num_outputs=16, pool_stride=2)
            with tf.variable_scope('Layer9'):  # 128*128
                convs = layers.conv2d(mid_fusion, num_outputs=64, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(mid_fusion, num_outputs=64)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=64)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer10'):  # 64*64
                convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(res, num_outputs=128)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=128)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
                high_fusion = EPGM(res, beta, self.is_training, num_outputs=128, pool_stride=8)
            with tf.variable_scope('Layer11'):  # 32*32
                convs = layers.conv2d(high_fusion, num_outputs=256, kernel_size=1, stride=2)
                convs = layers.batch_norm(convs)
                conv1 = layers.conv2d(high_fusion, num_outputs=256)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=256)
                bn = layers.batch_norm(conv2)
                pool = layers.avg_pool2d(bn)
                res = tf.add(convs, pool)
            with tf.variable_scope('Layer12'):  # 16*16
                conv1 = layers.conv2d(res, num_outputs=512)
                actv1 = tf.nn.relu(layers.batch_norm(conv1))
                conv2 = layers.conv2d(actv1, num_outputs=512)
                head = layers.batch_norm(conv2)
            with tf.variable_scope('decision_fusion'):
                low_fea = conv_bn_relu(low_fusion, self.is_training, num_outputs=16, kernel_size=2, stride=2)
                mid_fusion = tf.concat([mid_fusion, low_fea], axis=3)
                mid_fea = conv_bn_relu(mid_fusion, self.is_training, num_outputs=128, kernel_size=4, stride=4)
                high_fusion = tf.concat([high_fusion, mid_fea], axis=3)
                high_fea = conv_bn(high_fusion, self.is_training, num_outputs=512, kernel_size=2, stride=2)
                head = tf.concat([head, high_fea], axis=3)
                head_fea = GAP(head)
        ip = layers.fully_connected(head_fea, num_outputs=2,
                                    activation_fn=None, normalizer_fn=None,
                                    weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01),
                                    biases_initializer=tf.constant_initializer(0.), scope='ip')
        self.outputs = ip
        return self.outputs


