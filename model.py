import os,sys,shutil
import tensorflow as tf
import numpy as np
import argparse
import cv2,math,glob,random,time
import time
import matplotlib.pyplot as plt

REGULARIZER_COF = 2e-4

def _fc_variable( weight_shape,name="fc"):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = (input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)

        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer =regularizer)
        bias   = tf.get_variable("b", [weight_shape[1]],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _deconv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape    ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _deconv2d( x, W, output_shape, stride=1):
    # x           : [nBatch, height, width, in_channels]
    # output_shape: [nBatch, height, width, out_channels]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")


def _conv_layer(x, input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormc"+name)
    h = tf.nn.leaky_relu(h)
    return h

def _deconv_layer(x,input_layer, output_layer, stride=2, filter_size=3, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*stride,w*stride,output_layer], stride=stride) + deconv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormd"+name)
    h = tf.nn.leaky_relu(h)
    return h


def buildGenerator(x,reuse=False,isTraining=True,name="Generator"):

    with tf.variable_scope(name, reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        # 1/1

        h = _conv_layer(x, 3, 32, 1, 7,"1-1e_g")
        h = _conv_layer(h, 32, 32, 1, 5,"1-2e_g")
        enc1 = h
        # -> 1/2
        h = _conv_layer(h, 32, 64, 2, 5,"2-1e_g")
        h = _conv_layer(h, 64, 64, 1, 5,"2-2e_g")
        h = _conv_layer(h, 64, 64, 1, 5,"2-3e_g")
        enc2 = h
        # -> 1/4
        h = _conv_layer(h, 64, 128, 2, 5,"3-1e_g")
        h = _conv_layer(h, 128, 128, 1, 5,"3-2e_g")
        h = _conv_layer(h, 128, 128, 1, 5,"3-3e_g")
        enc3 = h
        # ->1/8
        h = _conv_layer(h, 128, 256, 2, 5,"4-1e_g")
        h = _conv_layer(h, 256, 256, 1, 5,"4-2e_g")
        h = _conv_layer(h, 256, 256, 1, 5,"4-3e_g")
        enc4 = h
        # -> 1/16
        h = _conv_layer(h, 256, 256, 2, 5,"5-1e_g")
        enc5 = h
        h = _conv_layer(h, 256, 256, 1, 5,"5-2e_g")
        h = _conv_layer(h, 256, 256, 1, 5,"5-3e_g")
        h = h + enc5
        enc5 = h
        h = _conv_layer(h, 256, 256, 1, 5,"5-4e_g")
        h = _conv_layer(h, 256, 256, 1, 5,"5-5e_g")
        h = h + enc5
        # ->1/8
        h = _deconv_layer(h, 256, 256, 2, 5, "5d_g")
        h = tf.concat([h,enc4,],axis=3)
        h = _conv_layer(h, 512, 256, 1, 5,"4-1d_g")
        h = _conv_layer(h, 256, 256, 1, 5,"4-2d_g")
        # -> 1/4
        h = _deconv_layer(h, 256, 128, 2, 5, "4d_g")
        h = tf.concat([h,enc3],axis=3)
        h = _conv_layer(h, 256, 128, 1, 5,"3-1d_g")
        h = _conv_layer(h, 128, 128, 1, 5,"3-2d_g")
        # -> 1/2
        h = _deconv_layer(h, 128, 64, 2, 5, "3d_g")
        h = tf.concat([h,enc2],axis=3)
        h = _conv_layer(h, 128, 64, 1, 5,"2-1d_g")
        h = _conv_layer(h, 64, 64, 1, 5,"2-2d_g")
        # -> 1/1
        h = _deconv_layer(h, 64, 32, 2, 5, "2d_g")
        h = tf.concat([h,enc1],axis=3)
        h = _conv_layer(h, 64, 32, 1, 5,"1-1d_g")
        h = _conv_layer(h, 32, 32, 1, 5,"1-2d_g")

        conv_w, conv_b = _conv_variable([7,7,32,3],name="convo_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)
    return y

def _conv_layer_dis(x,input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormc"+name)
    h = tf.nn.leaky_relu(h)
    return h

def buildDiscriminator(x,y,reuse=False,isTraining=False,nBatch=16, ksize=5):
    with tf.variable_scope("Discriminator") as scope:
        if reuse: scope.reuse_variables()
        bs, h, w, c_x = x.get_shape().as_list()
        bs, h, w, c_y = y.get_shape().as_list()
        input_channels = 6
        fn_l = 32
        h =  tf.concat([x,y], axis=3)
        # conv1
        h = _conv_layer_dis(h, input_channels, fn_l, 2, ksize, "1-1di")
        # conv2
        h = _conv_layer_dis(h, fn_l, fn_l*2, 2, ksize, "2-1di")
        # conv3
        h = _conv_layer_dis(h, fn_l*2, fn_l*4, 2, ksize, "3-1di")
        # conv4
        h = _conv_layer_dis(h, fn_l*4, fn_l*8, 2, ksize, "4-1di")

        # conv5
        d_convo_w, d_convo_b = _conv_variable([5,5,fn_l*8,1],name="conv5")
        h = _conv2d(h,d_convo_w, stride=1) + d_convo_b
    return h
