#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:55:15 2021

@author: tianyu
"""


from keras.layers import MaxPooling2D,LeakyReLU,Reshape,Conv2D,BatchNormalization,Dense,Flatten,Dropout,Lambda,Activation 
from BiasNet import BiasNet
import tensorflow as tf


def p_conv(tupl):
    '''
    Apply convolution with input features and convolutional kernels 
    
    tupl: a tuple of (input feature, convolutional kernels)
    
    output: the output features after applied conovlution 
    '''

    ip,kernal = tupl
    pos = kernal
    out = tf.nn.convolution(ip,pos,padding='SAME')
    return out

def p_kernal(ip,dim,ch_in,ch_out):
    '''
    Hyper-network that generate convolutional weights
    
    ip: input feature maps to the hyper-network
    dim: dimension of the convolutional kernel 
    ch_in: number of the input channels for the kernel to convolve with 
    ch_out: number of the output channels after applying convolution 
    
    Output: convolutional kernel with shape (batch, dim, dim, ch_in, ch_out)
    '''
    
    num_c = int(ch_in * ch_out)
    pos = Conv2D(16, (3, 3), activation=None, use_bias=True, kernel_initializer='he_normal')(ip)
    pos = LeakyReLU(alpha=0.1)(pos)
    
    pos = MaxPooling2D()(pos)
    
    pos = Conv2D(16, (3, 3), activation=None, use_bias=True, kernel_initializer='he_normal')(pos)
    pos = LeakyReLU(alpha=0.1)(pos)
    
    pos = MaxPooling2D()(pos)

    pos = Conv2D(8, (3, 3), activation=None, use_bias=True, kernel_initializer='he_normal')(pos)
    pos = LeakyReLU(alpha=0.1)(pos)
    
    pos = Flatten()(pos)
    pos = Dense(128)(pos)
    pos = Dropout(0.2)(pos)
    pos = Dense(dim*dim*num_c)(pos)

    pos = Reshape((dim, dim,ch_in,ch_out))(pos)
    return pos

def hyper_block(ip,dim,ch_in,ch_out,acti='relu',bn=False):
    '''
    Generate an input conditioned convolutional kernel and apply it to the same input features
    
    ip: input features
    dim: dimension of the convolutional kernel
    ch_in: number of channels of the input features
    ch_out: number of the output channels after the convolution 
    acti: type of the activation function
    bn: a boolean, whether to use batch_normalization
    
    output: the output features after applying conovlution 
    '''
    
    kernel = p_kernal(ip,dim,ch_in,ch_out)
    n = Lambda(lambda x: tf.squeeze(tf.map_fn(p_conv, (tf.expand_dims(x[0], 1), x[1]), dtype=tf.float32),axis=1))([ip,kernel])
    n = BiasNet()(n)
    n = BatchNormalization()(n) if bn else n
    if acti:
       n = Activation(acti)(n)
    return n


