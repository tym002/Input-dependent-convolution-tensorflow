#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:53:17 2020

@author: tianyu
"""

from keras.layers import Layer


class BiasNet(Layer):
    '''
    Create an independt bias tensor and add it to the input tensor
    
    output: Output tensor with the same shape as the input
    '''
    def __init__(self,**kwargs):

        super(BiasNet, self).__init__(**kwargs)


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, input_shape[3]),
                                      initializer='he_normal',  
                                      trainable=True)
        super(BiasNet, self).build(input_shape)

    def call(self, x, **kwargs):

        return x + self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
