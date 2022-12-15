# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 15:45:21 2022

@author: USUARIO
"""

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.layers import Activation, Concatenate


def conv_block(input, num_filters):
    x = Conv3D(num_filters, (3, 3, 3), padding="same")(input)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation("relu")(x)

    x = Conv3D(num_filters, (3, 3, 3), padding="same")(x)
    x = BatchNormalization(momentum=0.9)(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling3D((3, 3, 3), strides=(2, 2, 2), padding="same")(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, n_classes):
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 48)
    s2, p2 = encoder_block(p1, 96)
    s3, p3 = encoder_block(p2, 192)

    b1 = conv_block(p3, 384) #Bridge

    d1 = decoder_block(b1, s3, 192)
    d2 = decoder_block(d1, s2, 96)
    d3 = decoder_block(d2, s1, 48)

    if n_classes == 1:  #Binary
      activation = 'sigmoid'
    else:
      activation = 'softmax'

    outputs = Conv3D(n_classes, 1, padding="same", activation=activation)(d3)  #Change the activation based on n_classes
    print(activation)

    model = Model(inputs, outputs, name="U-Net")
    return model