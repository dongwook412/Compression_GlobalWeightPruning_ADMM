#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Activation, BatchNormalization

def CNN(train_shape, num_class, n_layer = 1):
    model = Sequential() 
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=train_shape, use_bias=False))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    for _ in range(n_layer-1):
        model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(num_class, activation='softmax', use_bias=False))
    return model
