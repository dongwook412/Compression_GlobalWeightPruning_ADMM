#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def MNIST():

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, X_test, Y_train, Y_test
