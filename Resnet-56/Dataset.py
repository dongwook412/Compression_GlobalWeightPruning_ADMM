from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

def Cifar10():
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train / 255.0
    X_test = X_test / 255.0
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, X_test, Y_train, Y_test


def Cifar10_2():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = np.array(y_train).reshape(-1)
    y_test = np.array(y_test).reshape(-1)

    std = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 1, 3))
    mean = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 1, 3))
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std


    x_val = x_train[45000:50000]
    y_val = y_train[45000:50000]

    x_train = x_train[0:45000]
    y_train = y_train[0:45000]
    return x_train, x_val, x_test, y_train, y_val, y_test
