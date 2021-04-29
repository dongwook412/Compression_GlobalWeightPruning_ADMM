from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np

def MNIST():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test / 255.0
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    return X_train, X_test, Y_train, Y_test


def MNIST2():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = X_test / 255.0
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    n = int(len(X_train)/5)
    
    X_val = X_train[:n]
    Y_val = Y_train[:n]
    X_train = X_train[n:]
    Y_train = Y_train[n:]
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
