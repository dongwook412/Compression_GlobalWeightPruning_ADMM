from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import Input, Model
import tensorflow as tf


def Alexnet(num_class):
    inputs = Input((227,227,3))
    x = inputs
    
    x = Conv2D(96, (11, 11), padding='valid', activation='relu', strides = (4,4), name='conv1')(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    
    x = Conv2D(256, (5, 5), padding='same', activation='relu', strides = (1,1), groups=2, name='conv2')(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    
    x = Conv2D(384, (3, 3), padding='same', activation='relu', strides = (1,1), name='conv3')(x)
    
    x = Conv2D(384, (3, 3), padding='same', activation='relu', strides = (1,1), groups=2, name='conv4')(x)
    
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu', strides = (1,1), groups=2, name='conv5')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    
    x = Flatten()(x)
    
    x = Dense(4096, activation = 'relu', name='fc6')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4096, activation = 'relu', name='fc7')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation = 'softmax', name='fc8')(x)
    outputs = x 
    
    model = Model(inputs, outputs)

    return model

def Alexnet2(num_class):
    inputs = Input((227,227,3))
    x = inputs
    
    x = Conv2D(96, (11, 11), padding='valid', activation='relu', strides = (4,4), name='conv1', use_bias=False)(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    
    x = Conv2D(256, (5, 5), padding='same', activation='relu', strides = (1,1), groups=2, name='conv2', use_bias=False)(x)
    x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=2e-05, beta=0.75, bias=1)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    
    x = Conv2D(384, (3, 3), padding='same', activation='relu', strides = (1,1), name='conv3', use_bias=False)(x)
    
    x = Conv2D(384, (3, 3), padding='same', activation='relu', strides = (1,1), groups=2, name='conv4', use_bias=False)(x)
    
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu', strides = (1,1), groups=2, name='conv5', use_bias=False)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    
    x = Flatten()(x)
    
    x = Dense(4096, activation = 'relu', name='fc6', use_bias=False)(x)
    x = Dropout(0.5)(x)
    
    x = Dense(4096, activation = 'relu', name='fc7', use_bias=False)(x)
    x = Dropout(0.5)(x)
    x = Dense(num_class, activation = 'softmax', name='fc8', use_bias=False)(x)
    outputs = x 
    
    model = Model(inputs, outputs)

    return model
