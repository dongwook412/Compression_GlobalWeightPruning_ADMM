from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D

def LeNet5(train_shape, num_class):
    model = Sequential()
    model.add(Conv2D(32, kernel_size = 5, padding="same", input_shape=train_shape, use_bias=False))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(strides=2))
    model.add(Conv2D(48, kernel_size = 5, padding="same", use_bias=False))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, use_bias=False))
    model.add(Activation("relu"))
    model.add(Dense(84, use_bias=False))
    model.add(Activation("relu"))
    model.add(Dense(num_class, use_bias=False))
    model.add(Activation("softmax"))
    return model

