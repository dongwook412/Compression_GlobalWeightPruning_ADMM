from absl import app
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
# from Dataset import Cifar10
from Dataset import Cifar10_2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from Model import cifar_resnet56

root_dir = '/home/ubuntu/weight_pruning_admm/Result/weights/'
detail_dir = 'Resnet56/cifar10/train_step/'
training_epochs = 10#182
batch_size = 128
learning_rate = 0.00001
weight_decay = 1e-4

def train():
    tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow. 
    # X_train, X_test, Y_train, Y_test = Cifar10()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = Cifar10_2()


    print(f'X_train.shape : {X_train.shape}')
    print(f'X_val.shape : {X_val.shape}')
    print(f'X_test.shape : {X_test.shape}')
    print(f'Y_train.shape : {Y_train.shape}')
    print(f'Y_val.shape : {Y_val.shape}')
    print(f'Y_test.shape : {Y_test.shape}')

    # EXPERMIENT
    # X_train = X_train[:1000]
    # X_test = X_test[:200]
    # Y_train = Y_train[:1000]
    # Y_test = Y_test[:200]
    ##


    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)

    model = cifar_resnet56(load_weights=True)
    model.summary()

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[400, 32000, 48000],
                                                                    values=[0.01, 0.1, 0.01, 0.001])
    #optimizer = tf.keras.optimizers.SGD(schedule, momentum=0.9)
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001, decay=1e-4,  momentum=0.9)
    model.compile(optimizer, loss_fn, metrics=['accuracy'])


    training_steps = 64000
    validation_interval = 2000

    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=validation_interval, epochs=training_steps // validation_interval,
    #                     validation_data=(X_val, Y_val), workers=4)
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=50,
    #                     epochs=1,
    #                     validation_data=(X_val, Y_val), workers=4)

    #model.save_weights(root_dir+detail_dir+'weights')

    # test data 대입하여 accuracy 저장
    acc = sum(np.argmax(model.predict(X_test), axis=1) == Y_test) / len(Y_test)
    f = open(root_dir+detail_dir+'accuracy.txt', 'w')
    f.write(str(acc))
    f.close()



def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    train()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
