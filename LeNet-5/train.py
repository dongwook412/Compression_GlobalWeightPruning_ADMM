from absl import app
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# from Dataset import Cifar10
from Dataset import MNIST2
from Model import LeNet5
from tensorflow.keras.preprocessing.image import ImageDataGenerator

root_dir = '/home/hbdw/바탕화면/weight_pruning_admm/Result/weights/'
detail_dir = 'LeNet5/MNIST/train_step/'
training_epochs = 500
batch_size = 128
learning_rate = 0.1

def train():
    tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow. 
    # X_train, X_test, Y_train, Y_test = Cifar10()
    X_train, X_val, X_test, Y_train, Y_val, Y_test = MNIST2()
    # EXPERMIENT
    #X_train = X_train[:1000]
    #X_test = X_test[:200]
    #Y_train = Y_train[:1000]
    #Y_test = Y_test[:200]
    # ##

    # model에 대입할 변수들 저장
    train_shape = X_train.shape[1:]
    num_class = Y_train.shape[1]


    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1)  # randomly shift images vertically (fraction of total height)

    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)


    model = LeNet5(train_shape, num_class)
    model.summary()

    # callback(learning rate 조정 및 조기 종료)
    # reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    earlystopping = EarlyStopping(monitor='val_loss',  # 모니터 기준 설정 (val loss)
                                  patience=30,  # patience회 Epoch동안 개선되지 않는다면 종료
                                  )

    def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // 20))
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    ## W 초기값 훈련
    sgd = tf.keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=X_train.shape[0] // batch_size, epochs=training_epochs, validation_data=(X_val, Y_val),callbacks=[reduce_lr, earlystopping])
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.fit(X_train, Y_train, epochs=training_epochs, batch_size=batch_size, validation_split=0.25, callbacks=[reduceLR, earlystopping])
    # model.fit_generator(datagen.flow(X_train, Y_train, batch_size = batch_size), steps_per_epoch=len(X_train) / batch_size, epochs=training_epochs, validation_data=(X_test, Y_test), callbacks=[reduceLR, earlystopping])
    model.save_weights(root_dir+detail_dir+'weights')

    # test data 대입하여 accuracy 저장
    acc = sum(np.argmax(model.predict(X_test), axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
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
