from absl import app
import os
import shutil
import tensorflow as tf
import numpy as np
from Model import Alexnet
from admm_utills import make_dict, my_projection
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# flags.DEFINE_integer('k_step', 10, 'ADMM step number')
# flags.DEFINE_integer('epochs', 30, 'ADMM step(W update) training epoch')
# flags.DEFINE_integer('retraining_epochs', 10, 'After ADMM step, retraining epoch')
# flags.DEFINE_integer('steps_per_epoch', 32, 'After ADMM step, retraining epoch')
# flags.DEFINE_float('learning_rate', 0.001, 'After ADMM step, retraining epoch')
# flags.DEFINE_string('data', 'mnist', 'data')

k_step = 10
epochs = 10
batch_size = 1024
learning_rate = 0.001


def admm(rho, p_lambda, all_percent, absolute_path):
    #####################################################################################################################################################
    # 데이터 불러오기 및 전처리 ##########################################################################################################################
    #####################################################################################################################################################
    def myFunc(image):
        imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)
        image -= imagenet_mean
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    train_datagen = ImageDataGenerator(
        preprocessing_function=myFunc)
    test_datagen = ImageDataGenerator(
        preprocessing_function=myFunc)

    class_list = []
    for i in range(0, 1000):
        class_list.append(str(i))

    train_generator = train_datagen.flow_from_directory(
        '/home/ubuntu/weight_pruning_admm/data/Imagenet/image/train',
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical', classes=class_list)
    # class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        '/home/ubuntu/weight_pruning_admm/data/Imagenet/image/val',
        target_size=(227, 227),
        batch_size=batch_size,
        class_mode='categorical', classes=class_list)
    # class_mode='binary')
    #####################################################################################################################################################

    #####################################################################################################################################################
    # 기본적인 설정 ######################################################################################################################################
    #####################################################################################################################################################
    strategy = tf.distribute.MirroredStrategy() # Multi GPU

    path = '/rho:{},lambda:{},all_percent:{}'.format(rho, p_lambda, all_percent)  # 상세 경로

    logdir = "./data/log"
    total_n_data = 1278567
    total_steps = total_n_data//batch_size  # len(X_train)//batch_size+1 # ADMM에서 k=1-step당 step 수

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0005, momentum=0.9, nesterov=True)

    #####################################################################################################################################################

    #####################################################################################################################################################
    # 모델 불러오기 및 Compile ###########################################################################################################################
    #####################################################################################################################################################
    def ADMM_iteration_loss(y_true, y_pred): # ADMM에 사용할 loss
        origin_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred))
        weight_loss = 0
        admm_loss = 0

        for i, layer in enumerate(layers):
            if 'conv' in layer.name or 'fc' in layer.name:
                weight_loss = weight_loss + tf.nn.l2_loss(layer.weights[0])
                admm_loss = admm_loss + tf.nn.l2_loss([layer.weights[0] - Z_dict[layer.name][0] +
                                                       U_dict[layer.name][
                                                           0]])

        return origin_loss + rho * weight_loss + p_lambda * admm_loss



    with strategy.scope():
        model = Alexnet(1000)
        weights_dict = np.load('/home/ubuntu/weight_pruning_admm/code/Alexnet/bvlc_alexnet.npy', encoding='bytes',
                               allow_pickle=True).item()
        # model.summary()
        layers = model.layers
        for i, layer in enumerate(layers):
            if 'conv' in layer.name or 'fc' in layer.name:
                layer.set_weights(weights_dict[layer.name])

        model.compile(optimizer=optimizer, loss=ADMM_iteration_loss, metrics=['accuracy'])
    #####################################################################################################################################################

    print(f'rho:{rho}, lambda:{p_lambda}')

    #####################################################################################################################################################
    # ADMM ##############################################################################################################################################
    #####################################################################################################################################################
    print('############ ADMM ############')

    Z_dict, U_dict = make_dict(model,
                               all_percent)  # ADMM을 적용하기 위한 Z, U 값의 초기화(Z는 Weight에서 projection시킨 결과, U는 0 값으로 사용)

    for k in range(k_step):
        # W 학습
        print("[k-step : %d/%d]" % (k+1, k_step))
        model.fit(train_generator, epochs=epochs, steps_per_epoch=total_steps, workers=10)

        for layer in layers:  # 학습된 W와 현재 U를 통해 Z를 학습
            if 'conv' in layer.name or 'fc' in layer.name:
                Z_dict[layer.name] = layer.get_weights()[0] + U_dict[layer.name][
                    0]  # 우리가 알고있는 커널크기(4차원, numpy)에서 list 하나로 씌워진 형태라서 나중에 다시 해줄 예정
        Z_dict = my_projection(Z_dict)

        for layer in layers:  # 학습된 W와 Z 값을 통해 U를 구함
            if 'conv' in layer.name or 'fc' in layer.name:
                U_dict[layer.name] = [U_dict[layer.name][0] + layer.get_weights()[0] - Z_dict[layer.name][0]]

    save_path = absolute_path + '/admm_step' + path
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')

    # test data 대입하여 accuracy 저장
    loss, acc1, acc5 = model.evaluate(validation_generator, steps=10, workers=6)
    f = open(save_path+"/accuracy.txt", 'w')
    f.write('acc1 : ' + str(acc1) + '\n')
    f.write('acc5 : ' + str(acc5))
    f.close()


def main(_argv):
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

    rho_list = [0.001, 0.005]
    p_lambda_list = [0.001, 0.005]

    for rho in rho_list:
        for p in p_lambda_list:
            admm(rho, p, 95.3,'/home/ubuntu/weight_pruning_admm/Result/weights/Alexnet/ImageNet_95.3_10_10_30')
            tf.keras.backend.clear_session()  # When start the train function, initialize the tensorflow.


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass