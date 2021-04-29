from absl import app
import os
import shutil
import tensorflow as tf
import numpy as np
from Model import Alexnet
from admm_utills import apply_prune, make_dict, my_projection
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
retraining_epochs = 2
batch_size = 1024
learning_rate = 0.001


def admm_pruning(rho, p_lambda, all_percent, absolute_path):
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
    path = '/rho:{},lambda:{},all_percent:{}'.format(rho, p_lambda, all_percent)  # 상세 경로

    logdir = "./data/log"
    total_n_data = 1278567
    total_steps = total_n_data//batch_size  # len(X_train)//batch_size+1 # ADMM에서 k=1-step당 step 수
    # retraining_total_steps = (len(X_train)//batch_size+1)*retraining_epochs
    retraining_total_steps = total_n_data//batch_size*retraining_epochs

    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0005, momentum=0.9, nesterov=True)
    optimizer_retraining = tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0005, momentum=0.9, nesterov=True)

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir)  # metric에 관한 log 저장
    #####################################################################################################################################################

    #####################################################################################################################################################
    # 모델 불러오기 #####################################################################################################################################
    #####################################################################################################################################################
    model = Alexnet(1000)
    weights_dict = np.load('/home/ubuntu/weight_pruning_admm/code/Alexnet/bvlc_alexnet.npy', encoding='bytes',
                           allow_pickle=True).item()
    # model.summary()
    layers = model.layers
    for i, layer in enumerate(layers):
        if 'conv' in layer.name or 'fc' in layer.name:
            layer.set_weights(weights_dict[layer.name])

        # 가중치 제대로 불러왔는지 확인
        # if 'conv' in layer.name:
        #     print(f'layer name : {layer.name}')
        #     print(sum(sum(sum(sum(layer.get_weights()[0] != weights_dict[layer.name][0])))))
        #
        # if 'fc' in layer.name:
        #     print(f'layer name : {layer.name}')
        #     print(sum(sum(layer.get_weights()[0] != weights_dict[layer.name][0])))

    # model.load_weights('/home/ubuntu/weight_pruning_admm/code/Alexnet/train_weights/weights')
    layers = model.layers
    #####################################################################################################################################################

    print(f'rho:{rho}, lambda:{p_lambda}')
    #####################################################################################################################################################
    # ADMM ##############################################################################################################################################
    #####################################################################################################################################################
    print('############ ADMM ############')

    Z_dict, U_dict = make_dict(model,
                               all_percent)  # ADMM을 적용하기 위한 Z, U 값의 초기화(Z는 Weight에서 projection시킨 결과, U는 0 값으로 사용)

    def ADMM_iteration_loss(y_true, y_pred):
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

    model.compile(optimizer=optimizer,loss=ADMM_iteration_loss, metrics=['accuracy'])

    for k in range(k_step):
        # W 학습
        print("[k-step : %d/%d]" % (k+1, k_step))
        model.fit(train_generator, epochs=epochs, steps_per_epoch=total_steps, workers=6)

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
    acc = model.evaluate(validation_generator, steps=10, workers=6)[1]
    f = open(save_path+"/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()


    #####################################################################################################################################################
    # Weight pruning ####################################################################################################################################
    #####################################################################################################################################################
    print('############ Weight Pruning ############')

    W_dict = {}  # Weight들을 제거하고자 하는 layer에 대응하여 dictionary 형태로 만듬
    for layer in layers:
        if 'conv' in layer.name or 'fc' in layer.name:
            W_dict[layer.name] = layer.get_weights()[0]

    dict_nzidx, model = apply_prune(model, W_dict,
                                all_percent)  # True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
    save_path = absolute_path + '/prune_step' + path
    layers = model.layers

    zero_count = 0
    size = 0

    print('---------------------------------------')
    for layer in layers:
        if 'conv' in layer.name or 'fc' in layer.name:
            print('{} : {}'.format(layer.name, np.sum(layer.get_weights()[0] == 0) / layer.get_weights()[0].size))
            zero_count += np.sum(layer.get_weights()[0] == 0)
            size += layer.get_weights()[0].size

    print()
    print('zero_count : {}'.format(zero_count))
    print('size : {}'.format(size))
    print('zero_percent : {}'.format(zero_count / size))
    print('---------------------------------------')
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')

    # test data 대입하여 accuracy 저장
    acc = model.evaluate(validation_generator, steps=1, workers=6)[1]
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################


    #####################################################################################################################################################
    # Retraining ########################################################################################################################################
    #####################################################################################################################################################
    print('############ Retraining ############')


    total_steps = 10  # Retraining일 때의 전체 step 수

    @tf.function
    def retrain_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            origin_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(pred_result, target))
            weight_loss = 0
            admm_loss = 0

            total_loss = origin_loss + p_lambda * weight_loss + rho * admm_loss

        gradients = tape.gradient(total_loss, model.trainable_weights)

        for i, trainable_weight in enumerate(model.trainable_weights):
            name = trainable_weight.name.split('/')[0]
            is_kernel = 'kernel' in trainable_weight.name.split('/')[1]
            if ('conv' in name or 'fc' in name) and is_kernel:
                gradients[i] = tf.multiply(tf.cast(tf.constant(dict_nzidx[name][0]), tf.float32), gradients[i])

        optimizer_retraining.apply_gradients(zip(gradients, model.trainable_weights))
        return total_loss

    acc_list = []
    for epoch in range(epochs):
        for global_steps in range(total_steps):
            data = train_generator.next()
            image_data, target = data[0], data[1]
            total_loss = retrain_step(image_data, target)
            print("=> STEP %4d/%4d    total_loss : %4.2f" % (global_steps + 1, total_steps, total_loss))

            acc = model.evaluate(validation_generator, steps=10, workers=6)[1]
            acc_list.append(acc)
            if acc >= 0.57:
                break

            if max(acc_list[-10:]) > acc:
                break


    save_path = absolute_path + '/retraining_step' + path
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')

    # test data 대입하여 accuracy 저장
    acc = model.evaluate(validation_generator, steps=1, workers=6)[1]
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.config.run_functions_eagerly(True)
    # def admm_pruning(rho, p_lambda, all_percent, absolute_path):
    #    for i in range(1, 11):
    #        tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow.
    #        admm_pruning(0.005, 0.0001, 80, '/home/hb/Desktop/ADMM/fc/result/Convolutional2', n_layer = i)

    rho_list = [0.001]
    p_lambda_list = [0.005]
    # rho_list = [0.03, 0.04]
    # p_lambda_list = [0.005, 0.007]
    for rho in rho_list:
        for p in p_lambda_list:
            admm_pruning(rho, p, 95.0,
                         '/home/ubuntu/weight_pruning_admm/Result/weights/Alexnet/ImageNet_95.0_10_10_30')
            tf.keras.backend.clear_session()  # When start the train function, initialize the tensorflow.


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass