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


retraining_epochs = 1
batch_size = 1024
learning_rate = 0.001
absolute_path = '/home/ubuntu/weight_pruning_admm/Result/weights/Alexnet/ImageNet_95.3_10_10_30'
path = '/rho:0.001,lambda:0.005,all_percent:95.3'

weights_dir = absolute_path+'/admm_step'+path

def admm_retraining(all_percent):
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

    logdir = "./data/log"
    total_n_data = 1278567
    retraining_total_steps = total_n_data//batch_size*retraining_epochs

    optimizer_retraining = tf.keras.optimizers.SGD(lr=learning_rate, decay=0.0005, momentum=0.9, nesterov=True)

    def acc(model, n_batch):
        result = 0
        for i in range(n_batch):
            batch_data = validation_generator.next()
            pred_data = model.predict(batch_data[0])
            result += sum(tf.keras.metrics.categorical_accuracy(batch_data[1], pred_data))/len(pred_data)
        return result/n_batch

    #####################################################################################################################################################


    #####################################################################################################################################################
    # 모델 불러오기 #####################################################################################################################################
    #####################################################################################################################################################
    model = Alexnet(1000)
    model.load_weights(weights_dir+'/weights')
    layers = model.layers
    #####################################################################################################################################################


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
    #acc = model.evaluate(validation_generator, steps=1, workers=6)[1]
    acc = acc(model, 10)
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################


    #####################################################################################################################################################
    # Retraining ########################################################################################################################################
    #####################################################################################################################################################
    print('############ Retraining ############')


    total_steps = 2 # Retraining일 때의 전체 step 수


    def retrain_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            total_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(pred_result, target))

        gradients = tape.gradient(total_loss, model.trainable_weights)

        for i, trainable_weight in enumerate(model.trainable_weights):
            name = trainable_weight.name.split('/')[0]
            is_kernel = 'kernel' in trainable_weight.name.split('/')[1]
            if ('conv' in name or 'fc' in name) and is_kernel:
                gradients[i] = tf.multiply(tf.cast(tf.constant(dict_nzidx[name][0]), tf.float32), gradients[i])

        optimizer_retraining.apply_gradients(zip(gradients, model.trainable_weights))
        return total_loss

    # acc_list = []
    for epoch in range(retraining_epochs):
        for global_steps in range(total_steps):
            data = train_generator.next()
            image_data, target = data[0], data[1]
            total_loss = retrain_step(image_data, target)
            print("=> STEP %4d/%4d    total_loss : %4.2f" % (global_steps + 1, total_steps, total_loss))

            # acc = model.evaluate(validation_generator, steps=10, workers=6)[1]
            # acc_list.append(acc)
            # if acc >= 0.57:
            #     break
            #
            # if max(acc_list[-10:]) > acc:
            #     break


    save_path = absolute_path + '/retraining_step' + path
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')


    def acc2(model, n_batch):
        result = 0
        for i in range(n_batch):
            batch_data = validation_generator.next()
            pred_data = model.predict(batch_data[0])
            result += sum(tf.keras.metrics.categorical_accuracy(batch_data[1], pred_data))/len(pred_data)
        return result/n_batch

    # test data 대입하여 accuracy 저장
    acc = acc2(model, 10)
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

    admm_retraining(95.3)
    tf.keras.backend.clear_session()  # When start the train function, initialize the tensorflow.


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass