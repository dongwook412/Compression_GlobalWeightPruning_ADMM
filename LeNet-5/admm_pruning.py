from absl import app
import os
import shutil
import tensorflow as tf
import numpy as np
from Dataset import MNIST
from Model import LeNet5
from admm_utills import train_step, apply_prune, make_dict, my_projection
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# flags.DEFINE_integer('k_step', 10, 'ADMM step number')
# flags.DEFINE_integer('epochs', 30, 'ADMM step(W update) training epoch')
# flags.DEFINE_integer('retraining_epochs', 10, 'After ADMM step, retraining epoch')
# flags.DEFINE_integer('steps_per_epoch', 32, 'After ADMM step, retraining epoch')
# flags.DEFINE_float('learning_rate', 0.001, 'After ADMM step, retraining epoch')
# flags.DEFINE_string('data', 'mnist', 'data')

k_step = 15
epochs = 15
retraining_epochs = 50
batch_size = 128
learning_rate = 0.001


def admm_pruning(rho, p_lambda, all_percent, absolute_path):
    #####################################################################################################################################################
    # 데이터 불러오기 및 전처리 ########################################################################################################################## 
    #####################################################################################################################################################
    X_train, X_test, Y_train, Y_test = MNIST()
    
    # model에 대입할 변수들 저장
    train_shape = X_train.shape[1:]
    num_class = Y_train.shape[1]

    # Test용
    # X_train = X_train[:1000]
    # X_test = X_test[:200]
    # Y_train = Y_train[:1000]
    # Y_test = Y_test[:200]    

    val_index = X_train.shape[0]//4*3 # 25% 사용
    X_train, X_val = X_train[:val_index], X_train[val_index:]
    Y_train, Y_val = Y_train[:val_index], Y_train[val_index:]
    #####################################################################################################################################################
    
        
    
    
    #####################################################################################################################################################
    # 기본적인 설정 ######################################################################################################################################
    #####################################################################################################################################################
    path = '/rho:{},lambda:{},all_percent:{}'.format(rho, p_lambda, all_percent) # 상세 경로 

    logdir = "./data/log"
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) # step 개수 count
    total_steps = len(X_train)//batch_size+1 # ADMM에서 k=1-step당 step 수
    #retraining_total_steps = (len(X_train)//batch_size+1)*retraining_epochs
    retraining_total_steps = (len(X_train)//batch_size+1) 
    
    optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True) 
    optimizer_retraining = tf.keras.optimizers.SGD(lr=learning_rate, decay=1e-4, momentum=0.9, nesterov=True)
 
    #optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # admm_step에 사용
    #optimizer_retraining = tf.keras.optimizers.Adam(learning_rate=learning_rate) # retraining에 사용(위의 optimizer를 사용하였더니 기존 정보를 참고하는듯?)

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir) # metric에 관한 log 저장
    #####################################################################################################################################################
   
    
   
    
    #####################################################################################################################################################
    # 모델 불러오기 #####################################################################################################################################
    #####################################################################################################################################################
    model = LeNet5(train_shape, num_class)
    #model.summary()
    model.load_weights(absolute_path+'/train_step/weights')
    layers = model.layers
    #####################################################################################################################################################
    
    
    
    
    #####################################################################################################################################################
    # Data augmentation #################################################################################################################################
    #####################################################################################################################################################
    datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(X_train)
    #####################################################################################################################################################
    
    
    
    
    print(f'rho:{rho}, lambda:{p_lambda}')
    #####################################################################################################################################################
    # ADMM ##############################################################################################################################################
    #####################################################################################################################################################
    Z_dict, U_dict = make_dict(model, all_percent) # ADMM을 적용하기 위한 Z, U 값의 초기화(Z는 Weight에서 projection시킨 결과, U는 0 값으로 사용)

    ## batch로 나눠줌
    #train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    # ADMM step
    for k in range(k_step): # k-step(ADMM step 개수)
        for epoch in range(epochs): # Weight 학습
            global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  # step 개수 count
            #print('k-step : {},  epoch : {}'.format(k+1, epoch + 1))
            for image_data, target in datagen.flow(X_train, Y_train, batch_size=batch_size):
            #   for image_data, target in train_dataset:
                global_steps, _ = train_step(writer, model, image_data, target, X_val, Y_val, optimizer, k_step, epochs, total_steps, global_steps, retraining_total_steps, rho, p_lambda, Z_dict, U_dict, is_admm=True, k=k, epoch=epoch) # 현재 Z,U가지고 W를 학습, admm 학습 과정이므로 is_admm = True
                if global_steps > retraining_total_steps:
                    break
                
        for layer in layers: # 학습된 W와 현재 U를 통해 Z를 학습
            if 'conv2d' in layer.name or 'dense' in layer.name:
                Z_dict[layer.name] = layer.get_weights()[0] + U_dict[layer.name][0]  # 우리가 알고있는 커널크기(4차원, numpy)에서 list 하나로 씌워진 형태라서 나중에 다시 해줄 예정
        Z_dict = my_projection(Z_dict)

        for layer in layers: # 학습된 W와 Z 값을 통해 U를 구함
            if 'conv2d' in layer.name or 'dense' in layer.name:
                U_dict[layer.name] = [U_dict[layer.name][0] + layer.get_weights()[0] - Z_dict[layer.name][0]]

    save_path = absolute_path+'/admm_step'+path
    os.makedirs(save_path)
    model.save_weights(save_path+'/weights')

    # test data 대입하여 accuracy 저장
    acc = sum(np.argmax(model.predict(X_test), axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
    f = open(save_path+"/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################
    
    
    
    
    #####################################################################################################################################################
    # Weight pruning ####################################################################################################################################
    #####################################################################################################################################################    
    W_dict = {}  # Weight들을 제거하고자 하는 layer에 대응하여 dictionary 형태로 만듬
    for layer in layers:
        if 'conv2d' in layer.name or 'dense' in layer.name:
            W_dict[layer.name] = layer.get_weights()[0]

    dict_nzidx, _ = apply_prune(model, W_dict, all_percent)  # True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
    save_path = absolute_path + '/prune_step' + path
    
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')

    # test data 대입하여 accuracy 저장
    acc = sum(np.argmax(model.predict(X_test), axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################
    
    
    # print(model.predict(X_test)[0:10])    
    # print(np.argmax(model.predict(X_test), axis=1)[0:1000])
    # print(np.argmax(model.predict(X_test), axis=1)[1000:2000])
    # print(np.argmax(model.predict(X_test), axis=1)[2000:3000])
    # print(np.argmax(model.predict(X_test), axis=1)[3000:4000])
    # print(np.argmax(model.predict(X_test), axis=1)[4000:5000])
    # print(np.argmax(model.predict(X_test), axis=1)[5000:6000])
    # print(np.argmax(model.predict(X_test), axis=1)[6000:7000])
    
    
    
    
    #####################################################################################################################################################
    # Retraining ########################################################################################################################################
    #####################################################################################################################################################
    
    total_steps = retraining_total_steps  # Retraining일 때의 전체 step 수

    for epoch in range(retraining_epochs):  # W 학습
        print('epoch : {}'.format(epoch + 1))
        # for image_data, target in train_dataset:
        global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  # step 개수 count. retraining에서 다시 시작
        for image_data, target in datagen.flow(X_train, Y_train, batch_size=batch_size):
            global_steps, _ = train_step(writer, model, image_data, target, X_val, Y_val, optimizer_retraining, k_step, epochs, total_steps, global_steps, retraining_total_steps, rho, p_lambda, is_admm=False, dict_nzidx=dict_nzidx)  # Retraining이므로 is_admm=False
            if global_steps > retraining_total_steps:
                break
    save_path = absolute_path + '/retraining_step' + path
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')

    # test data 대입하여 accuracy 저장
    acc = sum(np.argmax(model.predict(X_test), axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################
    


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
        # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # def admm_pruning(rho, p_lambda, all_percent, absolute_path):
#    for i in range(1, 11):
#        tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow. 
#        admm_pruning(0.005, 0.0001, 80, '/home/hb/Desktop/ADMM/fc/result/Convolutional2', n_layer = i)

    rho_list = [0.001, 0.003, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04]
    p_lambda_list = [0.005, 0.007, 0.01, 0.03, 0.05, 0.07, 0.1]
    #rho_list = [0.03, 0.04]
    #p_lambda_list = [0.005, 0.007]
    for rho in rho_list:
        for p in p_lambda_list:
            admm_pruning(rho, p, 99.0, '/home/hbdw/바탕화면/weight_pruning_admm/Result/weights/LeNet5/MNIST_MORE_99.0_15_15_50')
            tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow. 

        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
