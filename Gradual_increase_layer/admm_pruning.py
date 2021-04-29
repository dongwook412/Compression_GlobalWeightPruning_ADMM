from absl import app
import os
import shutil
import tensorflow as tf
import numpy as np
from Dataset import MNIST
from Model import CNN
from admm_utills import train_step, apply_prune, make_dict, my_projection

# flags.DEFINE_integer('k_step', 10, 'ADMM step number')
# flags.DEFINE_integer('epochs', 30, 'ADMM step(W update) training epoch')
# flags.DEFINE_integer('retraining_epochs', 10, 'After ADMM step, retraining epoch')
# flags.DEFINE_integer('steps_per_epoch', 32, 'After ADMM step, retraining epoch')
# flags.DEFINE_float('learning_rate', 0.001, 'After ADMM step, retraining epoch')
# flags.DEFINE_string('data', 'mnist', 'data')

k_step = 40
epochs = 20
retraining_epochs = 150
batch_size = 128
learning_rate = 0.001
learning_rate_retraining = 0.001

def admm_pruning(rho, p_lambda, all_percent, absolute_path, n_layer = 1):
    #####################################################################################################################################################
    # 데이터 불러오기 및 전처리 ########################################################################################################################## 
    #####################################################################################################################################################
    X_train, X_test, Y_train, Y_test = MNIST()
    
    # model에 대입할 변수들 저장
    train_shape = X_train.shape[1:]
    num_class = Y_train.shape[1]
    if len(train_shape) == 2: # 이미지가 흑백일 때는 차원을 하나 늘려줘야 함.
        X_train = X_train.reshape((-1, train_shape[0], train_shape[1], 1))
        X_test = X_test.reshape((-1, train_shape[0], train_shape[1], 1))
        train_shape = X_train.shape[1:] 

    # # Test용
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
    path = '/rho_{}_lambda_{}_all_percent_{}_kstep_{}_current'.format(rho, p_lambda, all_percent, k_step) # 상세 경로 
    absolute_path = absolute_path + f'/layer_{n_layer}'
    
    # logdir = f"./data/log_{n_layer}"
    logdir = f"./data/log_{n_layer}"

    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64) # step 개수 count
    total_steps = len(X_train)//batch_size+1 # ADMM에서 k=1-step당 step 수
    retraining_total_steps = (len(X_train)//batch_size+1)*retraining_epochs
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # admm_step에 사용
    optimizer_retraining = tf.keras.optimizers.Adam(learning_rate=learning_rate_retraining) # retraining에 사용(위의 optimizer를 사용하였더니 기존 정보를 참고하는듯?)

    if os.path.exists(logdir): shutil.rmtree(logdir)
    writer = tf.summary.create_file_writer(logdir) # metric에 관한 log 저장
    #####################################################################################################################################################
   
    
   
    
    #####################################################################################################################################################
    # 모델 불러오기 ######################################################################################################################################
    #####################################################################################################################################################
    model = CNN(train_shape, num_class, n_layer)
    model.summary()
    model.load_weights(absolute_path+'/train_step/weights')
    layers = model.layers
    #####################################################################################################################################################
    
    
    
    
    #####################################################################################################################################################
    # ADMM ##############################################################################################################################################
    #####################################################################################################################################################
    Z_dict, U_dict = make_dict(model, all_percent) # ADMM을 적용하기 위한 Z, U 값의 초기화(Z는 Weight에서 projection시킨 결과, U는 0 값으로 사용)

    ## batch로 나눠줌
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size)
    # ADMM step
    for k in range(k_step): # k-step(ADMM step 개수)
        for epoch in range(epochs): # Weight 학습
            global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  # step 개수 count
            #print('k-step : {},  epoch : {}'.format(k+1, epoch + 1))
            for image_data, target in train_dataset:
                global_steps = train_step(writer, model, image_data, target, X_val, Y_val, optimizer, k_step, epochs, total_steps, global_steps, retraining_total_steps, rho, p_lambda, Z_dict, U_dict, is_admm=True, k=k, epoch=epoch) # 현재 Z,U가지고 W를 학습, admm 학습 과정이므로 is_admm = True

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

    dict_nzidx = apply_prune(model, W_dict, all_percent)  # True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
    save_path = absolute_path + '/prune_step' + path
    
    os.makedirs(save_path)
    model.save_weights(save_path + '/weights')

    # test data 대입하여 accuracy 저장
    acc = sum(np.argmax(model.predict(X_test), axis=1) == np.argmax(Y_test, axis=1)) / len(Y_test)
    f = open(save_path + "/accuracy.txt", 'w')
    f.write(str(acc))
    f.close()
    #####################################################################################################################################################
    
    
    
    
    #####################################################################################################################################################
    # Retraining ########################################################################################################################################
    #####################################################################################################################################################
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)  # step 개수 count. retraining에서 다시 시작
    total_steps = retraining_total_steps  # Retraining일 때의 전체 step 수

    for epoch in range(retraining_epochs):  # W 학습
        print('epoch : {}'.format(epoch + 1))
        for image_data, target in train_dataset:
            global_steps = train_step(writer, model, image_data, target, X_val, Y_val, optimizer_retraining, k_step, epochs, total_steps, global_steps, retraining_total_steps, rho, p_lambda, is_admm=False, dict_nzidx=dict_nzidx)  # Retraining이므로 is_admm=False

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

    for i in [2, 4, 6, 8, 10]:
        tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow. 
        admm_pruning(0.005, 0.005, 90, '/home/hbdw/바탕화면/weight_pruning_admm/Result/weights/Gradual/MNIST',n_layer = i)
    
    tf.keras.backend.clear_session() # When start the train function, initialize the tensorflow. 

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
