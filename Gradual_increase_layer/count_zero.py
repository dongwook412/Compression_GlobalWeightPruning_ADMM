from absl import app
import numpy as np
import tensorflow as tf
from Dataset import MNIST
from Model import CNN

def count_zero(n_layer, directory):    

    #모델 생성
    X_train, X_test, Y_train, Y_test = MNIST()
    
    # model에 대입할 변수들 저장
    train_shape = X_train.shape[1:]
    num_class = Y_train.shape[1]
    
    if len(train_shape) == 2: # 이미지가 흑백일 때는 차원을 하나 늘려줘야 함.
        X_train = X_train.reshape((-1, train_shape[0], train_shape[1], 1))
        X_test = X_test.reshape((-1, train_shape[0], train_shape[1], 1))
        train_shape = X_train.shape[1:] 
    
    model = CNN(train_shape, num_class, n_layer=n_layer)
    model.load_weights(directory)
    layers = model.layers
    
    zero_count = 0
    size = 0
    
    print('---------------------------------------')
    print(f'[{n_layer} layers]')
    for layer in layers:
        if 'conv2d' in layer.name or 'dense' in layer.name:
            print('{} : {}'.format(layer.name, round(np.sum(layer.get_weights()[0]==0)/layer.get_weights()[0].size * 100, 2)))
            zero_count += np.sum(layer.get_weights()[0]==0)
            size += layer.get_weights()[0].size

    print()
    print('zero_count : {}'.format(zero_count))
    print('size : {}'.format(size))
    print('zero_percent : {}'.format(round(zero_count/size * 100, 2)))
    print('---------------------------------------')
    print()
    print()


def main(_argv):
    root_dir = '/home/hbdw/바탕화면/weight_pruning_admm/Result/weights/Gradual/MNIST'
    file_dir = '/retraining_step/rho_0.005_lambda_0.005_all_percent_90_kstep_40_current/weights'
    # for i in range(1, 11):
    for i in [2, 4, 6, 8, 10]:
        directory = root_dir + f'/layer_{i}' + file_dir
        count_zero(i, directory)
        tf.keras.backend.clear_session()
        
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
