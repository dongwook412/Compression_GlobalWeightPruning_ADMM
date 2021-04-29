from absl import app
import numpy as np
from Model import Alexnet

def count_zero(directory):    
    '''
    #모델 생성
    X_train, X_test, Y_train, Y_test = MNIST()
    
    # model에 대입할 변수들 저장
    train_shape = X_train.shape[1:]
    num_class = Y_train.shape[1]
    
    if len(train_shape) == 2: # 이미지가 흑백일 때는 차원을 하나 늘려줘야 함.
        X_train = X_train.reshape((-1, train_shape[0], train_shape[1], 1))
        X_test = X_test.reshape((-1, train_shape[0], train_shape[1], 1))
        train_shape = X_train.shape[1:] 
    '''
    model = Alexnet(1000)
    model.load_weights(directory)
    layers = model.layers
    
    zero_count = 0
    size = 0
    
    print('---------------------------------------')
    for layer in layers:
        if 'conv' in layer.name or 'fc' in layer.name:
            print('{} : {}'.format(layer.name, np.sum(layer.get_weights()[0]>0.01)/layer.get_weights()[0].size))
            zero_count += np.sum(layer.get_weights()[0] > 0.01)
            size += layer.get_weights()[0].size

    print()
    print('zero_count : {}'.format(zero_count))
    print('size : {}'.format(size))
    print('zero_percent : {}'.format(zero_count/size))
    print('---------------------------------------')
    print()
    print()


def main(_argv):
    directory = '/home/ubuntu/weight_pruning_admm/Result/weights/Alexnet/ImageNet_95.0_10_10_30/prune_step/rho:0.001,lambda:0.005,all_percent:95.0/weights'
    count_zero(directory)
    directory = '/home/ubuntu/weight_pruning_admm/Result/weights/Alexnet/ImageNet_95.0_10_10_30/admm_step/rho:0.001,lambda:0.005,all_percent:95.0/weights'
    count_zero(directory)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
