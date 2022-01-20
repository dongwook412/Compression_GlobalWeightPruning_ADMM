# -*- coding: utf-8 -*-
# from absl import app, flags, logging
# from absl.flags import FLAGS
from absl import app, logging
import os
import tensorflow as tf
import numpy as np
from functools import reduce

def make_dict(model, all_percent):
    """
    Z, U 초기값 만드는 과정
    Model weight들을 update하는 것이 아니므로 layer.get_weights()를 통해 값만 가져옴
    :param model: Layer Weight를 가져오기 위한 parameter
    :return Z_dict, U_dict: Weight Pruning에 사용되는 Parameter(초기값)
    """
    layers = model.layers
    Z_dict = {}
    U_dict = {}
    for layer in layers:
        if 'conv2d' in layer.name:

            Z_dict[layer.name] = layer.get_weights()[0] # 우리가 알고있는 커널크기(4차원, numpy)에서 list 하나로 씌워진 형태라서 나중에 다시 해줄 예정
            U_dict[layer.name] = [np.zeros_like(layer.get_weights()[0])]
    Z_dict= my_projection(Z_dict, all_percent) # my_projection에 들어갈 때는 list(5차원) 벗기고 들어감(numpy(4차원)로만 들어가게)
    return Z_dict, U_dict

def train_step(writer, model, image_data, target, X_val, Y_val, optimizer, k_step, epochs, total_steps, global_steps, retraining_total_steps, rho, Z_dict=None, U_dict=None, is_admm =True, dict_nzidx=None, k=0, epoch=0): # W 학습
        """
        Weight 최적화
        :param image_data: Input
        :param target: Output
        :param Z_dict: ADMM step에 사용되는 varaible(Weight가 Z_dict-U_dict과 유사하게 훈련하도록 사용. Z_dict은 projection시킨 값)
        :param U_dict: ADMM step에 사용되는 varaible
        :param is_admm: True일 때 이전 단계의 Z_dict, U_dict을 통해 Weight 최적화, False일 때 gradient의 dict_nzidx를 참고하여 특정 부분을 0으로 바꾸고 Weight 최적화
        :param dict_nzidx: True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
        """
        layers = model.layers
        
        with tf.GradientTape() as tape:
            pred_result = model(image_data, training=True)
            #val_pred_result = model.predict(X_val)
            # 일반적인 분류 loss.
            origin_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target,pred_result, from_logits=True))
            #val_origin_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(val_pred_result, Y_val))
            admm_loss = 0

            if is_admm:  # admm일때만 Z,U를 통해 최적화, retrain일때는 origin_loss만 사용
                for i, layer in enumerate(layers):
                    if 'conv2d' in layer.name:
                        admm_loss = admm_loss + tf.nn.l2_loss([layer.weights[0] - Z_dict[layer.name][0] +
                                                               U_dict[layer.name][
                                                                   0]])  # 7번식의 W-Z+U. get_weights는 값만 불러오는듯. layer.weights는 모델의 weight를 참조

            # 7번식 loss 완료
            total_loss = origin_loss+ rho*admm_loss # 각 parameter를 곱해줌(lambda, rho)
            #val_loss = val_origin_loss + p_lambda * weight_loss + rho * admm_loss  # 각 parameter를 곱해줌(lambda, rho)
            #total_loss = origin_loss + rho * admm_loss  # 각 parameter를 곱해줌(lambda, rho). admm이 아닐때는 admm_loss가 0값을 가짐(origin_loss만 남음)
            gradients = tape.gradient(total_loss, model.trainable_weights)  # 기울기 계산

            if not is_admm:  # retraining step에서만 쓰임. gradient들을 projection시킴(기존 코드에서 apply_prune_on_grads와 같음)
                # gradients: 각 layer의 weight에 대한 gradient [5차원(gradients[i]로 부르면 dict_nzidx[name][0]와 형태 같음)]
                for i, trainable_weight in enumerate(model.trainable_weights):  # model.trainable_weights는 trainable한 layer들의 이름과 weight 정보를 가지고 있음
                    name = trainable_weight.name.split('/')[0]  # conv2d/kernel:0 형식으로 나와서 split
                    if 'conv2d' in name:
                        gradients[i] = tf.multiply(tf.cast(tf.constant(dict_nzidx[name][0]), tf.float32), gradients[i])  # dict_nzidx는 0인 부분이 False인 형태이므로 이를 0.0으로 변경(True는 1.0)하고 gradients와 곱해줌

            if is_admm:
                optimizer.apply_gradients(zip(gradients, model.trainable_weights))  # admm-step에 대한 optimizer에 기울기 적용
            else:
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_weights))  # retraining에 대한 optimizer에 기울기 적용
                
            if global_steps%100==0 or global_steps==total_steps or global_steps==retraining_total_steps: # 출력 정도 조절
                #val_pred_result = model.predict(X_val)
                val_acc = sum(np.argmax(model.predict(X_val), axis=1) == Y_val) / len(Y_val)
                #val_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(val_pred_result, Y_val))
                if is_admm:
                    # admm step
                    tf.print("=> [k-step : %d/%d, epoch : %d/%d]  STEP %4d/%4d   admm_loss: %8.6f  origin_loss: %4.2f  total_loss: %4.2f  val_acc: %4.2f" % (
                                 k + 1, k_step, epoch + 1, epochs, global_steps, total_steps,
                                 admm_loss, origin_loss, total_loss, val_acc))
                    


                else:
                    # retraining step
                    tf.print("=> STEP %4d/%4d  total_loss: %4.2f, val_acc: %4.2f, lr: %10.10f" % (
                                 global_steps, retraining_total_steps, total_loss, val_acc, optimizer._decayed_lr('float32').numpy()))
                    

            # update learning rate
            global_steps.assign_add(1)

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/admm_loss", admm_loss, step=global_steps)
            writer.flush()
        return global_steps, model
            
def apply_prune(model, W_dict, all_percent):
    """
    위에 있는 my_projection과 거의 동일
    - W의 shape를 저장하고, W를 모두 펴준 후 concatenate 진행하고 all_percent만큼 pruning
    - 이후 저장된 shape를 통해 원래 형태로 reshape
    W_dict을 받아서 all_percent만큼 pruning하고 이를 set_weights를 통해 model weight update
    :param W_dict: key(제고하고자 하는 layer name), value(해당하는 layer의 weight)
    :param all_percent: 제거 비율
    :return dict_nzidx: True(0이 아닌 weight)/False(0인 weight) [5차원(list로 씌워진 상태), key는 layer.name으로 구성]
    """
    layers = model.layers
    dict_nzidx = {}

    shape_list = []
    W_reshaped = []
    for k, v in W_dict.items():
        shape_list.append(v.shape)
        W_reshaped.append(v.reshape(-1))

    #W = np.array(list(W_dict.values()))
    #shape_list = list(map(lambda W: W.shape, W))  # W의 shape 저장
    #W_reshaped = list(map(lambda W: W.reshape(-1), W))  # W들을 모두 reshape
    concat = np.concatenate(W_reshaped, axis=0)  # reshape한 것들을 concatenate
    pcen = np.percentile(abs(concat), all_percent) # all_percent에 해당하는 값을 구함
    print("percentile " + str(pcen))
    under_threshold = abs(concat) < pcen # pcen보다 작은 값들을 0으로 만들어줌(projection)
    concat[under_threshold] = 0

    length_list = [] # layer마다 weight들의 개수들을 저장(pruning 이후의 concatenate를 자르기 위해)
    flatten_result = []  # length_list를 이용하여 concatenate된 벡터를 자름
    result = [] # 최종적으로 flatten_result를 기존 형태로 reshape한 결과

    for i in range(len(shape_list)):
        length_list.append(reduce(lambda x, y: x * y, shape_list[i]))

    start = 0
    for length in length_list:
        flatten_result.append(concat[start: length + start])  # concat에서 합친 것을 나눠줌
        start = length + start

    for i, flatten in enumerate(flatten_result):
        result.append(flatten.reshape(shape_list[i]))  # reshape한 것을 되돌려줌

    for i, key, in enumerate(W_dict): # dict_nzidx 생성
        dict_nzidx[key] = [np.array(abs(result[i])) >= pcen]  # mask 완료

    i = 0
    for layer in layers: # pruning 완료된 W_dict를 이용하여 model weight update
        if 'conv2d' in layer.name:
            #연구 부분print('before : {}/{}'.format(np.sum(layer.get_weights()[0] == 0), reduce(lambda x, y: x * y, layer.get_weights()[0].shape)))
            layer.set_weights([np.array(result[i])])  # weight를 prune시켜 업데이트
            #연구 부print('after : {}/{} = {}'.format(np.sum(layer.get_weights()[0] == 0), reduce(lambda x, y: x * y, layer.get_weights()[0].shape), np.sum(layer.get_weights()[0] == 0)/reduce(lambda x, y: x * y, layer.get_weights()[0].shape)))
            i += 1

    return dict_nzidx, model

def my_projection(Z_dict, all_percent):
    """
    Z_dict을 all_percent만큼 projection(percent에 해당하는 값(pcen)보다 작을 경우 0으로 만들어줌)
    :param Z_dict: 대응되는 Layer의 Weight값을 4차원 형태로 표현
    :param all_percent: 제거 비율
    :return Z_dict: Projection 완료된 Z_dict(4차원 형태를 list로 씌워 5차원으로 표현)
    """
    shape_list = []
    Z_reshaped = []
    for k,v in Z_dict.items():
        shape_list.append(v.shape)
        Z_reshaped.append(v.reshape(-1))

    #Z = np.array(list(Z_dict.values()))
    #shape_list = list(map(lambda Z: Z.shape, Z))  # Z의 shape 저장
    #Z_reshaped = list(map(lambda Z: Z.reshape(-1), Z))  # Z들을 모두 reshape
    concat = np.concatenate(Z_reshaped, axis=0)  # reshape한 것들을 concatenate
    pcen = np.percentile(abs(concat), all_percent)  # all_percent에 해당하는 값을 구함
    print("percentile " + str(pcen))
    under_threshold = abs(concat) < pcen  # pcen보다 작은 값들을 0으로 만들어줌(projection)
    concat[under_threshold] = 0

    length_list = []  # layer마다 weight들의 개수들을 저장(pruning 이후의 concatenate를 자르기 위해)
    flatten_result = []  # length_list를 이용하여 concatenate된 벡터를 자름
    result = []  # 최종적으로 flatten_result를 기존 형태로 reshape한 결과

    for i in range(len(shape_list)):
        length_list.append(reduce(lambda x, y: x * y, shape_list[i]))

    start = 0
    for length in length_list:
        flatten_result.append(concat[start: length + start])  # concat에서 합친 것을 나눠줌
        start = length + start

    for i, flatten in enumerate(flatten_result):
        result.append(flatten.reshape(shape_list[i]))  # reshape한 것을 되돌려줌

    for i, key, in enumerate(Z_dict):
        Z_dict[key] = [np.array(result[i])]
    return Z_dict # 반환 값은 list로 씌워진 5차원 형태




