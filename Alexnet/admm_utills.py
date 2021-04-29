# -*- coding: utf-8 -*-
# from absl import app, flags, logging
# from absl.flags import FLAGS
from absl import app, logging
import os
import tensorflow as tf
import numpy as np
from functools import reduce

def make_dict(model, all_percent):

    layers = model.layers
    Z_dict = {}
    U_dict = {}
    for layer in layers:
        if 'conv' in layer.name or 'fc' in layer.name:
            Z_dict[layer.name] = layer.get_weights()[0] 
            U_dict[layer.name] = [np.zeros_like(layer.get_weights()[0])]
    Z_dict= my_projection(Z_dict, all_percent) 
    return Z_dict, U_dict

@tf.function
def train_step(writer, model, image_data, target, X_val, Y_val, optimizer, k_step, epochs, total_steps, global_steps, retraining_total_steps, rho, p_lambda, Z_dict=None, U_dict=None, is_admm =True, dict_nzidx=None, k=0, epoch=0):

    layers = model.layers
    with tf.GradientTape() as tape:
        pred_result = model(image_data, training=True)
        #val_pred_result = model.predict(X_val)
        origin_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(pred_result, target))
        #val_origin_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(val_pred_result, Y_val))
        weight_loss = 0
        admm_loss = 0
        if is_admm:
            for i, layer in enumerate(layers):
                if 'conv' in layer.name or 'fc' in layer.name:
                    weight_loss = weight_loss + tf.nn.l2_loss(layer.weights[0])
                    admm_loss = admm_loss + tf.nn.l2_loss([layer.weights[0] - Z_dict[layer.name][0] +
                                                           U_dict[layer.name][
                                                               0]])
        total_loss = origin_loss + p_lambda*weight_loss + rho*admm_loss

    gradients = tape.gradient(total_loss, model.trainable_weights)
    
    if not is_admm:

        for i, trainable_weight in enumerate(model.trainable_weights):
            name = trainable_weight.name.split('/')[0]
            if 'conv' in name or 'fc' in name:
                gradients[i] = tf.multiply(tf.cast(tf.constant(dict_nzidx[name][0]), tf.float32), gradients[i])

    if is_admm:
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    else:
        optimizer.apply_gradients(
            zip(gradients, model.trainable_weights))
    if True: #global_steps%100==0 or global_steps==total_steps or global_steps==retraining_total_steps:
        #val_pred_result = model.predict(X_val)
        #val_acc = sum(np.argmax(model.predict(X_val), axis=1) == np.argmax(Y_val, axis=1)) / len(Y_val)
        val_acc = 0
        #val_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(val_pred_result, Y_val))
        if is_admm:
            # admm step

            print("=> [k-step : %d/%d, epoch : %d/%d]  STEP %4d/%4d" %(k + 1, k_step, epoch + 1, epochs, global_steps, total_steps))
            # tf.print("=> [k-step : %d/%d, epoch : %d/%d]  STEP %4d/%4d   admm_loss: %8.6f  weight_loss: %4.2f  origin_loss: %4.2f  total_loss: %4.2f  val_acc: %5.5f" % (
            #              k + 1, k_step, epoch + 1, epochs, global_steps, total_steps,
            #              admm_loss, weight_loss, origin_loss, total_loss, val_acc))


        else:
            # retraining step
            tf.print("=> STEP %4d/%4d  total_loss: %4.2f, val_acc: %5.5f, lr: %10.10f" % (
                         global_steps, retraining_total_steps, total_loss, val_acc, optimizer._decayed_lr('float32').numpy()))

    # update learning rate
    #global_steps += 1

    # writing summary data
    #with writer.as_default():
    #    tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
    #    tf.summary.scalar("loss/admm_loss", admm_loss, step=global_steps)
    #writer.flush()
    return total_loss#global_steps, model

def apply_prune(model, W_dict, all_percent):

    layers = model.layers
    dict_nzidx = {}

    W = np.array(list(W_dict.values()))
    shape_list = list(map(lambda W: W.shape, W))  
    W_reshaped = list(map(lambda W: W.reshape(-1), W))  
    concat = np.concatenate(W_reshaped, axis=0)  
    pcen = np.percentile(abs(concat), all_percent) 
    print("percentile " + str(pcen))
    under_threshold = abs(concat) < pcen 
    concat[under_threshold] = 0

    length_list = []
    flatten_result = []  
    result = []

    for i in range(len(shape_list)):
        length_list.append(reduce(lambda x, y: x * y, shape_list[i]))

    start = 0
    for length in length_list:
        flatten_result.append(concat[start: length + start])  
        start = length + start

    for i, flatten in enumerate(flatten_result):
        result.append(flatten.reshape(shape_list[i])) 

    for i, key, in enumerate(W_dict): 
        dict_nzidx[key] = [np.array(abs(result[i])) >= pcen]

    i = 0
    for layer in layers: 
        if 'conv' in layer.name or 'fc' in layer.name:
            layer.set_weights([np.array(result[i]), layer.get_weights()[1]])
            #layer.weights[0] = np.array(result[i])

            i += 1

    return dict_nzidx, model

def my_projection(Z_dict, all_percent=10):

    
    Z = np.array(list(Z_dict.values()))
    shape_list = list(map(lambda Z: Z.shape, Z))  
    Z_reshaped = list(map(lambda Z: Z.reshape(-1), Z))  
    concat = np.concatenate(Z_reshaped, axis=0)  
    pcen = np.percentile(abs(concat), all_percent) 
    print("percentile " + str(pcen))
    under_threshold = abs(concat) < pcen 
    concat[under_threshold] = 0

    length_list = []  
    flatten_result = []  
    result = []  

    for i in range(len(shape_list)):
        length_list.append(reduce(lambda x, y: x * y, shape_list[i]))

    start = 0
    for length in length_list:
        flatten_result.append(concat[start: length + start]) 
        start = length + start

    for i, flatten in enumerate(flatten_result):
        result.append(flatten.reshape(shape_list[i])) 

    for i, key, in enumerate(Z_dict):
        Z_dict[key] = [np.array(result[i])]
    return Z_dict 



