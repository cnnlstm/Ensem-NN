from __future__ import division
import sys
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.regularizers import l2,l1
import numpy as np
import sys
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, Lambda
from keras.layers.merge import add,concatenate
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.activations import relu
import tensorflow as tf
from functools import partial
import pdb
import os
import h5py
os.environ['CUDA_VISIBLE_DEVICES']="5"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
from keras.utils import plot_model
np.random.seed(seed=1234)


loss = 'categorical_crossentropy'
lr = 0.01
momentum = 0.9


out_dir_name = 'ordinary_3'
activation = "relu"
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)
dropout = 0.0
reg = l1(1.e-4)

batch_size = 128
epochs = 600
verbose = 1
shuffle = 1
train_x_mean = 0.5909#14884877
max_len = 300

feat_dim = 150


n_classes = 60


data_root = "/data2/attention_1/"
train_list = "/home/siat/xyy/3d/cross_subject/subject_train_video_list.txt"
test_list = "/home/siat/xyy/3d/cross_subject/subject_test_video_list.txt"
scale = 1
train_lines = []
f = open(train_list,"r")  
lines_ = f.readlines()
for i in range(len(lines_)//scale):
  train_lines.append(lines_[i*scale])
train_lines = sorted(train_lines)
np.random.shuffle(train_lines)
len_train = len(train_lines)
print ("successfully,len(train_list)",len_train)


h = open(test_list,"r")  
test_lines_ = h.readlines()
test_lines = [] 
for i in range(len(test_lines_)//scale):
  test_lines.append(test_lines_[i*scale])
test_lines = sorted(test_lines)
np.random.shuffle(test_lines)
len_test = len(test_lines)
print ("successfully,len(test_list)",len_test)


def fast_list2arr(data, offset=None, dtype=None):
    num = len(data)
    out_data = np.empty((num,)+data[0].shape, dtype=dtype if dtype else data[0].dtype)
    for i in xrange(num):
        out_data[i] = data[i] - offset if offset else data[i]
    return out_data


def read_h5(path):
    #print path
    g = h5py.File(path)
    data = g['data']
    data = data[:]
    label = g['label']
    label = label[:]
    return data,label


def nturgbd_train_datagen():
    x_yed = []
    y_yed = []
    batch_count = 0
    while 1:
        for line in train_lines:
            data,label = read_h5(data_root+line[:-1]+".h5")
            nonzeros = np.where(np.array([np.sum(data[i])>0 for i in range(0,data.shape[0])])==False)[0]
            if len(nonzeros) == 0:
              last_time = 0
            else:
              last_time = nonzeros[0]
            data[:last_time] = data[:last_time] - train_x_mean
            x_yed.append(data)
            y_yed.append(label)
            batch_count += 1
            if batch_count == batch_size:
                X = fast_list2arr(x_yed)
                Y = fast_list2arr(y_yed)
                x_yed = []
                y_yed = []
                batch_count = 0
                yield X,Y






def nturgbd_test_datagen():
    x_yed = []
    y_yed = []
    batch_count = 0
    while 1:
        for line in test_lines:
            data,label = read_h5(data_root+line[:-1]+".h5")
            nonzeros = np.where(np.array([np.sum(data[i])>0 for i in range(0,data.shape[0])])==False)[0]
            if len(nonzeros) == 0:
              last_time = 0
            else:
              last_time = nonzeros[0]
            data[:last_time] = data[:last_time] - train_x_mean
            x_yed.append(data)
            y_yed.append(label)
            batch_count += 1
            if batch_count == batch_size:
                X = fast_list2arr(x_yed)
                Y = fast_list2arr(y_yed)
                x_yed = []
                y_yed = []
                batch_count = 0
                yield X,Y



def resnet(
           n_classes, 
           feat_dim,
           max_len,
           dropout=dropout,
           kernel_regularizer=l1(1.e-4),
           activation="relu"):

  config = [ 
             [(1,8,64)],
             [(1,8,64)],
             [(1,8,64)],
             [(2,8,128)],
             [(1,8,128)],
             [(1,8,128)],
             [(2,8,256)],
             [(1,8,256)],
             [(1,8,256)],
           ]
  initial_stride = 1
  initial_filter_dim = 8
  initial_num = 64
  res_s = []
  res_num = [2,5,8]
  
  input = Input(shape=(max_len,feat_dim))
  model = input

  model = Conv1D(initial_num, 
                 initial_filter_dim,
                 strides=initial_stride,
                 padding="same",
                 kernel_initializer="he_normal",
                 kernel_regularizer=kernel_regularizer)(model)
  
  for depth in range(0,len(config)):
    res_s.append(model)
    for stride,filter_dim,num in config[depth]:

      model = Conv1D(num, 
                   filter_dim,
                   strides=stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(model)

      model = BatchNormalization(axis=2)(model)
      model = Activation(activation)(model)
      model = Dropout(dropout)(model)
      print model
      if depth in res_num:
        print len(res_s)
        res = res_s[-3]
        res_shape = K.int_shape(res)
        model_shape = K.int_shape(model)
        if res_shape[2] != model_shape[2]:
          res = Conv1D(num, 
                       1,
                       strides=2,
                       padding="same",
                       kernel_initializer="he_normal",
                       kernel_regularizer=kernel_regularizer)(res)
        model = add([res,model])

  bn = BatchNormalization(axis=2)(model)
  model = Activation(activation)(bn)


  pool_window_shape = K.int_shape(model)
  gap = AveragePooling1D(pool_window_shape[1],
                           strides=1)(model)
  flatten = Flatten()(gap)
  dense = Dense(units=n_classes, 
                activation="softmax",
                kernel_initializer="he_normal")(flatten)
  model = Model(inputs=input, outputs=dense)
  return model



def train():
  model = resnet(n_classes,feat_dim,max_len,)
  plot_model(model, to_file=out_dir_name+'.png')
  model.compile(loss=loss,
                optimizer=optimizer,
                metrics=['accuracy'])

  if not os.path.exists('weights/'+out_dir_name):
    os.makedirs('weights/'+out_dir_name) 
  weight_path = 'weights/'+out_dir_name+'/{epoch:03d}_{val_acc:0.3f}.hdf5'
  checkpoint = ModelCheckpoint(weight_path, 
                               monitor='val_acc', 
                               verbose=1, 
                               save_best_only=True, mode='max')
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                factor=0.1,
                                patience=10, 
                                verbose=1,
                                mode='auto',
                                cooldown=3,
                                min_lr=0.0001)

  callbacks_list = [checkpoint,reduce_lr]
  

  model.fit_generator(nturgbd_train_datagen(),
                      steps_per_epoch=len_train/batch_size+1,
                      epochs=epochs,
                      verbose=1,
                      callbacks=callbacks_list,
                      validation_data=nturgbd_test_datagen(),
                      validation_steps=len_test/batch_size+1,
                      initial_epoch=0
                     )



if __name__ == "__main__":
  train()
