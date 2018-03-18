# -*- coding: utf-8 -*-

from __future__ import print_function
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras import layers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np
from six.moves import range



# 构建卷积神经网络
def cnn_model(train_data, train_label, test_data, test_label):
    w, h = 128, 192
    model = Sequential()
    # 卷积层 12 * w * h大小
    model.add(Conv2D(
        filters=12,
        kernel_size=(3, 3),
        padding='valid',
        data_format='channels_first',
        input_shape=(1, w, h)))
    model.add(Activation('relu'))  # 激活函数使用修正线性单元
    # 池化层
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='valid'))
    # 卷积层
    model.add(Conv2D(
        filters=24,
        kernel_size=(3, 3),
        padding='valid',
        data_format='channels_first'))
    model.add(Activation('relu'))
    # 池化层
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='valid'))
    model.add(Conv2D(
        filters=48,
        kernel_size=(3, 3),
        padding='valid',
        data_format='channels_first'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        border_mode='valid'))
    model.add(Flatten())
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Activation(LeakyReLU(0.3)))
    model.add(Dropout(0.4))
    model.add(Dense(5, init='normal'))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('----------------training-----------------------')
    model.fit(train_data, train_label, batch_size=20, nb_epoch=50, shuffle=True, show_accuracy=True,
              validation_split=0.1)
    print('----------------testing------------------------')
    loss, accuracy = model.evaluate(test_data, test_label)
    print('\n test loss:', loss)
    print('\n test accuracy', accuracy)
